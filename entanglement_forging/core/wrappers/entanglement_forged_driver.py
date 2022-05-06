# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""EntanglementForgedDriver."""

import numpy as np
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriver
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicStructureDriverResult,
    ParticleNumber,
    ElectronicEnergy,
)
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)

from qiskit_nature.properties.second_quantization.electronic.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)


class EntanglementForgedDriver(ElectronicStructureDriver):
    """EntanglementForgedDriver."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        hcore: np.ndarray,
        mo_coeff: np.ndarray,
        eri: np.ndarray,
        num_alpha: int,
        num_beta: int,
        nuclear_repulsion_energy: float,
    ):
        """Entanglement forging driver

        Args:
            hcore: hcore integral
            mo_coeff: MO coefficients
            eri: eri integral
            num_alpha: number of alpha electrons
            num_beta: number of beta electrons
            nuclear_repulsion_energy: nuclear repulsion energy
        """
        super().__init__()

        self._hcore = hcore
        self._mo_coeff = mo_coeff
        self._eri = eri
        self._num_alpha = num_alpha
        self._num_beta = num_beta
        self._nuclear_repulsion_energy = nuclear_repulsion_energy

    def run(self) -> ElectronicStructureDriverResult:
        """Returns QMolecule constructed from input data."""
        # Create ParticleNumber property
        particle_number = ParticleNumber(
            self._mo_coeff.shape[0], (self._num_alpha, self._num_beta)
        )

        # Convert one body integrals from AO to MO basis
        one_body_in_mo_basis = np.dot(
            np.dot(np.transpose(self._mo_coeff), self._hcore), self._mo_coeff
        )

        # Convert two-body integrals from AO to MO basis
        dim = self._eri.shape[0]
        two_body_in_mo_basis = np.zeros((dim, dim, dim, dim))
        for a_i in range(dim):
            temp1 = np.einsum("i,i...->...", self._mo_coeff[:, a_i], self._eri)
            for b_i in range(dim):
                temp2 = np.einsum("j,j...->...", self._mo_coeff[:, b_i], temp1)
                temp3 = np.einsum("kc,k...->...c", self._mo_coeff, temp2)
                two_body_in_mo_basis[a_i, b_i, :, :] = np.einsum(
                    "ld,l...c->...cd", self._mo_coeff, temp3
                )

        # Create ElectronicEnergy property with integrals in both bases and the nuclear repulsion energy
        integrals: List[ElectronicIntegrals] = []
        integrals.append(
            OneBodyElectronicIntegrals(ElectronicBasis.AO, (self._hcore, None))
        )
        integrals.append(
            TwoBodyElectronicIntegrals(
                ElectronicBasis.AO, (self._eri, None, None, None)
            )
        )
        integrals.append(
            OneBodyElectronicIntegrals(ElectronicBasis.MO, (one_body_in_mo_basis, None))
        )
        integrals.append(
            TwoBodyElectronicIntegrals(
                ElectronicBasis.MO,
                (
                    two_body_in_mo_basis,
                    two_body_in_mo_basis,
                    two_body_in_mo_basis,
                    None,
                ),
            )
        )
        electronic_energy = ElectronicEnergy(
            integrals, nuclear_repulsion_energy=self._nuclear_repulsion_energy
        )

        # Define the transform from AO to MO
        ele_basis_xform = ElectronicBasisTransform(
            ElectronicBasis.AO, ElectronicBasis.MO, self._mo_coeff
        )

        result = ElectronicStructureDriverResult()
        result.add_property(electronic_energy)
        result.add_property(particle_number)
        result.add_property(ele_basis_xform)

        return result
