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

"""Classical energies."""

import numpy as np
from pyscf import gto, scf, mp, ao2mo, fci

from entanglement_forging.core.cholesky_hamiltonian import get_fermionic_ops_with_cholesky
from entanglement_forging.core.orbitals_to_reduce import OrbitalsToReduce


# pylint: disable=invalid-name
class ClassicalEnergies:  # pylint: disable=too-many-instance-attributes disable=too-few-public-methods
    """Runs classical energy computations.

    Attributes:
        HF (float) = Hartree Fock energy
        MP2 (float) = MP2 Energy
        FCI (float) = Full Configuration Interaction energy
        shift (float) = The sum of the nuclear repulsion energy
                        and the energy shift due to orbital freezing
    """

    def __init__(self, qmolecule, all_orbitals_to_reduce):  # pylint: disable=too-many-locals
        """ Initialize the classical energies.

        Args:
            qmolecule (qiskit_nature.drivers.QMolecule): Molecule data class containing driver
                                                         result.
            all_to_reduce (entanglement_forging.core.orbitals_to_reduce.OrbitalsToReduce):
                All orbitals to be reduced.
            epsilon_cholesky (float): The threshold for the Cholesky decomposition
                                      (typically a number close to 0)
        """
        self.qmolecule = qmolecule
        self.all_orbitals_to_reduce = all_orbitals_to_reduce
        self.orbitals_to_reduce = OrbitalsToReduce(self.all_orbitals_to_reduce, qmolecule)
        self.epsilon_cholesky = 1e-10
        n_electrons = qmolecule.num_molecular_orbitals - len(self.orbitals_to_reduce.all)
        n_alpha_electrons = qmolecule.num_alpha - len(self.orbitals_to_reduce.occupied())
        n_beta_electrons = qmolecule.num_beta - len(self.orbitals_to_reduce.occupied())
        fermionic_op = get_fermionic_ops_with_cholesky(qmolecule.mo_coeff,
                                                       qmolecule.hcore, qmolecule.eri,
                                                       opname='H',
                                                       halve_transformed_h2=True,
                                                       occupied_orbitals_to_reduce=
                                                       self.orbitals_to_reduce.occupied(),
                                                       virtual_orbitals_to_reduce=
                                                       self.orbitals_to_reduce.virtual(),
                                                       epsilon_cholesky=self.epsilon_cholesky)
        # hi - 2D array representing operator coefficients of one-body integrals in the AO basis.
        _, _, freeze_shift, h1, h2 = fermionic_op
        Enuc = freeze_shift + qmolecule.nuclear_repulsion_energy
        # 4D array representing operator coefficients of two-body integrals in the AO basis.
        h2 = 2 * h2
        mol_FC = gto.M(verbose=0)
        mol_FC.charge = 0
        mol_FC.nelectron = n_alpha_electrons + n_beta_electrons
        mol_FC.spin = n_alpha_electrons - n_beta_electrons
        mol_FC.incore_anyway = True
        mol_FC.nao_nr = lambda *args: n_electrons
        mol_FC.energy_nuc = lambda *args: Enuc
        mf_FC = scf.RHF(mol_FC)
        mf_FC.get_hcore = lambda *args: h1
        mf_FC.get_ovlp = lambda *args: np.eye(n_electrons)
        mf_FC._eri = ao2mo.restore(8, h2, n_electrons)
        rho = np.zeros((n_electrons, n_electrons))
        for i in range(n_alpha_electrons):
            rho[i, i] = 2.0

        Ehf = mf_FC.kernel(rho)
        ci_FC = fci.FCI(mf_FC)
        Emp = mp.MP2(mf_FC).kernel()[0]
        Efci, _ = ci_FC.kernel()

        self.HF = Ehf
        self.MP2 = Ehf + Emp
        self.FCI = Efci
        self.shift = Enuc
