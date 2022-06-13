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

"""Forged operator."""

from typing import List

import numpy as np
from qiskit_nature.properties.second_quantization.electronic.bases import (
    ElectronicBasis,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriver

from .cholesky_hamiltonian import get_fermionic_ops_with_cholesky
from .orbitals_to_reduce import OrbitalsToReduce
from ..utils.log import Log


# pylint: disable=too-many-locals,too-few-public-methods
class ForgedOperator:
    """A class for the forged operator.

    Attributes:
        h_1_op (utils.legacy.weighted_pauli_operator.WeightedPauliOperator):
            TODO  # pylint: disable=fixme
        h_chol_ops (list(utils.legacy.weighted_pauli_operator.WeightedPauliOperator)):
            TODO  # pylint: disable=fixme

    E.g. a WeightedPauliOperator can be constructued as follows:
    WeightedPauliOperator([[(0.29994114732731),
                            Pauli(z=[False, False, False],
                                  x=[False, False, False])],
                          [(-0.13390761441581106),
                           Pauli(z=[True, False, False],
                                 x=[False, False, False])]]])
    """

    def __init__(
        self,
        problem: ElectronicStructureProblem,
        all_orbitals_to_reduce: List[int],
        calculate_tensor_cross_terms: bool = False,
    ):
        self.problem = problem
        self.all_orbitals_to_reduce = all_orbitals_to_reduce
        self.orbitals_to_reduce = OrbitalsToReduce(
            self.all_orbitals_to_reduce, self.problem
        )
        self.epsilon_cholesky = 1e-10
        self._calculate_tensor_cross_terms = calculate_tensor_cross_terms

        if isinstance(problem.driver, ElectronicStructureDriver):
            electronic_basis_transform = self.problem.grouped_property.get_property(
                "ElectronicBasisTransform"
            )
            electronic_energy = self.problem.grouped_property.get_property(
                "ElectronicEnergy"
            )
            particle_number = self.problem.grouped_property.get_property(
                "ParticleNumber"
            )

            mo_coeff = electronic_basis_transform.coeff_alpha
            hcore = electronic_energy.get_electronic_integral(
                ElectronicBasis.AO, 1
            )._matrices[0]
            eri = electronic_energy.get_electronic_integral(
                ElectronicBasis.AO, 2
            )._matrices[0]
            num_alpha = particle_number.num_alpha
            num_beta = particle_number.num_beta

        else:
            mo_coeff = problem.driver._mo_coeff
            hcore = problem.driver._hcore
            eri = problem.driver._eri
            num_alpha = problem.driver._num_alpha
            num_beta = problem.driver._num_beta

        fermionic_results = get_fermionic_ops_with_cholesky(
            mo_coeff,
            hcore,
            eri,
            opname="H",
            halve_transformed_h2=True,
            occupied_orbitals_to_reduce=self.orbitals_to_reduce.occupied(),
            virtual_orbitals_to_reduce=self.orbitals_to_reduce.virtual(),
            epsilon_cholesky=self.epsilon_cholesky,
        )
        self.h_1_op, self.h_chol_ops, _, _, _ = fermionic_results

        # assert (
        #    num_alpha == num_beta
        # ), "Currently only supports molecules with equal number of alpha and beta particles."

    def construct(self):
        """Constructs the forged operator by extracting the Pauli operators and weights.

        The forged operator takes the form: Forged Operator = sum_ij w_ij T_ij + sum_ab w_ab S_ij,
        where w_ij and w_ab are coefficients, T_ij and S_ij are operators, and where the first term
        corresponds to the tensor product states while the second term corresponds to the
        superposition states. For more detail, refer to the paper
        TODO: add citation and equation ref

        Returns:
            tuple: a tuple containing:
                - tensor_paulis (list of str): e.g. ['III', 'IIZ', 'IXX', 'IYY', 'IZI',
                                                     'IZZ', 'XXI', 'XZX', 'YYI', 'YZY',
                                                     'ZII', 'ZIZ', 'ZZI']
                - superpos_paulis (list of str): e.g. ['III', 'IIZ', 'IXX', 'IYY', 'IZI',
                                                       'XXI', 'XZX', 'YYI', 'YZY', 'ZII']
                - w_ij (numpy.ndarray): 2D array
                - w_ab (numpy.ndarray): 2D array
        """

        hamiltonian_ops = [self.h_1_op]
        if self.h_chol_ops is not None:
            for chol_op in self.h_chol_ops:
                hamiltonian_ops.append(chol_op)
        op1 = hamiltonian_ops[0]
        cholesky_ops = hamiltonian_ops[1:]
        # The block below calculate the Pauli-pair prefactors W_ij and returns
        # them as a dictionary
        tensor_paulis = set()
        superpos_paulis = set()
        paulis_each_op = [
            {
                label: weight
                for label, weight in op.primitive.to_list()
                if np.abs(weight) > 0
            }
            for op in [op1] + list(cholesky_ops)
        ]
        paulis_each_op = [paulis_each_op[0]] + [p for p in paulis_each_op[1:] if p]
        for op_idx, paulis_this_op in enumerate(paulis_each_op):
            pnames = list(paulis_this_op.keys())
            tensor_paulis.update(pnames)
            if (not self._calculate_tensor_cross_terms) and (op_idx == 0):
                pass
            else:
                superpos_paulis.update(pnames)
        # ensure Identity string is represented since we will need it
        identity_string = "I" * len(pnames[0])
        tensor_paulis.add(identity_string)
        Log.log("num paulis for tensor states:", len(tensor_paulis))
        Log.log("num paulis for superpos states:", len(superpos_paulis))
        tensor_paulis = list(sorted(tensor_paulis))
        superpos_paulis = list(sorted(superpos_paulis))
        pauli_ordering_for_tensor_states = {
            pname: idx for idx, pname in enumerate(tensor_paulis)
        }
        pauli_ordering_for_superpos_states = {
            pname: idx for idx, pname in enumerate(superpos_paulis)
        }
        w_ij = np.zeros((len(tensor_paulis), len(tensor_paulis)))
        w_ab = np.zeros((len(superpos_paulis), len(superpos_paulis)))
        # Processes the non-Cholesky operator
        identity_idx = pauli_ordering_for_tensor_states[identity_string]
        for pname_i, w_i in paulis_each_op[0].items():
            i = pauli_ordering_for_tensor_states[pname_i]
            w_ij[i, identity_idx] += np.real(w_i)  # H_spin-up
            w_ij[identity_idx, i] += np.real(w_i)  # H_spin-down

            # In the special case where bn=bm, we need terms from the
            # single body system represented in the cross terms. Divide
            # by two to account for two independent spins in the Born-Oppenheimer
            # Hamiltonian, and the extra factor of 2 in the tensor pool
            if self._calculate_tensor_cross_terms:
                w_ab[i, identity_idx] += np.real(w_i)
                w_ab[identity_idx, i] += np.real(w_i)
        # Processes the Cholesky operators (indexed by gamma)
        for paulis_this_gamma in paulis_each_op[1:]:
            for pname_1, w_1 in paulis_this_gamma.items():
                i = pauli_ordering_for_tensor_states[pname_1]
                superpos_ordering1 = pauli_ordering_for_superpos_states[
                    pname_1
                ]  # pylint: disable=invalid-name
                for pname_2, w_2 in paulis_this_gamma.items():
                    j = pauli_ordering_for_tensor_states[pname_2]
                    superpos_ordering2 = pauli_ordering_for_superpos_states[
                        pname_2
                    ]  # pylint: disable=invalid-name
                    w_ij[i, j] += np.real(w_1 * w_2)
                    w_ab[superpos_ordering1, superpos_ordering2] += np.real(
                        w_1 * w_2
                    )  # pylint: disable=invalid-name
        return tensor_paulis, superpos_paulis, w_ij, w_ab
