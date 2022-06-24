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

"""Ground state computation using a minimum eigensolver with entanglement forging."""

import time
import warnings
from typing import Iterable, Union, Dict, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms import MinimumEigensolver
from qiskit.circuit import Instruction
from qiskit.opflow import OperatorBase, PauliSumOp
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit_nature import ListOrDictType
from qiskit_nature.algorithms.ground_state_solvers import (
    GroundStateSolver,
    MinimumEigensolverFactory,
)
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization import BaseProblem
from qiskit_nature.problems.second_quantization.electronic import (
    ElectronicStructureProblem,
)

from entanglement_forging.core.classical_energies import ClassicalEnergies
from entanglement_forging.core.forged_operator import ForgedOperator
from entanglement_forging.core.entanglement_forged_config import (
    EntanglementForgedConfig,
)
from entanglement_forging.core.wrappers.entanglement_forged_vqe import (
    EntanglementForgedVQE,
)
from entanglement_forging.core.wrappers.entanglement_forged_vqe_result import (
    EntanglementForgedVQEResult,
)
from entanglement_forging.utils.log import Log


# pylint: disable=too-many-arguments,protected-access
class EntanglementForgedGroundStateSolver(GroundStateSolver):
    """Ground state computation using a minimum solver with entanglement forging.

    Attr:
        transformation: Qubit Operator Transformation
        ansatz (Two):
        orbitals_to_reduce (list of int): Orbital list to be frozen or removed.
    """

    def __init__(
        self,
        qubit_converter: QubitConverter,
        ansatz_u: QuantumCircuit,
        bitstrings_u: Iterable[Iterable[int]],
        config: EntanglementForgedConfig,
        ansatz_v: QuantumCircuit = None,
        bitstrings_v: Iterable[Iterable[int]] = None,
        orbitals_to_reduce: bool = None,
    ):

        # Validate ansatz and bitstrings
        if not ansatz_v:
            ansatz_v = ansatz_u

        if not bitstrings_v:
            bitstrings_v = bitstrings_u

        for ansatz, bitstrings, name in (
            (ansatz_u, bitstrings_u, "U"),
            (ansatz_v, bitstrings_v, "V"),
        ):
            if ansatz.num_qubits != len(bitstrings[0]):
                raise ValueError(
                    f"The number of qubits in ansatz {name} does "
                    "not match the number of bits in bitstrings."
                )

            if any(len(bitstrings[0]) != len(bitstr) for bitstr in bitstrings):
                raise ValueError(f"All {name} bitstrings must be the same length.")

        if len(bitstrings_u) != len(bitstrings_v):
            raise ValueError(
                "The same number of bitstrings should be passed for U and V."
            )

        if len(bitstrings_u[0]) != len(bitstrings_v[0]):
            raise ValueError("Bitstrings for U and V should be the same length.")

        # Initialize the GroundStateSolver
        super().__init__(qubit_converter)

        # Set which orbitals to ignore when calculating the Hamiltonian
        if orbitals_to_reduce is None:
            orbitals_to_reduce = []
        self.orbitals_to_reduce = orbitals_to_reduce

        # Set private class fields
        self._ansatz_u = ansatz_u
        self._ansatz_v = ansatz_v

        self._bitstrings_u = bitstrings_u
        self._bitstrings_v = bitstrings_v

        self._config = config  # pylint: disable=arguments-differ

    # pylint: disable=arguments-differ
    def solve(
        self,
        problem: ElectronicStructureProblem,
    ) -> EntanglementForgedVQEResult:
        """Compute Ground State properties of chemical problem.

        Args:
            problem: a qiskit_nature.problems.second_quantization
                .electronic.electronic_structure_problem object.
            aux_operators: additional auxiliary operators to evaluate
            **kwargs: keyword args to pass to solver

        Raises:
            ValueError: If the transformation is not of the type FermionicTransformation.
            ValueError: If the qubit mapping is not of the type JORDAN_WIGNER.

        Returns:
            An eigenstate result.
        """

        if not isinstance(problem, ElectronicStructureProblem):
            raise ValueError(
                "This version only supports an ElectronicStructureProblem."
            )
        if not isinstance(self.qubit_converter.mapper, JordanWignerMapper):
            raise ValueError("This version only supports the JordanWignerMapper.")

        start_time = time.time()

        problem.driver.run()

        # Decompose the Hamiltonian operators into a form appropraite for EF
        forged_operator = ForgedOperator(
            problem, self.orbitals_to_reduce, self._calculate_tensor_cross_terms()
        )

        # Calculate energies clasically using pySCF
        classical_energies = ClassicalEnergies(problem, self.orbitals_to_reduce)

        self._solver = EntanglementForgedVQE(
            ansatz_u=self._ansatz_u,
            ansatz_v=self._ansatz_v,
            bitstrings_u=self._bitstrings_u,
            bitstrings_v=self._bitstrings_v,
            config=self._config,
            forged_operator=forged_operator,
            classical_energies=classical_energies,
        )
        result = self._solver.compute_minimum_eigenvalue(forged_operator.h_1_op)

        elapsed_time = time.time() - start_time
        Log.log(f"VQE for this problem took {elapsed_time} seconds")
        res = EntanglementForgedVQEResult(
            parameters_history=self._solver._paramsets_each_iteration,
            energies_history=self._solver._energy_each_iteration_each_paramset,
            schmidts_history=self._solver._schmidt_coeffs_each_iteration_each_paramset,
            energy_std_each_parameter_set=self._solver.energy_std_each_parameter_set,
            energy_offset=self._solver._add_this_to_energies_displayed,
            eval_count=self._solver._eval_count,
        )
        res.combine(result)
        return res

    def _calculate_tensor_cross_terms(self) -> bool:
        """
        Determine whether circuits should be generated to account for
        the special superposition terms needed when bn==bm for two
        bitstrings within a given subsystem's (U or V) bitstring list.
        """
        bsu = self._bitstrings_u
        bsv = self._bitstrings_v

        # Search for any duplicate bitstrings within the subsystem lists
        for i, bu1 in enumerate(bsu):
            for j, bu2 in enumerate(bsu):
                if i == j:
                    continue
                if bu1 == bu2:
                    return True
        for i, bv1 in enumerate(bsv):
            for j, bv2 in enumerate(bsv):
                if i == j:
                    continue
                if bv1 == bv2:
                    return True

        return False

    def returns_groundstate(self) -> bool:
        """Whether this class returns only the ground state energy or also the ground state itself.

        Returns:
            True, if this class also returns the ground state in the results object.
            False otherwise.
        """
        return True

    def evaluate_operators(
        self,
        state: Union[
            str,
            dict,
            Result,
            list,
            np.ndarray,
            Statevector,
            QuantumCircuit,
            Instruction,
            OperatorBase,
        ],
        operators: Union[PauliSumOp, OperatorBase, list, dict],
    ) -> Union[float, Iterable[float], Dict[str, Iterable[float]]]:
        """Evaluates additional operators at the given state."""
        warnings.warn(
            "evaluate_operators not implemented for "
            "forged EntanglementForgedGroundStateSolver."
        )
        return []

    @property
    def solver(self) -> Union[MinimumEigensolver, MinimumEigensolverFactory]:
        """Returns the minimum eigensolver or factory."""
        return self._solver

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: Optional[
            ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]
        ] = None,
    ) -> Tuple[PauliSumOp, Optional[ListOrDictType[PauliSumOp]]]:
        """Gets the operator and auxiliary operators, and transforms the provided auxiliary operators"""
        raise NotImplementedError(
            "get_qubit_operators has not been implemented in EntanglementForgedGroundStateEigensolver"
        )
