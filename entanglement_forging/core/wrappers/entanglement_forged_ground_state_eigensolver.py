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
from typing import Iterable, Union, Dict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.opflow import OperatorBase, PauliSumOp
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit_nature.algorithms.ground_state_solvers import GroundStateSolver
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
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
    OptimalParams,
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
        ansatz: QuantumCircuit,
        bitstrings_u: Iterable[Iterable[int]],
        config: EntanglementForgedConfig,
        bitstrings_v: Iterable[Iterable[int]] = None,
        orbitals_to_reduce: bool = None,
    ):
        # Ensure the bitstrings are well formed
        if any(len(bitstrings_u[0]) != len(bitstr) for bitstr in bitstrings_u):
            raise ValueError("All U bitstrings must be the same length.")

        if bitstrings_v:
            if len(bitstrings_u) != len(bitstrings_v):
                raise ValueError(
                    "The same number of bitstrings should be passed for U and V."
                )

            if len(bitstrings_u[0]) != len(bitstrings_v[0]):
                raise ValueError("Bitstrings for U and V should be the same length.")

            if any(len(bitstrings_v[0]) != len(bitstr) for bitstr in bitstrings_v):
                raise ValueError("All V bitstrings must be the same length.")

        # Initialize the GroundStateSolver
        super().__init__(qubit_converter)

        # Set which orbitals to ignore when calculating the Hamiltonian
        if orbitals_to_reduce is None:
            orbitals_to_reduce = []
        self.orbitals_to_reduce = orbitals_to_reduce

        # Prevent unnecessary duplication of circuits if subsystems are identical
        if (bitstrings_v is None) or (bitstrings_u == bitstrings_v):
            self._bitstrings_v = []
        else:
            self._bitstrings_v = bitstrings_v

        # Set private class fields
        self._ansatz = ansatz
        self._bitstrings_u = bitstrings_u
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

        # Instantiate EFVQE object
        solver = EntanglementForgedVQE(
            ansatz=self._ansatz,
            bitstrings_u=self._bitstrings_u,
            bitstrings_v=self._bitstrings_v,
            config=self._config,
            forged_operator=forged_operator,
            classical_energies=classical_energies,
        )

        result = solver.compute_minimum_eigenvalue(forged_operator.h_1_op)

        elapsed_time = time.time() - start_time
        Log.log(f"VQE for this problem took {elapsed_time} seconds")
        res = EntanglementForgedVQEResult(
            parameters_history=solver._paramsets_each_iteration,
            energies_history=solver._energy_each_iteration_each_paramset,
            schmidts_history=solver._schmidt_coeffs_each_iteration_each_paramset,
            energy_std_each_parameter_set=solver.energy_std_each_parameter_set,
            energy_offset=solver._add_this_to_energies_displayed,
            eval_count=solver._eval_count,
        )
        res.combine(result)
        return res

    def _calculate_tensor_cross_terms(self) -> bool:
        """
        Determine whether circuits should be generated to account for
        the tensor terms which should up in the cross-terms in the special
        case where bn==bm within a given subsystem's (U or V) bitstring list.
        """
        bsu = self._bitstrings_u
        bsv = self._bitstrings_u

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
