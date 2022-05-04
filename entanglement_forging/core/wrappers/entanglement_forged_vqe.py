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

"""Entanglement forged VQE."""

import datetime
import time
import warnings
from typing import List, Tuple, Callable, Union

import numpy as np
from qiskit import transpile
from qiskit.algorithms import VQE
from qiskit.ignis.mitigation.measurement import complete_meas_cal
from qiskit.quantum_info import Pauli
from qiskit.utils import QuantumInstance
from qiskit.opflow import OperatorBase

from entanglement_forging.core.wrappers.entanglement_forged_vqe_result import (
    DataResults,
    Bootstrap,
    AuxiliaryResults,
)
from entanglement_forging.utils.bootstrap_result import resample_result
from entanglement_forging.utils.copysample_circuits import (
    copysample_circuits,
    combine_copysampled_results,
)
from entanglement_forging.utils.forging_subroutines import (
    prepare_circuits_to_execute,
    make_stateprep_circuits,
    eval_forged_op_with_result,
    get_optimizer_instance,
)
from entanglement_forging.utils.generic_execution_subroutines import execute_with_retry
from entanglement_forging.utils.legacy.op_converter import (
    to_tpb_grouped_weighted_pauli_operator,
)
from entanglement_forging.utils.legacy.tpb_grouped_weighted_pauli_operator import (
    TPBGroupedWeightedPauliOperator,
)
from entanglement_forging.utils.legacy.weighted_pauli_operator import (
    WeightedPauliOperator,
)
from entanglement_forging.utils.log import Log
from entanglement_forging.utils.meas_mit_fitters_faster import CompleteMeasFitter
from entanglement_forging.utils.pseudorichardson import make_pseudorichardson_circuits


# pylint: disable=too-many-branches,too-many-arguments,too-many-locals,too-many-instance-attributes,too-many-statements
class EntanglementForgedVQE(VQE):
    """A class for Entanglement Forged VQE <https://arxiv.org/abs/2104.10220>."""

    def __init__(
        self,
        ansatz,
        bitstrings_u,
        config,
        forged_operator,
        classical_energies,
        bitstrings_v=[],
    ):
        """Initialize the EntanglementForgedVQE class."""
        if ansatz.num_qubits != len(bitstrings_u[0]):
            raise ValueError(
                "The number of qubits in ansatz does "
                "not match the number of bits in reduced_bitstrings."
            )

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

        super().__init__(
            ansatz=ansatz,
            optimizer=get_optimizer_instance(config),
            initial_point=config.initial_params,
            max_evals_grouped=config.max_evals_grouped,
            callback=None,
        )
        self.ansatz = ansatz
        self.config = config
        self._energy_each_iteration_each_paramset = []
        self._paramsets_each_iteration = []
        self._schmidt_coeffs_each_iteration_each_paramset = []
        self._zero_noise_extrap = config.zero_noise_extrap

        self.bitstrings_u = bitstrings_u
        self._bitstrings_s_u = np.asarray(bitstrings_u)

        self.bitstrings_v = bitstrings_v

        # Make circuits which prepare states U|b_n_u> and U|phi^p_nm_u>
        (
            self._tensor_prep_circuits_u,
            self._superpos_prep_circuits_u,
            superpos_coeffs_u,
        ) = make_stateprep_circuits(
            bitstrings_u, config.fix_first_bitstring, suffix="u"
        )

        self._tensor_prep_circuits_v = []
        self._superpos_prep_circuits_v = []
        superpos_coeffs_v = []
        self._bitstrings_s_v = np.array([])
        if self.bitstrings_v:
            # Make circuits which prepare states V|b_n_v> and V|phi^p_nm_v>
            # Where U = V but bitstring |b_n_v> does not necessarily equal |b_n_u>
            self._bitstrings_s_v = np.asarray(bitstrings_v)
            (
                self._tensor_prep_circuits_v,
                self._superpos_prep_circuits_v,
                superpos_coeffs_v,
            ) = make_stateprep_circuits(
                bitstrings_v, config.fix_first_bitstring, suffix="v"
            )

            # Offset the superpos coeffs associated with V so the indices match up
            # We multiply by 2 to account for the +/- terms for each superposition pair
            offset = 2 * len(self._superpos_prep_circuits_u)
            for i, coeff in enumerate(superpos_coeffs_v):
                superpos_coeffs_v[i][0] += offset

        self._superpos_coeffs = superpos_coeffs_u + superpos_coeffs_v

        self._iteration_start_time = np.nan
        self._running_estimate_of_schmidts = np.array(
            [1.0] + [0.1] * (len(self._bitstrings_s_u) - 1)
        )
        self._running_estimate_of_schmidts /= np.linalg.norm(
            self._running_estimate_of_schmidts
        )
        self.copysample_job_size = config.copysample_job_size
        self._backend = config.backend
        self.quantum_instance = QuantumInstance(backend=self._backend)
        self._initial_layout = config.qubit_layout
        self._shots = config.shots
        self._meas_error_mit = config.meas_error_mit
        self._meas_error_shots = config.meas_error_shots
        self._meas_error_refresh_period_minutes = (
            config.meas_error_refresh_period_minutes
        )
        self._meas_error_refresh_timestamp = None
        self._coupling_map = self._backend.configuration().coupling_map
        self._meas_fitter = None
        self._bootstrap_trials = config.bootstrap_trials
        self._no_bs0_circuits = config.fix_first_bitstring
        self._rep_delay = config.rep_delay

        self.forged_operator = forged_operator
        self._add_this_to_energies_displayed = classical_energies.shift
        self._hf_energy = classical_energies.HF

        self.aux_results: List[Tuple[str, AuxiliaryResults]] = []

        self.parameter_sets = []
        self.energy_mean_each_parameter_set = []
        self.energy_std_each_parameter_set = []

        # Paramters for get_energy_evaluation. Moved them here to match parent function signature
        self._shots_multiplier = 1
        self._bootstrap_trials = 0

        statevector_sims = ["aer_simulator_statevector", "statevector_simulator"]
        if self._backend.name() in statevector_sims:
            self._is_sv_sim = True
        else:
            self._is_sv_sim = False

    @property
    def shots_multiplier(self):
        """Return the shots multiplier."""
        return self._shots_multiplier

    @shots_multiplier.setter
    def shots_multiplier(self, multiplier):
        """Set the shots multiplier."""
        self._shots_multiplier = multiplier

    @property
    def bootstrap_trials(self):
        """Return the bootstrap trials."""
        return self._bootstrap_trials

    @bootstrap_trials.setter
    def bootstrap_trials(self, trials):
        """Set the bootstrap trials."""
        self._bootstrap_trials = trials

    def get_energy_evaluation(
        self, operator: OperatorBase, return_expectation: bool = False
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:

        if self._is_sv_sim:
            self._shots_multiplier = 1

        ansatz_params = self.ansatz.parameters
        _, expectation = self.construct_expectation(
            ansatz_params, operator, return_expectation=True
        )

        def energy_evaluation(parameters):
            Log.log("------ new iteration energy evaluation -----")
            # num_parameter_sets = len(parameters) // self.ansatz.num_parameters
            # parameter_sets = np.split(parameters, num_parameter_sets)
            parameter_sets = np.reshape(parameters, (-1, self.ansatz.num_parameters))
            new_iteration_start_time = time.time()

            Log.log(
                "duration of last iteration:",
                new_iteration_start_time - self._iteration_start_time,
            )

            self._iteration_start_time = new_iteration_start_time

            Log.log("Parameter sets:", parameter_sets)

            # Compute the expectation value of the forged operator with respect
            # to the ansatz at the given parameters
            eval_forged_result = self._evaluate_forged_operator(
                parameter_sets=parameter_sets,
                hf_value=self._hf_energy,
                add_this_to_mean_values_displayed=self._add_this_to_energies_displayed,
                shots_multiplier=self._shots_multiplier,
                bootstrap_trials=self._bootstrap_trials,
            )

            (
                energy_mean_each_parameter_set,
                bootstrap_means_each_parameter_set,
                energy_mean_raw_each_parameter_set,
                energy_mean_sv_each_parameter_set,
                schmidt_coeffs_each_parameter_set,
                schmidt_coeffs_raw_each_parameter_set,
                schmidt_coeffs_sv_each_parameter_set,
            ) = eval_forged_result

            self._schmidt_coeffs_each_iteration_each_paramset.append(
                schmidt_coeffs_each_parameter_set
            )

            # TODO Not Implemented  # pylint: disable=fixme
            energy_std_each_parameter_set = [0] * len(energy_mean_each_parameter_set)
            # TODO Not Implemented  # pylint: disable=fixme
            energy_std_raw_each_parameter_set = [0] * len(
                energy_mean_each_parameter_set
            )
            energy_std_sv_each_parameter_set = [0] * len(energy_mean_each_parameter_set)

            self._energy_each_iteration_each_paramset.append(
                energy_mean_each_parameter_set
            )
            self._paramsets_each_iteration.append(parameter_sets)

            timestamp_string = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

            # additional results
            for params, bootstrap_means in zip(
                parameter_sets, bootstrap_means_each_parameter_set
            ):
                self.aux_results.append(
                    (
                        "bootstrap",
                        Bootstrap(
                            eval_count=self._eval_count,
                            eval_timestamp=timestamp_string,
                            parameters=params,
                            bootstrap_values=[
                                v + self._add_this_to_energies_displayed
                                for v in bootstrap_means
                            ],
                        ),
                    )
                )

            data_titles = ["data", "data_noextrapolation", "data_statevec"][:-1]
            for (
                filename,
                energies_each_paramset,
                stds_each_paramset,
                schmidts_each_paramset,
            ) in zip(
                data_titles,
                [
                    energy_mean_each_parameter_set,
                    energy_mean_raw_each_parameter_set,
                    energy_mean_sv_each_parameter_set,
                ],
                [
                    energy_std_each_parameter_set,
                    energy_std_raw_each_parameter_set,
                    energy_std_sv_each_parameter_set,
                ],
                [
                    schmidt_coeffs_each_parameter_set,
                    schmidt_coeffs_raw_each_parameter_set,
                    schmidt_coeffs_sv_each_parameter_set,
                ],
            ):

                for params, energy_mean, energy_std, schmidts in zip(
                    parameter_sets,
                    energies_each_paramset,
                    stds_each_paramset,
                    schmidts_each_paramset,
                ):
                    self.aux_results.append(
                        (
                            filename,
                            DataResults(
                                eval_count=self._eval_count,
                                eval_timestamp=timestamp_string,
                                energy_hartree=(
                                    energy_mean + self._add_this_to_energies_displayed
                                ),
                                energy_std=energy_std,
                                parameters=parameters,
                                schmidts=schmidts,
                            ),
                        )
                    )

            for params, energy_mean, energy_std in zip(
                parameter_sets,
                energy_mean_each_parameter_set,
                energy_std_each_parameter_set,
            ):
                self._eval_count += 1
                if self._callback is not None:
                    self._callback(self._eval_count, params, energy_mean, energy_std)

            self.parameter_sets = parameter_sets
            self.energy_mean_each_parameter_set = energy_mean_each_parameter_set
            self.energy_std_each_parameter_set = energy_std_each_parameter_set
            return (
                np.array(energy_mean_each_parameter_set)
                if len(energy_mean_each_parameter_set) > 1
                else energy_mean_each_parameter_set[0]
            )

        if return_expectation:
            return energy_evaluation, expectation

        return energy_evaluation

    def _evaluate_forged_operator(
        self,
        parameter_sets,
        hf_value=0,
        add_this_to_mean_values_displayed=0,
        shots_multiplier=1,
        bootstrap_trials=0,
    ):
        """Computes the expectation value of the forged operator
        with respect to the ansatz at the given parameters."""
        # These calculations are parameter independent for
        # a given operator so could be moved outside the optimization loop:

        # Get the weighted Pauli deconstruction of the operator
        (
            pauli_names_for_tensor_states,
            pauli_names_for_superpos_states,
            w_ij_tensor_states,
            w_ij_superpos_states,
        ) = self.forged_operator.construct()
        # We have a bunch of 6-qb Paulis we want evaluated.
        # We bundle these in a single operator that will
        #   use qiskit's standard routines to define the minimum
        #   number of circuits required to evaluate these Paulis:
        op_for_generating_tensor_circuits = to_tpb_grouped_weighted_pauli_operator(
            WeightedPauliOperator(
                paulis=[[1, Pauli(pname)] for pname in pauli_names_for_tensor_states]
            ),
            TPBGroupedWeightedPauliOperator.sorted_grouping,
        )
        if pauli_names_for_superpos_states:
            op_for_generating_superpos_circuits = (
                to_tpb_grouped_weighted_pauli_operator(
                    WeightedPauliOperator(
                        paulis=[
                            [1, Pauli(pname)]
                            for pname in pauli_names_for_superpos_states
                        ]
                    ),
                    TPBGroupedWeightedPauliOperator.sorted_grouping,
                )
            )

        else:
            op_for_generating_superpos_circuits = None
        circuits_to_execute = []

        for params_idx, params in enumerate(parameter_sets):
            Log.log("Constructing the circuits for parameter set", params, "...")

            # Get all the tensor circuits associated with ansatz U
            tensor_circuits_to_execute_u = prepare_circuits_to_execute(
                params,
                self._tensor_prep_circuits_u,
                op_for_generating_tensor_circuits,
                self._ansatz,
                self._is_sv_sim,
            )

            # Get all tensor circuits associated with ansatz V
            tensor_circuits_to_execute_v = prepare_circuits_to_execute(
                params,
                self._tensor_prep_circuits_v,
                op_for_generating_tensor_circuits,
                self._ansatz,
                self._is_sv_sim,
            )

            # Combine all tensor circuits into a single list
            tensor_circuits_to_execute = (
                tensor_circuits_to_execute_u + tensor_circuits_to_execute_v
            )

            if pauli_names_for_superpos_states:
                # Get superposition circuits associated with ansatz U
                superpos_circuits_to_execute_u = prepare_circuits_to_execute(
                    params,
                    self._superpos_prep_circuits_u,
                    op_for_generating_superpos_circuits,
                    self._ansatz,
                    self._is_sv_sim,
                )
                # Get superposition circuits associated with ansatz V
                superpos_circuits_to_execute_v = prepare_circuits_to_execute(
                    params,
                    self._superpos_prep_circuits_v,
                    op_for_generating_superpos_circuits,
                    self._ansatz,
                    self._is_sv_sim,
                )

                # Combine all superposition circuits into a single list
                superpos_circuits_to_execute = (
                    superpos_circuits_to_execute_u + superpos_circuits_to_execute_v
                )

            else:
                superpos_circuits_to_execute_u = []
                superpos_circuits_to_execute_v = []
                superpos_circuits_to_execute = []

            if params_idx == 0:
                Log.log(
                    "inferred number of pauli groups for tensor statepreps:",
                    len(tensor_circuits_to_execute)
                    / (
                        len(self._tensor_prep_circuits_u)
                        + len(self._tensor_prep_circuits_v)
                    ),
                )
                if self._superpos_prep_circuits_u:
                    Log.log(
                        "inferred number of pauli groups for superposition statepreps:",
                        len(superpos_circuits_to_execute)
                        / (
                            len(self._superpos_prep_circuits_u)
                            + len(self._superpos_prep_circuits_v)
                        ),
                    )
            circuits_to_execute += (
                tensor_circuits_to_execute + superpos_circuits_to_execute
            )
        Log.log("Transpiling circuits...")
        Log.log(self._initial_layout)
        circuits_to_execute = transpile(
            circuits_to_execute,
            self._backend,
            initial_layout=self._initial_layout,
            coupling_map=self._coupling_map,
        )

        if not isinstance(circuits_to_execute, list):
            circuits_to_execute = [circuits_to_execute]
        Log.log("Building pseudo-richardson circuits...")
        circuits_to_execute = make_pseudorichardson_circuits(
            circuits_to_execute,
            simple_richardson_orders=[
                int(x) for x in (np.asarray(self._zero_noise_extrap) - 1) / 2
            ],
        )
        if self.copysample_job_size:
            Log.log("Copysampling circuits...")
            Log.log(
                "num circuits to execute before copysampling:", len(circuits_to_execute)
            )
            weight_each_stateprep = np.abs(
                np.array(self._running_estimate_of_schmidts)[:, np.newaxis]
                * np.array(self._running_estimate_of_schmidts)[np.newaxis, :]
            )
            if self._no_bs0_circuits:
                ## IMPORTANT: assumes special-case of NOT executing HF bitstring state-prep
                weight_each_stateprep[0, 0] = 0
            weight_each_circuit = []
            for qcirc in circuits_to_execute:
                name_parts = qcirc.name.split("_")
                stretch_factor = float(name_parts[-2].split("richardson")[1])
                indices_of_involved_bitstrings = [
                    int("".join(c for c in x if c.isdigit()))
                    for x in name_parts[1].split("bs")[1:]
                ]
                if len(indices_of_involved_bitstrings) == 1:
                    i, j = (
                        indices_of_involved_bitstrings[0],
                        indices_of_involved_bitstrings[0],
                    )
                elif len(indices_of_involved_bitstrings) == 2:
                    i, j = indices_of_involved_bitstrings
                else:
                    raise ValueError(
                        "Circuit name should be of form [params]_bs#_... "
                        "or [params]_bs#bs#_... indicating which 1 or 2 "
                        "bitstrings it involves, but instead name is:",
                        qcirc.name,
                    )
                if len(self._zero_noise_extrap) <= 2:
                    weight_each_circuit.append(
                        weight_each_stateprep[i, j] / stretch_factor
                    )
                else:
                    warnings.warn(
                        "Weighted sampling when more than 2 stretch factors are present "
                        "is not supported (may or may not just work, haven't "
                        "looked into it). Reverting to uniform sampling of stretch factors."
                    )
                    weight_each_circuit.append(weight_each_stateprep[i, j] / 1)
            circuits_to_execute = copysample_circuits(
                circuits_to_execute,
                weights=weight_each_circuit,
                new_job_size=self.copysample_job_size * len(parameter_sets),
            )
            Log.log(
                "num circuits to execute after copysampling:", len(circuits_to_execute)
            )

        if self._meas_error_mit:
            if (not self._meas_fitter) or (
                (time.time() - self._meas_error_refresh_timestamp) / 60
                > self._meas_error_refresh_period_minutes
            ):
                Log.log("Generating measurement fitter...")
                physical_qubits = np.asarray(self._initial_layout).tolist()
                cal_circuits, state_labels = complete_meas_cal(
                    range(len(physical_qubits))
                )
                result = execute_with_retry(
                    cal_circuits, self._backend, self._meas_error_shots, self._rep_delay
                )
                self._meas_fitter = CompleteMeasFitter(result, state_labels)
                self._meas_error_refresh_timestamp = time.time()

        Log.log("Executing", len(circuits_to_execute), "circuits...")

        result = execute_with_retry(
            circuits_to_execute,
            self._backend,
            self._shots * shots_multiplier,
            self._rep_delay,
        )

        if self._meas_error_mit:
            Log.log("Applying meas fitter/filter...")
            result = self._meas_fitter.filter.apply(result)

        Log.log("Done executing. Analyzing results...")
        op_mean_each_parameter_set = [None] * len(parameter_sets)
        op_std_each_parameter_set = [None] * len(parameter_sets)
        schmidt_coeffs_each_parameter_set = [None] * len(parameter_sets)
        op_mean_raw_each_parameter_set = [None] * len(parameter_sets)
        op_std_raw_each_parameter_set = [None] * len(parameter_sets)
        schmidt_coeffs_raw_each_parameter_set = [None] * len(parameter_sets)
        op_mean_sv_each_parameter_set = [None] * len(parameter_sets)
        schmidt_coeffs_sv_each_parameter_set = [None] * len(parameter_sets)
        if self.copysample_job_size:
            result = combine_copysampled_results(result)
        if bootstrap_trials:
            Log.log(f"Bootstrap: resampling result {bootstrap_trials} times...")
            bootstrap_results = [
                resample_result(result) for _ in range(bootstrap_trials)
            ]
            Log.log(
                "Done bootstrapping new counts, starting analysis of bootstrapped data."
            )
        else:
            bootstrap_results = []
        bootstrap_means_each_parameter_set = []
        for idx, params in enumerate(parameter_sets):
            bootstrap_means = []
            for is_bootstrap_index, res in enumerate([result] + bootstrap_results):
                results_extrap, results_raw = eval_forged_op_with_result(
                    res,
                    w_ij_tensor_states,
                    w_ij_superpos_states,
                    params,
                    self._bitstrings_s_u,
                    op_for_generating_tensor_circuits,
                    op_for_generating_superpos_circuits,
                    self._zero_noise_extrap,
                    bitstrings_s_v=self._bitstrings_s_v,
                    hf_value=hf_value,
                    statevector_mode=self._is_sv_sim,
                    superpos_coeffs=self._superpos_coeffs,
                    add_this_to_mean_values_displayed=add_this_to_mean_values_displayed,
                    no_bs0_circuits=self._no_bs0_circuits,
                )
                op_mean, op_std, schmidts = results_extrap
                op_mean_raw, op_std_raw, schmidts_raw = results_raw
                if not is_bootstrap_index:
                    op_mean_each_parameter_set[idx] = op_mean
                    op_std_each_parameter_set[idx] = op_std
                    op_mean_raw_each_parameter_set[idx] = op_mean_raw
                    op_std_raw_each_parameter_set[idx] = op_std_raw
                    schmidt_coeffs_raw_each_parameter_set[idx] = schmidts_raw
                    Log.log("Optimal schmidt coeffs sqrt(p) =", schmidts)
                    schmidt_coeffs_each_parameter_set[idx] = schmidts
                else:
                    bootstrap_means.append(op_mean)
            bootstrap_means_each_parameter_set.append(bootstrap_means)
        self._running_estimate_of_schmidts = np.mean(
            schmidt_coeffs_each_parameter_set, axis=0
        )
        return (
            op_mean_each_parameter_set,
            bootstrap_means_each_parameter_set,
            op_mean_raw_each_parameter_set,
            op_mean_sv_each_parameter_set,
            schmidt_coeffs_each_parameter_set,
            schmidt_coeffs_raw_each_parameter_set,
            schmidt_coeffs_sv_each_parameter_set,
        )

    def get_optimal_vector(self):
        """Prevents the VQE superclass version of this function from running."""
        warnings.warn(
            "get_optimal_vector not implemented for forged VQE. Returning None."
        )
