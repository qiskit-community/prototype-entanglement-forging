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

""" A set of subroutines that are used by the VQE_forged_entanglement class,
which is defined in the vqe_forged_entanglement module
"""

import inspect
import warnings
import re
from typing import Iterable, Dict, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms import optimizers
from qiskit.algorithms.optimizers import SPSA
from qiskit.algorithms.optimizers.spsa import powerseries
from qiskit_nature import QiskitNatureError

from .generic_execution_subroutines import compute_pauli_means_and_cov_for_one_basis
from .log import Log
from .prepare_bitstring import prepare_bitstring
from .pseudorichardson import richardson_extrapolate
from .legacy.weighted_pauli_operator import WeightedPauliOperator


Matrix = Iterable[Iterable[float]]

# pylint: disable=too-many-locals,too-many-arguments,too-many-branches,invalid-name
def make_stateprep_circuits(
    bitstrings: Iterable[Iterable[int]], no_bs0_circuits: bool = True, suffix: str = ""
):
    """Builds the circuits preparing states |b_n> and |phi^p_nm>
    as defined in <https://arxiv.org/abs/2104.10220>. Also returns
    a list of tuples which describe any superposition terms which
    carry a coefficient.

    Assumes that the operator amplitudes are real,
    thus does not construct superposition states with odd p.
    """

    # If empty, just return
    if len(bitstrings) == 0:
        return [], [], []

    # If the spin-up and spin-down spin orbitals are together a 2*N qubit system,
    # the bitstring should be N bits long.
    bitstrings = np.asarray(bitstrings)
    tensor_prep_circuits = [
        prepare_bitstring(bs, name="bs" + suffix + str(bs_idx))
        for bs_idx, bs in enumerate(bitstrings)
    ]

    if no_bs0_circuits:
        # Drops the HF bitstring, which is assumed to be first,
        # since hops in our ansatz are chosen to leave it unchanged! (IMPORTANT)
        tensor_prep_circuits = tensor_prep_circuits[1:]

    superpos_prep_circuits = []
    hybrid_superpos_coeffs = {}
    for bs1_idx, bs1 in enumerate(bitstrings):
        for bs2_relative_idx, bs2 in enumerate(bitstrings[bs1_idx + 1 :]):
            diffs = np.where(bs1 != bs2)[0]

            # TODO implement p -> -p as needed for problems with complex amplitudes  # pylint: disable=fixme
            if len(diffs) > 0:
                i = diffs[0]
                if bs1[i]:
                    x = bs2
                    y = bs1  # pylint: disable=unused-variable
                else:
                    x = bs1
                    y = bs2  # pylint: disable=unused-variable
                S = np.delete(diffs, 0)
                qcirc = prepare_bitstring(np.concatenate((x[:i], [0], x[i + 1 :])))
                qcirc.h(i)
                psi_xplus, psi_xmin = [
                    qcirc.copy(
                        name=f"bs{suffix}{bs1_idx}bs{suffix}{bs1_idx+1+bs2_relative_idx}{name}"
                    )
                    for name in ["xplus", "xmin"]
                ]
                psi_xmin.z(i)
                for psi in [psi_xplus, psi_xmin]:
                    for target in S:
                        psi.cx(i, target)
                    superpos_prep_circuits.append(psi)

            # If the two bitstrings are equivalent -- bn==bm
            else:
                qcirc = prepare_bitstring(bs1)
                psi_xplus, psi_xmin = [
                    qcirc.copy(
                        name=f"bs{suffix}{bs1_idx}bs{suffix}{bs1_idx+1+bs2_relative_idx}{name}"
                    )
                    for name in ["xplus", "xmin"]
                ]
                hybrid_superpos_coeffs[
                    (suffix, str(bs1_idx), str(bs1_idx + 1 + bs2_relative_idx))
                ] = True
                superpos_prep_circuits += [psi_xplus, psi_xmin]
    return tensor_prep_circuits, superpos_prep_circuits, hybrid_superpos_coeffs


def prepare_circuits_to_execute(
    params: Iterable[float],
    stateprep_circuits: Iterable[QuantumCircuit],
    op_for_generating_circuits: WeightedPauliOperator,
    var_form: QuantumCircuit,
    statevector_mode: bool,
):
    """Given a set of variational parameters and list of 6qb state-preps,
    this function returns all (unique) circuits that must be run to evaluate those samples.
    """
    # If circuits are empty, just return
    if len(stateprep_circuits) == 0:
        return []

    circuits_to_execute = []
    # Generate the requisite circuits:
    # pylint: disable=unidiomatic-typecheck
    param_bindings = dict(zip(var_form.parameters, params))
    u_circuit = var_form.bind_parameters(param_bindings)
    for prep_circ in [qc.copy() for qc in stateprep_circuits]:
        circuit_name_prefix = str(params) + "_" + str(prep_circ.name) + "_"
        wavefn_circuit = prep_circ.compose(u_circuit)
        if statevector_mode:
            circuits_to_execute.append(
                wavefn_circuit.copy(name=circuit_name_prefix + "psi")
            )
        else:
            circuits_this_stateprep = (
                op_for_generating_circuits.construct_evaluation_circuit(
                    wave_function=wavefn_circuit.copy(),
                    statevector_mode=statevector_mode,  # <-- False, here.
                    use_simulator_snapshot_mode=False,
                    circuit_name_prefix=circuit_name_prefix,
                )
            )
            circuits_to_execute += circuits_this_stateprep
    return circuits_to_execute


# pylint: disable=unbalanced-tuple-unpacking
def eval_forged_op_with_result(
    result,
    w_ij_tensor_states: Matrix,
    w_ab_superpos_states: Matrix,
    params: Iterable[float],
    bitstrings_s_u: np.ndarray,
    op_for_generating_tensor_circuits: WeightedPauliOperator,
    op_for_generating_superpos_circuits: WeightedPauliOperator,
    richardson_stretch_factors: Iterable[float],
    statevector_mode: bool,
    hf_value: float,
    add_this_to_mean_values_displayed: float,
    bitstrings_s_v: np.ndarray = None,
    hybrid_superpos_coeffs: Dict[Tuple[int, int, str], bool] = None,
    no_bs0_circuits: bool = True,
    verbose: bool = False,
):
    """Evaluates the forged operator.

    Extracts necessary expectation values from the result object, then combines those pieces to
    compute the configuration-interaction Hamiltonian
    (Hamiltonian in the basis of determinants/bitstrings).
    For reference, also computes mean value obtained without Richardson
    """
    if bitstrings_s_v is None:
        bitstrings_s_v = []

    tensor_state_prefixes_u = [f"bsu{idx}" for idx in range(len(bitstrings_s_u))]
    tensor_state_prefixes_v = []

    if len(bitstrings_s_v) > 0:
        tensor_state_prefixes_v = [f"bsv{idx}" for idx in range(len(bitstrings_s_v))]

    tensor_state_prefixes = tensor_state_prefixes_u + tensor_state_prefixes_v
    tensor_expvals = _get_pauli_expectations_from_result(
        result,
        params,
        tensor_state_prefixes,
        op_for_generating_tensor_circuits,
        richardson_stretch_factors=richardson_stretch_factors,
        hybrid_superpos_coeffs=hybrid_superpos_coeffs,
        statevector_mode=statevector_mode,
        no_bs0_circuits=no_bs0_circuits,
    )
    tensor_expvals_extrap = richardson_extrapolate(
        tensor_expvals, richardson_stretch_factors, axis=2
    )

    superpos_state_prefixes_u = []
    superpos_state_prefixes_v = []
    superpos_state_indices = []
    lin_combos = ["xplus", "xmin"]  # ,'yplus','ymin']
    # num_bitstrings is the number of bitstring combos we have
    num_bitstrings = len(bitstrings_s_u)
    for x in range(num_bitstrings):
        for y in range(num_bitstrings):
            if x == y:
                continue
            bsu_string = f"bsu{min(x,y)}bsu{max(x,y)}"
            superpos_state_prefixes_u += [
                bsu_string + lin_combo for lin_combo in lin_combos
            ]

            # Determine whether we are handling the two subsystems separately
            asymmetric_bitstrings = False
            if len(bitstrings_s_v) > 0:
                asymmetric_bitstrings = True
                bsv_string = f"bsv{min(x,y)}bsv{max(x,y)}"
                superpos_state_prefixes_v += [
                    bsv_string + lin_combo for lin_combo in lin_combos
                ]

            superpos_state_indices += [(x, y)]

    superpos_state_prefixes = superpos_state_prefixes_u + superpos_state_prefixes_v

    superpos_expvals = _get_pauli_expectations_from_result(
        result,
        params,
        superpos_state_prefixes,
        op_for_generating_superpos_circuits,
        hybrid_superpos_coeffs=hybrid_superpos_coeffs,
        richardson_stretch_factors=richardson_stretch_factors,
        statevector_mode=statevector_mode,
        no_bs0_circuits=no_bs0_circuits,
    )

    superpos_expvals_extrap = richardson_extrapolate(
        superpos_expvals, richardson_stretch_factors, axis=2
    )

    forged_op_results_w_and_wo_extrapolation = []
    for (tensor_expvals_real, superpos_expvals_real) in [
        [tensor_expvals_extrap, superpos_expvals_extrap],
        [tensor_expvals[:, :, 0], superpos_expvals[:, :, 0]],
    ]:
        h_schmidt = compute_h_schmidt(
            tensor_expvals_real,
            superpos_expvals_real,
            w_ij_tensor_states,
            w_ab_superpos_states,
            superpos_state_indices,
            asymmetric_bitstrings,
        )
        if no_bs0_circuits:
            # IMPORTANT: ASSUMING HOPGATES CHOSEN S.T. HF BITSTRING
            # (FIRST BITSTRING) IS UNAFFECTED, ALLOWING CORRESPONDING
            # ENERGY TO BE FIXED AT HF VALUE CALCULATED BY QISKIT/PYSCF.
            h_schmidt[0, 0] = hf_value - add_this_to_mean_values_displayed
        # Update Schmidt coefficients to minimize operator (presumably energy):
        if verbose:
            Log.log(
                "Operator as Schmidt matrix: (diagonals have been shifted by given offset",
                add_this_to_mean_values_displayed,
                ")",
            )
            Log.log(
                h_schmidt + np.eye(len(h_schmidt)) * add_this_to_mean_values_displayed
            )
        evals, evecs = np.linalg.eigh(h_schmidt)
        schmidts = evecs[:, 0]
        op_mean = evals[0]
        op_std = None
        forged_op_results_w_and_wo_extrapolation.append([op_mean, op_std, schmidts])

    (
        forged_op_results_extrap,
        forged_op_results_raw,
    ) = forged_op_results_w_and_wo_extrapolation  # pylint: disable=unbalanced-tuple-unpacking

    return forged_op_results_extrap, forged_op_results_raw


def _get_pauli_expectations_from_result(
    result,
    params,
    stateprep_strings,
    op_for_generating_circuits,
    statevector_mode,
    hybrid_superpos_coeffs=None,
    richardson_stretch_factors=None,
    no_bs0_circuits=True,
):
    """Returns array containing ordered expectation values of
    Pauli strings evaluated for the various wavefunctions.

    Axes are [stateprep_idx, Pauli_idx, richardson_stretch_factor_idx, mean_or_variance]
    """
    if richardson_stretch_factors is None:
        richardson_stretch_factors = [1]
    if not op_for_generating_circuits:
        return np.empty((0, 0, len(richardson_stretch_factors), 2))
    params_string = str(params) + "_"
    if statevector_mode:
        op_matrices = np.asarray(
            [op.to_matrix() for op in [p[1] for p in op_for_generating_circuits.paulis]]
        )
    pauli_vals = np.zeros(
        (
            len(stateprep_strings),
            len(op_for_generating_circuits._paulis),  # pylint: disable=protected-access
            len(richardson_stretch_factors),
            2,
        )
    )
    pauli_names_temp = [p[1].to_label() for p in op_for_generating_circuits.paulis]
    for prep_idx, prep_string in enumerate(stateprep_strings):
        suffix = prep_string[2]
        bitstring_pair = [0, 0]
        tensor_circuit = True
        if prep_string.count("bs") > 1:
            tensor_circuit = False
            prep_string_digits = [
                int(float(s)) for s in re.findall(r"-?\d+\.?\d*", prep_string)
            ]
            bitstring_pair = [prep_string_digits[0], prep_string_digits[1]]
        if no_bs0_circuits and (prep_string == "bsu0" or prep_string == "bsv0"):
            # IMPORTANT: ASSUMING HOPGATES CHOSEN S.T.
            # HF BITSTRING (FIRST BITSTRING) IS UNAFFECTED,
            # ALLOWING CORRESPONDING ENERGY TO BE FIXED AT H
            # F VALUE CALCULATED BY QISKIT/PYSCF.
            pauli_vals[prep_idx, :, :, :] = np.nan
            continue
        circuit_prefix_prefix = "".join([params_string, prep_string])
        for rich_idx, stretch_factor in enumerate(richardson_stretch_factors):
            circuit_name_prefix = "".join(
                [circuit_prefix_prefix, f"_richardson{stretch_factor:.2f}", "_"]
            )
            if statevector_mode:
                psi = result.get_statevector(circuit_name_prefix + "psi")
                pauli_vals_temp = np.real(
                    np.einsum("i,Mij,j->M", np.conj(psi), op_matrices, psi)
                )
            else:
                pauli_vals_temp, _ = _eval_each_pauli_with_result(
                    tpbgwpo=op_for_generating_circuits,
                    result=result,
                    statevector_mode=statevector_mode,
                    use_simulator_snapshot_mode=False,
                    circuit_name_prefix=circuit_name_prefix,
                )

            pauli_vals_alphabetical = [
                x[1] for x in sorted(list(zip(pauli_names_temp, pauli_vals_temp)))
            ]
            if not np.all(np.isreal(pauli_vals_alphabetical)):
                warnings.warn(
                    "Computed Pauli expectation value has nonzero "
                    "imaginary part which will be discarded."
                )
            pauli_vals[prep_idx, :, rich_idx, 0] = np.real(pauli_vals_alphabetical)
        key = (suffix, str(bitstring_pair[0]), str(bitstring_pair[1]))
        if key in hybrid_superpos_coeffs.keys():
            if prep_string[-4:] == "xmin":
                pauli_vals[prep_idx] *= 0
            elif prep_string[-5:] == "xplus":
                pauli_vals[prep_idx] *= 2
            else:
                raise ValueError(f"Invalid circuit name: {prep_string}")
        elif not tensor_circuit:
            pauli_vals[prep_idx] *= 1 / 2

    return pauli_vals


# pylint: disable=protected-access
def _eval_each_pauli_with_result(
    tpbgwpo,
    result,
    statevector_mode,
    use_simulator_snapshot_mode=False,
    circuit_name_prefix="",
):
    """Ignores the weights of each pauli operator."""
    if tpbgwpo.is_empty():
        raise QiskitNatureError("Operator is empty, check the operator.")
    if statevector_mode or use_simulator_snapshot_mode:
        raise NotImplementedError()
    num_paulis = len(tpbgwpo._paulis)
    means = np.zeros(num_paulis)
    cov = np.zeros((num_paulis, num_paulis))
    for basis, p_indices in tpbgwpo._basis:
        counts = result.get_counts(circuit_name_prefix + basis.to_label())
        paulis = [tpbgwpo._paulis[idx] for idx in p_indices]
        paulis = [p[1] for p in paulis]  ## DISCARDING THE WEIGHTS
        means_this_basis, cov_this_basis = compute_pauli_means_and_cov_for_one_basis(
            paulis, counts
        )
        for p_idx, p_mean in zip(p_indices, means_this_basis):
            means[p_idx] = p_mean
        cov[np.ix_(p_indices, p_indices)] = cov_this_basis
    return means, cov


def compute_h_schmidt(
    tensor_expvals,
    superpos_expvals,
    w_ij_tensor_weights,
    w_ab_superpos_weights,
    superpos_state_indices,
    asymmetric_bitstrings,
):
    """Computes the schmidt decomposition of the Hamiltonian. TODO checkthis.  # pylint: disable=fixme

    Pauli val arrays contain expectation values <x|P|x> and their standard deviations.
    Axes are [x_idx, P_idx, mean_or_variance]
    W_ij: Coefficients W_ij. Axes: [index of Pauli string for eta, index of Pauli string for tau].
    asymmetric_bitstrings: A boolean which signifies whether the U and V subsystems have
                            different ansatze.
    """
    # Number of tensor stateprep circuits
    num_tensor_terms = int(np.shape(tensor_expvals)[0])

    if asymmetric_bitstrings:
        num_tensor_terms = int(
            num_tensor_terms / 2
        )  # num_tensor_terms should always be even here
        tensor_exp_vals_u = tensor_expvals[:num_tensor_terms, :, 0]
        tensor_exp_vals_v = tensor_expvals[num_tensor_terms:, :, 0]
    else:
        # Use the same expectation values for both subsystem calculations
        tensor_exp_vals_u = tensor_expvals[:, :, 0]
        tensor_exp_vals_v = tensor_expvals[:, :, 0]

    # Calculate the schmidt summation over the U and V subsystems and diagonalize the values
    h_schmidt_diagonal = np.einsum(
        "ij,xi,xj->x",
        w_ij_tensor_weights,
        tensor_exp_vals_u,
        tensor_exp_vals_v,
    )
    h_schmidt = np.diag(h_schmidt_diagonal)

    # If including the +/-Y superpositions (omitted at time of writing
    # since they typically have 0 net contribution) would change this to 4 instead of 2.
    num_lin_combos = 2

    num_superpos_terms = int(np.shape(superpos_expvals)[0])
    if asymmetric_bitstrings:
        num_superpos_terms = int(
            num_superpos_terms / 2
        )  # num_superpos_terms should always be even here
        pvss_u = superpos_expvals[:num_superpos_terms, :, 0]
        pvss_v = superpos_expvals[num_superpos_terms:, :, 0]
    else:
        pvss_u = superpos_expvals[:, :, 0]
        pvss_v = superpos_expvals[:, :, 0]

    # Calculate delta for U subsystem
    p_plus_x_u = pvss_u[0::num_lin_combos, :]
    p_minus_x_u = pvss_u[1::num_lin_combos, :]
    p_delta_x_u = p_plus_x_u - p_minus_x_u

    # Calculate delta for V subsystem
    if asymmetric_bitstrings:
        p_plus_x_v = pvss_v[0::num_lin_combos, :]
        p_minus_x_v = pvss_v[1::num_lin_combos, :]
        p_delta_x_v = p_plus_x_v - p_minus_x_v
    else:
        p_delta_x_v = p_delta_x_u

    h_schmidt_off_diagonals = np.einsum(
        "ab,xa,xb->x", w_ab_superpos_weights, p_delta_x_u, p_delta_x_v
    )

    for element, indices in zip(h_schmidt_off_diagonals, superpos_state_indices):
        h_schmidt[indices] = element
    return h_schmidt  # , H_schmidt_vars)


def get_optimizer_instance(config):
    """Returns optimizer instance based on config."""
    # Addressing some special cases for compatibility with various Qiskit optimizers:
    if config.optimizer_name == "adaptive_SPSA":
        optimizer = SPSA
    else:
        optimizer = getattr(optimizers, config.optimizer_name)
    optimizer_config = {}
    optimizer_arg_names = inspect.signature(optimizer).parameters.keys()
    iter_kw = [kw for kw in ["maxiter", "max_trials"] if kw in optimizer_arg_names][0]
    optimizer_config[iter_kw] = config.maxiter
    if "skip_calibration" in optimizer_arg_names:
        optimizer_config["skip_calibration"] = config.skip_any_optimizer_cal
    if "last_avg" in optimizer_arg_names:
        optimizer_config["last_avg"] = config.spsa_last_average
    if "tol" in optimizer_arg_names:
        optimizer_config["tol"] = config.optimizer_tol
    if "c0" in optimizer_arg_names:
        optimizer_config["c0"] = config.spsa_c0
    if "c1" in optimizer_arg_names:
        optimizer_config["c1"] = config.spsa_c1
    if "learning_rate" in optimizer_arg_names:
        optimizer_config["learning_rate"] = lambda: powerseries(
            config.spsa_c0, 0.602, 0
        )
    if "perturbation" in optimizer_arg_names:
        optimizer_config["perturbation"] = lambda: powerseries(config.spsa_c1, 0.101, 0)

    if config.initial_spsa_iteration_idx:
        if "int_iter" in optimizer_arg_names:
            optimizer_config["int_iter"] = config.initial_spsa_iteration_idx

    if "bootstrap_trials" in optimizer_arg_names:
        optimizer_config["bootstrap_trials"] = config.bootstrap_trials

    Log.log(optimizer_config)
    optimizer_instance = optimizer(**optimizer_config)
    return optimizer_instance
