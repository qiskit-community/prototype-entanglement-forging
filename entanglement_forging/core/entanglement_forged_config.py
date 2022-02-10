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

"""Class for configuration settings."""

import warnings

import numpy as np

from entanglement_forging.utils.log import Log


# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals,too-few-public-methods
class EntanglementForgedConfig:
    """Class for configuration settings.

    Attr:
        backend (Union[Backend, BaseBackend]): Instance of selected backend.
        qubit_layout (NoneType|qiskit.transpiler.Layout|dict|list): Initial position of virtual
                qubits on physical qubits. If this layout makes the circuit compatible with the
                coupling_map constraints, it will be used.
        initial_params (NoneType or list of int): A list specifying the initial optimization
                parameters.
        maxiter (int): Maximum number of optimizer iterations to perform.
        optimizer_name (str): e.g. 'SPSA', 'ADAM', 'NELDER_MEAD', 'COBYLA', 'L_BFGS_B', 'SLSQP' ...
        optimizer_tol' (float): Optimizer tolerance, e.g. 1e-6.
        skip_any_optimizer_cal (bool): Setting passed to any optimizer with a 'skip_calibration'
                argument.
        spsa_last_average (int): Number of times to average over final SPSA evaluations to
                determine optimal parameters (only used for SPSA).
        initial_spsa_iteration_idx (int): Iteration index to resume interrupted
                VQE run (only used for SPSA).
        spsa_c0 (float): The initial parameter 'a'. Determines step size to update
                parameters in SPSA (only used for SPSA).
        spsa_c1 (float): The initial parameter 'c'. The step size used to
                approximate gradient in SPSA (only used for SPSA).
        max_evals_grouped (int): Maximum number of evaluations performed simultaneously.
        rep_delay (float): Delay between programs in seconds.
        shots (int): The total number of shots for the simulation
                (overwritten for the statevector backend).
        fix_first_bitstring (bool): Bypasses computation of first bitstring and replaces
                result with HF energy. Can speed up the computation, but requires ansatz
                that leaves the HF state unchanged under var_form. bootstrap_trials (int):
                A setting for generating error bars (not used for the statevector backend).
        copysample_job_size (int or NoneType): A setting to approximately realize weighted
                sampling of circuits according to their relative significance
                (Schmidt coefficients). This number should be bigger than the number
                of unique circuits running (not used for the statevector backend).
        meas_error_mit (bool): Performs measurement error mitigation
                (not used for the statevector backend).
        meas_error_shots (int): The number of shots for measurement error mitigation
                (not used for the statevector backend).
        meas_error_refresh_period_minutes (float): How often to refresh the calibration
                matrix in measurement error mitigation, in minutes
                (not used for the statevector backend).
        zero_noise_extrap (bool): Linear extrapolation for gate error mitigation
                (ignored for the statevector backend)
    """

    def __init__(self,
                 backend,
                 qubit_layout=None,
                 initial_params=None,
                 maxiter=100,
                 optimizer_name='SPSA',
                 optimizer_tol=1e-6,
                 skip_any_optimizer_cal=True,
                 spsa_last_average=10,
                 initial_spsa_iteration_idx=0,
                 spsa_c0=2 * 2 * np.pi,
                 spsa_c1=0.1,
                 max_evals_grouped=99,
                 rep_delay=100e-6,
                 shots=1024,
                 fix_first_bitstring=False,
                 copysample_job_size=None,
                 meas_error_mit=False,
                 meas_error_shots=None,
                 meas_error_refresh_period_minutes=30,
                 bootstrap_trials=None,
                 zero_noise_extrap=False,
                 ):
        """ The constructor for the EntanglementForgedConfig class. """
        statevector_sims = ['statevector_simulator', 'aer_simulator_statevector']
        self.backend = backend
        self.backend_name = self.backend.configuration().backend_name
        self.qubit_layout = qubit_layout
        self.initial_params = initial_params
        self.maxiter = maxiter
        self.optimizer_name = optimizer_name
        self.optimizer_tol = optimizer_tol
        self.skip_any_optimizer_cal = skip_any_optimizer_cal
        self.spsa_last_average = spsa_last_average
        self.initial_spsa_iteration_idx = initial_spsa_iteration_idx
        self.spsa_c0 = spsa_c0
        self.spsa_c1 = spsa_c1
        self.max_evals_grouped = max_evals_grouped
        self.rep_delay = rep_delay
        self.shots = shots if self.backend_name not in statevector_sims else 1
        self.fix_first_bitstring = fix_first_bitstring
        self.bootstrap_trials = bootstrap_trials
        self.copysample_job_size = copysample_job_size \
            if self.backend_name not in statevector_sims else None
        self.meas_error_mit = meas_error_mit
        self.meas_error_shots = meas_error_shots
        self.meas_error_refresh_period_minutes = meas_error_refresh_period_minutes
        self.zero_noise_extrap = [1, 3] if zero_noise_extrap else [1]
        self.validate()

    def validate(self):
        """Validates the configuration settings."""
        statevector_sims = ['statevector_simulator', 'aer_simulator_statevector']
        # TODO check this. I might have mixed up the error mitigations  # pylint: disable=fixme
        if self.meas_error_mit and self.zero_noise_extrap is None:
            raise ValueError(
                'You have set meas_error_mit == True and must therefore '
                'specify zero_noise_extrap, e.g., zero_noise_extrap = [1,3].')

        if self.backend_name not in statevector_sims \
                and self.backend_name != 'qasm_simulator' \
                and self.backend is None:
            raise ValueError(
                'You are using a real backend, so you must specify provider.')

        if self.meas_error_mit and self.qubit_layout is None:
            raise ValueError(
                'If you would like to use measurement error mitigation, '
                'you must specify a qubit_layout.')

        if self.backend_name == 'statevector_simulator' and self.shots != 1:
            warnings.warn(
                "Ignoring setting for 'shots' as it is not used for the statevector simulator.")

        if self.backend_name not in statevector_sims \
                and self.copysample_job_size is not None:
            warnings.warn(
                "Ignoring setting for 'copysample_job_size' as it is "
                "not used for the statevector simulator.")

        if self.backend_name in statevector_sims and self.meas_error_mit is True:
            warnings.warn(
                "Ignoring setting for 'meas_error_mit' as it is "
                "not used for the statevector simulator.")

        if self.backend_name in statevector_sims \
                and self.meas_error_refresh_period_minutes is not None:
            warnings.warn(
                "Ignoring setting for 'meas_error_refresh_period_minutes' "
                "as it is not used for the statevector simulator.")

        if self.backend_name in statevector_sims and self.zero_noise_extrap != [
            1]:
            warnings.warn(
                "Ignoring setting for 'zero_noise_extrap' as it is "
                "not used for the statevector simulator.")

        Log.log("Configuration settings are valid.")
