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

"""Execution subroutines."""

import time

import numpy as np
from qiskit import assemble
from qiskit.providers.ibmq.job import (
    IBMQJobFailureError,
    IBMQJobApiError,
    IBMQJobInvalidStateError,
)

from entanglement_forging.utils.legacy.common import measure_pauli_z, covariance


def compute_pauli_means_and_cov_for_one_basis(paulis, counts):
    """Compute Pauli means and cov for one basis."""
    means = np.array([measure_pauli_z(counts, pauli) for pauli in paulis])
    cov = np.array(
        [
            [
                covariance(counts, pauli_1, pauli_2, avg_1, avg_2)
                for pauli_2, avg_2 in zip(paulis, means)
            ]
            for pauli_1, avg_1 in zip(paulis, means)
        ]
    )
    return means, cov


def execute_with_retry(circuits, backend, shots, rep_delay=None, noise_model=None):
    """Executes job with retry."""
    global result  # pylint: disable=global-variable-undefined,invalid-name
    trials = 0
    ran_job_ok = False
    while not ran_job_ok:
        try:
            qobj = assemble(
                circuits,
                backend=backend,
                shots=shots,
                rep_delay=rep_delay,
                noise_model=noise_model,
                seed_simulator=42,
            )
            if backend.name() in [
                "statevector_simulator",
                "qasm_simulator",
                "aer_simulator_statevector",
            ]:
                job = backend.run(qobj, noise_model=noise_model)
            else:
                job = backend.run(qobj)
            result = job.result()
            ran_job_ok = True
        except (IBMQJobFailureError, IBMQJobApiError, IBMQJobInvalidStateError) as err:
            print("Error running job, will retry in 5 mins.")
            print("Error:", err)
            # Wait 5 mins and try again. Hopefully this handles network outages etc,
            # and also if user cancels a (stuck) job through IQX.
            # Add more error types to the exception as new ones crop up (as appropriate).
            time.sleep(300)
            trials += 1
            # pylint: disable=raise-missing-from
            if trials > 100:
                raise RuntimeError(
                    "Timed out trying to run job successfully (100 attempts)"
                )
    return result


def reduce_bitstrings(bitstrings, orbitals_to_reduce):
    """Returns reduced bitstrings."""
    return np.delete(bitstrings, orbitals_to_reduce, axis=-1).tolist()
