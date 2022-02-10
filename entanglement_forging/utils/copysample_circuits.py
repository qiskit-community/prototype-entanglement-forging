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

"""Approximation of a weighted sampling of circuit."""

import re
from collections import Counter

import numpy as np

from entanglement_forging.utils.combined_result import CombinedResult


def copysample_circuits(circuits_to_execute, weights, new_job_size=800):
    """Approximates a weighted sampling of circuits.
    See "Weighted Sampling of Circuits: Implementation" in <https://arxiv.org/abs/2104.10220>."""

    # WARNING! THIS DOES MUTATE THE INPUT CIRCUITS (MODIFIES THEIR NAMES)
    # Could be avoided with a deepcopy, but for now left that out since
    # I didn't need it and worried it might be slow. Could add an argument
    # to optionally trigger the deepcopy if desired.

    # e.g. if expectation values E_i of all circuits are to be
    # added together as sum(w_i * E_i), then to minimize total variance: weights = w_i.

    new_job_size = int(new_job_size)
    num_initial_circuits = len(circuits_to_execute)
    assert (
        new_job_size >= num_initial_circuits
    ), "Try increasing copysample_job_size."  # turn this into a proper RaiseError

    weights = np.array(weights) / np.sum(weights)

    # First ensure everything gets measured at least once
    # (safer to have too-broad tails of sampling distribution).
    # In the future, may want to change this behavior for use-cases
    # w very large numbers of circuits, but analysis will need
    # to be written to accommodate possibility of some circuits
    # being completely absent from results.
    copies_each_circuit = np.ones(num_initial_circuits)
    copies_still_wanted_each_circuit = weights * new_job_size
    copies_still_wanted_each_circuit[copies_still_wanted_each_circuit < 1] = 1
    copies_still_wanted_each_circuit -= 1
    copies_still_wanted_each_circuit *= (new_job_size - num_initial_circuits) / np.sum(
        copies_still_wanted_each_circuit
    )
    num_slots_available = new_job_size - num_initial_circuits

    easy_copies = np.floor(copies_still_wanted_each_circuit)
    copies_each_circuit += easy_copies
    num_slots_available -= np.sum(easy_copies)
    copies_still_wanted_each_circuit %= 1

    if np.sum(copies_still_wanted_each_circuit):
        prob_for_randomly_picking_from_leftovers = (
            copies_still_wanted_each_circuit / np.sum(copies_still_wanted_each_circuit)
        )
        hard_to_fill_idxs = np.random.choice(
            num_initial_circuits,
            p=prob_for_randomly_picking_from_leftovers,
            size=int(round(new_job_size - np.sum(copies_each_circuit))),
            replace=False,
        )
        copies_each_circuit[hard_to_fill_idxs] += 1

    circuit_name_templates = [qc.name + "_copysample{}" for qc in circuits_to_execute]

    for qcirc, num_copies, name_template in zip(
        circuits_to_execute, copies_each_circuit, circuit_name_templates
    ):
        qcirc.name = name_template.format(0)
        if num_copies > 1:
            circuits_to_execute += [
                qcirc.copy(name=name_template.format(i))
                for i in range(1, int(num_copies))
            ]
    return circuits_to_execute


def combine_copysampled_results(aqua_result):
    """Combine results of compysampled circuits."""
    result_each_circuit = aqua_result.results
    original_circuit_names = [
        re.sub("_copysample[0-9]+", "", r.header.name) for r in result_each_circuit
    ]
    deduped_counts = {name: Counter({}) for name in set(original_circuit_names)}
    for name, res in zip(original_circuit_names, result_each_circuit):
        deduped_counts[name] += Counter(aqua_result.get_counts(res.header.name))
    return CombinedResult(
        original_circuit_names,
        [deduped_counts[name] for name in original_circuit_names],
    )
