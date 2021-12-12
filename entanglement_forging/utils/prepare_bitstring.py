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

"""Bitstring functions."""
from qiskit import QuantumCircuit


def prepare_bitstring(bitstring, name=None):
    """Prepares bitstrings."""
    # First bit in bitstring is the first qubit in the circuit.
    qcirc = QuantumCircuit(len(bitstring), name=name)
    for qb_idx, bit in enumerate(bitstring):
        if bit:
            qcirc.x(qb_idx)
    return qcirc
