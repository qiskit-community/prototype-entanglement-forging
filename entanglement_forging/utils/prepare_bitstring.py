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
