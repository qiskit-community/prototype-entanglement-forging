"""Orbitals to reduce."""

import numpy as np


class OrbitalsToReduce:
    """A class that describes which orbitals (all, occupied and virtual) to reduce.

    Attributes:
        all (list): All orbitals to be reduced.
        occupied (list): Only the occupied orbitals to be reduced.
        virtual (list): Only the virtual orbitals to be reduced.
    """

    def __init__(self, all_orbitals_to_reduce, qmolecule):
        """Initialize the orbitals to reduce.

        Args:
            all_orbitals_to_reduce (list): All orbitals to be reduced.
            qmolecule (qiskit_nature.drivers.QMolecule): Molecule data class containing driver
              result.
        """
        self.qmolecule = qmolecule
        self.all = all_orbitals_to_reduce

    def occupied(self):
        """Returns occupied orbitals."""
        orbitals_to_reduce_array = np.asarray(self.all)
        return orbitals_to_reduce_array[
            orbitals_to_reduce_array < self.qmolecule.num_alpha].tolist()

    def virtual(self):
        """Returns virtual orbitals."""
        orbitals_to_reduce_array = np.asarray(self.all)
        return orbitals_to_reduce_array[
            orbitals_to_reduce_array >= self.qmolecule.num_alpha].tolist()
