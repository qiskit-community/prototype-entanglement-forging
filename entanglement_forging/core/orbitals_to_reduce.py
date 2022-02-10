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

"""Orbitals to reduce."""

import numpy as np


class OrbitalsToReduce:
    """A class that describes which orbitals (all, occupied and virtual) to reduce.
    Attributes:
        all (list): All orbitals to be reduced.
        occupied (list): Only the occupied orbitals to be reduced.
        virtual (list): Only the virtual orbitals to be reduced.
    """
    def __init__(self, all_orbitals_to_reduce, problem):
        """Initialize the orbitals to reduce.
        Args:
            all_orbitals_to_reduce (list): All orbitals to be reduced.
            problem (ElectronicStructureProblem): Problem class which holds driver
        """
        self.problem = problem
        self.all = all_orbitals_to_reduce

    def occupied(self):
        """Returns occupied orbitals."""
        orbitals_to_reduce_array = np.asarray(self.all)
        particle_number = self.problem.grouped_property.get_property("ParticleNumber")
        num_alpha = particle_number.num_alpha
        return orbitals_to_reduce_array[orbitals_to_reduce_array < num_alpha].tolist()

    def virtual(self):
        """Returns virtual orbitals."""
        orbitals_to_reduce_array = np.asarray(self.all)
        particle_number = self.problem.grouped_property.get_property("ParticleNumber")
        num_alpha = particle_number.num_alpha
        return orbitals_to_reduce_array[orbitals_to_reduce_array >= num_alpha].tolist()
