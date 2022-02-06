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

    def __init__(self, all_orbitals_to_reduce, driver_result):
        self.driver_result = driver_result
        self.all = all_orbitals_to_reduce

    def occupied(self):
        """Returns occupied orbitals."""
        orbitals_to_reduce_array = np.asarray(self.all)
        num_alpha = np.shape(self.driver_result._properties['ElectronicBasisTransform'].coeff_alpha)[0]
        return orbitals_to_reduce_array[
            orbitals_to_reduce_array < num_alpha].tolist()

    def virtual(self):
        """Returns virtual orbitals."""
        orbitals_to_reduce_array = np.asarray(self.all)
        num_alpha = np.shape(self.driver_result._properties['ElectronicBasisTransform'].coeff_alpha)[0]
        return orbitals_to_reduce_array[
            orbitals_to_reduce_array >= num_alpha].tolist()
