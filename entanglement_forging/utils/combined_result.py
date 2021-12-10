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

"""Combined result."""


# pylint: disable=too-few-public-methods
class CombinedResult:
    """CombinedResult. Giving just enough functionality for Aqua to process it OK"""

    def __init__(self, expt_names, counts_each_expt):
        """CombinedResult.

        Args:
            expt_names:
            counts_each_expt:
        """
        self.counts_each_expt = counts_each_expt
        self.results = dict(zip(expt_names, counts_each_expt))

    def get_counts(self, experiment):
        """Returns counts."""
        if isinstance(experiment, int):
            counts = self.counts_each_expt[experiment]
        else:
            counts = self.results[experiment]
        return counts
