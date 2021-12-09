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
