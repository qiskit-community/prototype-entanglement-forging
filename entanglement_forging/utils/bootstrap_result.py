"""Bootstrap result."""

from collections import Counter

import numpy as np

from entanglement_forging.utils.combined_result import CombinedResult


def resample_counts(counts_dict):
    """Returns resampled counts."""
    labels = list(counts_dict.keys())
    counts = list(counts_dict.values())
    total = int(sum(counts))
    counts = np.array(counts, dtype=float)
    counts /= counts.sum()
    new_counts_dict = dict(Counter(np.random.default_rng().choice(labels, total, p=counts)))
    return new_counts_dict


def resample_result(result):
    """Returns resampled results."""
    # TODO Optimize this for speed or move out of experimental routine into post-processing  # pylint: disable=fixme
    return CombinedResult(result.results.keys(),
                          [resample_counts(counts) for counts in result.results.values()])
