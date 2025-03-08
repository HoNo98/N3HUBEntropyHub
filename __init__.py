"""
N3HUBEntropyHub

A package for surrogate-based entropy analysis using EntropyHub on NWB spike data.

This package provides a set of surrogate methods and a collection of entropy measures
from the EntropyHub library. It enables you to compute various entropy metrics on
binned spike data extracted from NWB files using different surrogate transformations.

Public API:
    original          - Processes the original spike data without any surrogate transformation.
    global_shuffle    - Applies a global shuffle to the binned spike data.
    circular_shift    - Applies a circular shift to the binned spike data.
    local_jitter      - Applies a local jitter shuffle to the binned spike data.
    phase_random      - Applies phase randomization to the binned spike data.
    pres_spike_sum    - Applies a spike sum permutation to the binned spike data.
    isi_based         - Applies an ISI-based (Inter-Spike Interval) surrogate to the spike data.

Usage Example:
    >>> import N3HUBEntropyHub as n3hub
    >>> result = n3hub.global_shuffle.permutation_entropy("path/to/your/file.nwb", order=3, delay=1, bin_size=0.01)
    >>> print("Global Shuffle - Permutation Entropy:", result)
    
For detailed usage instructions and additional configuration options,
please refer to the package documentation.
"""

__version__ = "0.1.0"
__author__ = "Hossein Nowrouzi-Nezhad <hnowrozinezhad@mail.com>"

from .core import original, global_shuffle, circular_shift, local_jitter, phase_random, pres_spike_sum, isi_based

