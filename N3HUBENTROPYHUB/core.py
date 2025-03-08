# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 16:51:06 2025

@author: Hossein Nowrouzi-Nezhad
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from pynwb import NWBHDF5IO
from dandi.download import download
import entropyhub as eh  

###############################################################################
# SURROGATE FUNCTIONS
###############################################################################
def shuffle_time_global(data):
    return np.random.permutation(data)

def shuffle_time_circular(data):
    shift = np.random.randint(len(data))
    return np.roll(data, shift)

def shuffle_local_jitter(data, window_size=50):
    x = data.copy()
    n = len(x)
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        seg = x[start:end]
        np.random.shuffle(seg)
        x[start:end] = seg
    return x

def shuffle_isi_based(spike_times):
    st = np.sort(spike_times)
    if len(st) < 2:
        return st
    isis = np.diff(st)
    np.random.shuffle(isis)
    rebuilt = [st[0]]
    for isi in isis:
        rebuilt.append(rebuilt[-1] + isi)
    return np.array(rebuilt)

def shuffle_phase_random(data):
    x = data.astype(float)
    n = len(x)
    X = np.fft.fft(x)
    half_n = n // 2
    freq_inds = np.arange(1, half_n)
    random_phases = np.exp(1j * 2 * np.pi * np.random.rand(len(freq_inds)))
    X[freq_inds] *= random_phases
    for i in freq_inds:
        X[n - i] = np.conjugate(X[i])
    x_sur = np.fft.ifft(X).real
    x_sur = np.maximum(np.round(x_sur), 0).astype(int)
    return x_sur

def shuffle_pres_spike_sum(data):
    return np.random.permutation(data)

###############################################################################
# NWB DATA HANDLING FUNCTIONS
###############################################################################
def get_spike_data(nwbfile, max_samples=1_000_000):
    if nwbfile.units is not None and len(nwbfile.units) > 0:
        if 'spike_times' in nwbfile.units.colnames:
            spike_times_col = nwbfile.units['spike_times']
            if len(spike_times_col) > 0:
                spike_times_array = np.array(spike_times_col[0])
                if len(spike_times_array) > 1:
                    return spike_times_array
    
    if "ophys" in nwbfile.processing:
        ophys_mod = nwbfile.processing["ophys"]
        for iface_name, iface in ophys_mod.data_interfaces.items():
            if "df" in iface_name.lower() or "df_f" in iface_name.lower():
                if hasattr(iface, 'roi_response_series'):
                    keys = list(iface.roi_response_series.keys())
                    if len(keys) > 0:
                        rrs = iface.roi_response_series[keys[0]]
                        df_data = np.array(rrs.data[:max_samples])
                        if df_data.ndim > 1:
                            df_data = df_data[:, 0]
                        thr = np.mean(df_data) + 3 * np.std(df_data)
                        rate = getattr(rrs, 'rate', None)
                        timestamps = getattr(rrs, 'timestamps', None)
                        if rate is not None:
                            dt = 1.0 / rate
                        elif timestamps is not None and len(timestamps) == len(df_data):
                            spike_times = [timestamps[i] for i, val in enumerate(df_data) if val > thr]
                            return np.array(spike_times)
                        else:
                            dt = 1.0 / 30.0
                        spike_times = [i * dt for i, val in enumerate(df_data) if val > thr]
                        return np.array(spike_times)
    return np.array([])

def bin_spike_times(spike_times, bin_size=0.01):
    if len(spike_times) < 2:
        return np.array([])
    duration = spike_times[-1]
    n_bins = int(np.ceil(duration / bin_size))
    binned = np.zeros(n_bins, dtype=int)
    bin_idx = (spike_times / bin_size).astype(int)
    bin_idx = bin_idx[bin_idx < n_bins]
    for i in bin_idx:
        binned[i] += 1
    return binned

def _load_binned_data(path, bin_size=0.01):
    with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
        nwbfile = io.read()
        spike_times = get_spike_data(nwbfile)
    binned = bin_spike_times(spike_times, bin_size)
    return spike_times, binned

###############################################################################
# ENTROPY FUNCTIONS DICTIONARY (Mapping names to EntropyHub functions)
###############################################################################
entropy_functions = {
    "permutation_entropy": eh.permutation_entropy,
    "approximate_entropy": eh.approximate_entropy,
    "sample_entropy": eh.sample_entropy,
    "fuzzy_entropy": eh.fuzzy_entropy,
    "spectral_entropy": eh.spectral_entropy,
    "dispersion_entropy": eh.dispersion_entropy,
    "renyi_entropy": eh.renyi_entropy,
    "svd_entropy": eh.svd_entropy,
    "distribution_entropy": eh.distribution_entropy,
    "multiscale_entropy": eh.multiscale_entropy,
    "composite_multiscale_entropy": eh.composite_multiscale_entropy,
    "refined_composite_multiscale_entropy": eh.refined_composite_multiscale_entropy,
    "symbolic_entropy": eh.symbolic_entropy,
    "permutation_lempel_ziv_complexity": eh.permutation_lempel_ziv_complexity,
    "conditional_entropy": eh.conditional_entropy,
    "cross_entropy": eh.cross_entropy,
    "shannon_entropy": eh.shannon_entropy,
    "energy_entropy": eh.energy_entropy,
    "differential_entropy": eh.differential_entropy,
    "wavelet_entropy": eh.wavelet_entropy
}

###############################################################################
# BASE CLASS FOR SURROGATE METHODS
###############################################################################
class SurrogateBase:
    surrogate_function = None  

    @classmethod
    def _compute_entropy(cls, path, entropy_name, bin_size=0.01, **kwargs):
    
        _, data = _load_binned_data(path, bin_size)
  
        if cls.surrogate_function is not None:
            data = cls.surrogate_function(data)
          
        return entropy_functions[entropy_name](data, **kwargs)

###############################################################################
# SURROGATE CLASSES
###############################################################################
class Original(SurrogateBase):
    surrogate_function = None

class GlobalShuffle(SurrogateBase):
    surrogate_function = staticmethod(shuffle_time_global)

class CircularShift(SurrogateBase):
    surrogate_function = staticmethod(shuffle_time_circular)

class LocalJitter(SurrogateBase):
    surrogate_function = staticmethod(shuffle_local_jitter)

class PhaseRandom(SurrogateBase):
    surrogate_function = staticmethod(shuffle_phase_random)

class PresSpikeSum(SurrogateBase):
    surrogate_function = staticmethod(shuffle_pres_spike_sum)

class ISIBased:
    @classmethod
    def _compute_entropy(cls, path, entropy_name, bin_size=0.01, **kwargs):
        with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
            nwbfile = io.read()
            spike_times = get_spike_data(nwbfile)
        surrogate_spike_times = shuffle_isi_based(spike_times)
        data = bin_spike_times(surrogate_spike_times, bin_size)
        return entropy_functions[entropy_name](data, **kwargs)

###############################################################################
# ADD ENTROPY METHODS TO EACH CLASS
###############################################################################
def make_entropy_method(ent_name):
    @classmethod
    def method(cls, path, bin_size=0.01, **kwargs):
        return cls._compute_entropy(path, ent_name, bin_size, **kwargs)
    return method

for name in entropy_functions.keys():
    setattr(Original, name, make_entropy_method(name))
    setattr(GlobalShuffle, name, make_entropy_method(name))
    setattr(CircularShift, name, make_entropy_method(name))
    setattr(LocalJitter, name, make_entropy_method(name))
    setattr(PhaseRandom, name, make_entropy_method(name))
    setattr(PresSpikeSum, name, make_entropy_method(name))
    setattr(ISIBased, name, make_entropy_method(name))

###############################################################################
# EXPOSE NAMESPACE-LEVEL ALIASES FOR USER CONVENIENCE
###############################################################################
original = Original
global_shuffle = GlobalShuffle
circular_shift = CircularShift
local_jitter = LocalJitter
phase_random = PhaseRandom
pres_spike_sum = PresSpikeSum
isi_based = ISIBased
