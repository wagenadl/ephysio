#!/usr/bin/python3

import numpy as np


class SpikeStats:
    def __init__(self, stim_s, spks_s, stim_idx=None, selector=None):
        '''SPIKESTATS - Class to get rasters, psth, per-stimulus type counts, etc.
        SPIKESTATS(stim_s, spks_s) constructs a SPIKESTATS object.
        STIM_S must be stimulus times in seconds.
        SPKS_S must be a dict of spike unit ids to time stamps in seconds.
        Optionally, STIM_IDX may specify a list of stimulus types for each stimulus. It must have the same
        length as STIM_S. Each entry is either None or a tuple of index values. All such tuples must have
        the same length. Values inside must count from zero; we automatically calculate the highest occurring
        value in each dimension.
        Optional argument SELECTOR specifies a function that must return True or False given a STIM_IDX value
        to include or exclude that trial. For instance, SELECTOR=lambda idx: idx is not None would drop all gray
        trials in our usual indexing convention.
        '''
        try:
            self.stim_s = stim_s.to_numpy() # in case it's a panda series
        except:
            self.stim_s = stim_s
        self.spks_s = spks_s
        self.stim_idx = stim_idx
        if stim_idx is not None:
            if selector is not None:
                self.stim_s = np.array([t for t,idx in zip(stim_s, stim_idx) if selector(idx)])
                self.stim_idx = [idx for idx in stim_idx if selector(idx)]
            ndims = 0
            for idx in stim_idx:
                if idx is not None:
                    if type(idx)==list or type(idx)==tuple:
                        ndims = len(idx)
                        break
            shape = np.zeros(ndims, np.int)
            for idx in stim_idx:
                if idx is not None:
                    for dim in range(ndims):
                        if idx[dim] >= shape[dim]:
                            shape[dim] = idx[dim] + 1
            self.idx_shape = shape
        else:
            self.idx_shape = None

    def latencies(self, celid, dt_start_ms=-50, dt_end_ms=150):
        '''LATENCIES - Extract peristimulus latencies for all spikes for a given neuron
        lat, tri = LATENCIES(celid, dt_start_ms, dt_end_ms) extracts latencies (in ms)
        and trial numbers for all of the spikes associated with the given CELID.
        We return latencies in the interval from DT_start_ms to DT_end_ms. It is legit (and common)
        for DT_start_ms to be negative.
        Note that undefined behavior results if the interval (DT_start_ms, DT_end_ms) is longer than
        the shortest interval between stimuli. In particular, we are not smart enough to return a
        spike twice in that case, once for each stimulus for which the interval encompasses the spike.
        '''

        t_spk_s = self.spks_s[celid]
        tri = np.searchsorted(self.stim_s + dt_start_ms / 1e3,
                              t_spk_s)  # this returns tri before which given spk should be inserted in list
        tri[tri > 0] -= 1
        lat_ms = 1e3 * (t_spk_s - self.stim_s[tri])
        keep = np.logical_and(lat_ms >= dt_start_ms, lat_ms < dt_end_ms)
        return lat_ms[keep], tri[keep]

    def spikecounts(self, celid, dt_start_ms=-50, dt_end_ms=150):
        '''SPIKECOUNTS - Count spikes in each trial
        nnn = SPIKECOUNTS(celid, dt_start_ms=-50, dt_end_ms=150) counts the number
        of spikes in individual trials and returns a numpy array.
        Arguments are as for LATENCIES, but caveat about overlapping trials does not apply.'''

        idxstart = np.searchsorted(self.spks_s[celid], self.stim_s + dt_start_ms / 1e3)
        idxend = np.searchsorted(self.spks_s[celid], self.stim_s + dt_end_ms / 1e3)
        nn = idxend - idxstart
        # lat, tri = latencies(stim_s, spks, celid, dt_start_ms=dt_start_ms, dt_end_ms=dt_end_ms)
        # nn = np.bincount(tri, minlength=len(stim_s))
        return nn

    def psth(self, celid, dt_start_ms=-50, dt_end_ms=150, binsize_ms=5, pertrial=False):
        '''PSTH - Count spikes in each trial
        nnn, bin_ms = PSTH(celid, dt_start_ms=-50, dt_end_ms=150, binsize_ms) counts up
        the number of spikes in latency bins across all trials.
        BIN_MS returns the *centers* of the latency bins.
        To get results for individual trials (as a LATENCYxTRIAL array), set PERTRIAL=True.
        To get averages per trial, divide by NTRIALS().
        To convert to firing rates, divide by NTRIALS() and by the BINSIZE.
        Arguments are as for LATENCIES. Caveat in LATENCIES applies.'''

        lat, tri = self.latencies(celid, dt_start_ms=dt_start_ms, dt_end_ms=dt_end_ms)
        bin_ms = np.arange(dt_start_ms, dt_end_ms + .0001, binsize_ms)
        bin_tri = np.arange(len(self.stim_s))
        cnts, _, _ = np.histogram2d(lat, tri, (bin_ms, bin_tri))
        if not pertrial:
            cnts = np.sum(cnts, 1)
        return cnts, (bin_ms[:-1] + bin_ms[1:]) / 2

    def ntrials(self):
        '''NTRIALS - Total number of trials
        NTRIALS() returns the total number of trials in our dataset.'''
        return len(self.stim_s)

    def trialcounts(self):
        '''TRIALCOUNTS - Count trials by type
        nnn = TRIALCOUNTS() returns a tensor of occurrences of each trial type, or None if no trial
        types were passed into the constructor.
        '''
        if self.stim_idx is None:
            raise Exception("Calculating trial counts by type requires knowing a trial type (idx)")
        nnn = np.zeros(self.idx_shape, np.int)
        for idx in self.stim_idx:
            if idx is not None:
                nnn[idx] += 1
        return nnn

    def totalspikecountsbytrialtype(self, celid, dt_start_ms=-50, dt_end_ms=150):
        '''TOTALSPIKECOUNTSBYTRIALTYPE - As SPIKECOUNTS, but accumulates over trials of the same type.
        Use TRIALCOUNTS to find out how many trials of the type there are, if you need to get
        averagecounts.'''
        if self.stim_idx is None:
            raise Exception("Calculating spike counts by trial type requires knowing a trial type (idx)")
        nn = self.spikecounts(celid, dt_start_ms, dt_end_ms)
        nnn = np.zeros(self.idx_shape, np.int)
        for n, idx in zip(nn, self.stim_idx):
            if idx is not None:
                nnn[idx] += n
        return nnn
