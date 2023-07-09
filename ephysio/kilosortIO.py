import numpy as np
import csv
import os
import re

class PhyLogReader:
    def __init__(self, root):
        self.sources = {}
        fn = f"{root}/phy.log"
        if os.path.exists(fn):
            with open(fn) as fd:
                for line in fd.readlines():
                    m = re.search("Merge.* (\\d+)[^0-9]+(\\d+)[^0-9]+(\\d+)\\.", line)
                    if m:
                        dst = m.group(3)
                        src1 = m.group(1)
                        src2 = m.group(2)
                        if dst not in self.sources:
                            self.sources[dst] = []
                            if src1 in self.sources:
                                self.sources[dst] += self.sources[src1]
                            else:
                                self.sources[dst].append(src1)
                            if src2 in self.sources:
                                self.sources[dst] += self.sources[src2]
                            else:
                                self.sources[dst].append(src2)

    def ultimateSource(self, clusterid):
        if clusterid in self.sources:
            return self.sources[clusterid]
        else:
            return []




class Reader:
    def __init__(self, folder, usecurated=True):
        '''INIT - Load kilosort data'''
        self.spk_tms = np.load(f'{folder}/spike_times.npy').flatten() # nspikes: timestamps in samples
        self.spk_cls = np.load(f'{folder}/spike_templates.npy').flatten() # nspikes: cluster numbers
        self.tmpls = np.load(f'{folder}/templates.npy') # ncluster x template_len x nchannels
        # Note that channels are not the same as electrodes: "dead" electrodes are not
        # represented as channels
        self.chan2elc = np.load(f'{folder}/channel_map.npy').flatten()
        self.elc2chan = { e: c for c, e in enumerate(self.chan2elc) }

        self.chanpos = np.load(f'{folder}/channel_positions.npy')
        self.kslabel = {}
        with open(f'{folder}/cluster_KSLabel.tsv') as f:
            rdr = csv.reader(f, delimiter='\t')
            rows = [x for x in rdr]
            for row in rows[1:]:
                self.kslabel[int(row[0])] = row[1]
        if usecurated:
            if os.path.exists(f'{folder}/cluster_group.tsv'):
                print("Using cluster group labels from phy2 to override ks labels")
                with open(f'{folder}/cluster_group.tsv') as f:
                    rdr = csv.reader(f, delimiter='\t')
                    rows = [x for x in rdr]
                    for row in rows[1:]:
                        self.kslabel[int(row[0])] = row[1]
            else:
                print(f"(No cluster group labels from phy2 in {folder}; using original ks labels.)")
        self.cnt = None

        self.elc4clust = {}
        if os.path.exists(f'{folder}/cluster_info.tsv'):
            with open(f'{folder}/cluster_info.tsv') as f:
                rdr = csv.reader(f, delimiter='\t')
                rows = [x for x in rdr]
                hdr = rows[0]
                idx = hdr.index('ch') # despite the header, DAW believes this
                                      # is an electrode number, not a ks channel
                for row in rows[1:]:
                    self.elc4clust[int(row[0])] = int(row[idx])

    def spikecount_unit(self, u=None):
        '''SPIKECOUNT_UNIT - Return number of spikes per unit
        SPIKECOUNT_UNIT() returns an array of spike counts for each unit.
        SPIKECOUNT_UNIT(u) returns the spike count for the given unit.'''
        if self.cnt is None:
            self.cnt, _ = np.histogram(self.spk_cls, np.arange(len(self.kslabel) + 1))
        if u is None:
            return self.cnt
        else:
            return self.cnt[u]

    def tipdist_electrode(self, elcs):
        '''TIPDIST_ELECTRODE - Distance between electrode and probe tip
        TIPDIST_ELECTRODE(elcs) returns the distance in microns between
        the given electrode(s) and the tip of the probe.
        See also: TIPDIST_UNIT, XPOS_ELECTRODE.'''
        chans = np.array([self.elc2chan[elc] for elc in np.array(elcs)])
        if len(chans):
            return self.chanpos[chans,1]
        else:
            return np.array([])

    def tipdist_unit(self, u, useavg=False):
        '''TIPDIST_UNIT - Distance between unit and probe tip
        TIPDIST_UNIT(u) returns the distance between the "key" electrode
        for the given unit and the tip of the probe, in microns.
        If USEAVG is True, returns the centroid distance of all the 
        electrodes that participate in the unit.
        See also: TIPDIST_ELECTRODE, XPOS_UNIT.'''
        elcs = self.electrodesforcluster(u)
        dd = self.tipdist_electrode(elcs)
        if useavg:
            return np.mean(dd)
        elif len(dd):
            return dd[0]
        else:
            return np.nan

    def xpos_electrode(self, elcs):
        '''XPOS_ELECTRODE - Transverse location of electrode
        XPOS_ELECTRODE(elcs) returns an array containing the transverse 
        locations of the given electrodes, i.e., in the direction 
        orthogonal to the probe's main axis, in microns.
        See also: XPOS_UNIT, TIPDIST_ELECTRODE.'''
        chans = np.array([self.elc2chan[elc] for elc in np.array(elcs)])
        if len(chans):
            return self.chanpos[chans,0]
        else:
            return np.array([])

    def xpos_unit(self, u, useavg=False):
        '''XPOS_UNIT - Transverse location of unit
        XPOS_UNIT(u) returns the transverse location of the "key"
        electrode for the given unit, i.e., in the direction 
        orthogonal to the probe's main axis, in microns.
        If USEAVG is True, returns the centroid distance of all the 
        electrodes that participate in the unit.
        See also: XPOS_ELECTRODE, TIPDIST_UNIT.'''
        elcs = self.electrodesforcluster(u)
        dd = self.xpos_electrode(elcs)
        if useavg:
            return np.mean(dd)
        elif len(dd):
            return dd[0]
        else:
            return np.nan

    def allspikes(self):
        '''ALLSPIKES - Times and cluster numbers for all spikes
        tms, cls = ALLSPIKES() returns time stamps in samples and cluster 
        numbers (counted from 0) as two numpy vectors.'''
        return self.spk_tms, self.spk_cls

    def nclusters(self):
        '''NCLUSTERS - Return number of clusters'''
        return self.tmpls.shape[0]

    def nchannels(self):
        '''NCHANELS - Return number of channels'''
        return len(self.chan2elc)

    def spikesforcluster(self, k):
        '''SPIKESFORCLUSTER - Times of spikes for given cluster
        tms = SPIKESFORCLUSTER(k) returns the spike times (in samples) 
        for cluster k.'''
        return self.spk_tms[self.spk_cls==k]

    def spikesbycluster(self, label=None):
        '''SPIKESBYCLUSTER - Map of spike times for all clusters
        SPIKEBYCLUSTER() returns a dict of cluster number to vector of 
        spike times.
        Optional argument LABEL specifies that clusters must have the given 
        label to be included in the dict.'''

        # The following clever trick is due to
        # https://stackoverflow.com/questions/68331835/split-numpy-2d-array-based-on-separate-label-array
        val, idx, cnt = np.unique(self.spk_cls,
                                  return_counts=True, return_inverse=True)
        # val is a vector of all the cluster numbers that occur.
        ttt = np.split(self.spk_tms[idx.argsort()], cnt.cumsum()[:-1])
        # ttt is a list of vectors of spike times per cluster.
        if label is None:
            # Construct a simply dictionary of cluster number to time vector.
            return { v: tt for v, tt in zip(val, ttt) }
        else:
            # Same, but only for clusters we care about.
            # (Thanks to the unique/argsort/split trick, this function is
            # now fast, so there is no need to pre-filter.)
            msk = self.labeledclusters(label)
            return { v: tt for v, tt in zip(val, ttt) if msk[v] }            


    def templateforcluster(self, k):
        '''TEMPLATEFORCLUSTER - Return template waveform for given cluster
        Result is TxC where T is the length (in samples) of the template
        and C is number of channels.'''
        return self.tmpls[k,:,:]

    def labeledclusters(self, label='good'):
        '''LABELEDCLUSTERS - Vector indicating clusters with given label
        v = LABELEDCLUSTERS() returns a boolean vector in which each entry 
        is True iff the corresponding cluster has the label "good". 
        Optional argument LABEL specifies another label to hunt for.
        Note that LABELEDCLUSTERS returns a "mask vector" rather than a 
        plain list, but of course np.nonzero(v)[0] is your friend if you
        want a list.'''
        v = np.zeros(self.nclusters(), bool)
        for k in range(self.nclusters()):
            if k in self.kslabel and self.kslabel[k]==label:
                v[k] = True
        return v

    def electrodesforcombinedclusters(self, kk, thresh, return_weights, win):
        elcw = {}
        for k in kk:
            for e, w in zip(self.electrodesforcluster(k, thresh,
                                                      return_weights=True,
                                                      win=win)):
                if e in elcw:
                    elcw[e] = max(w, elcw[e])
                else:
                    elcw[e] = w
        tmplelcs = np.array(list(elcw.keys()))
        usedw = np.array(list(elcw.values()))
        ordr = np.argsort(-usedw)
        tmplelcs = list(tmplelcs[ordr])
        if return_weights:
            usedw = list(usedw[ordr])
            return tmplelcs, usedw
        else:
            return tmplelcs

    def electrodesforcluster(self, k, thresh=.1, return_weights=False, win=None):
        '''ELECTRODESFORCLUSTER - List of electrodes involved in given cluster
        elcs = ELECTRODESFORCLUSTER(k) returns a list of the electrodes 
        involved in the given cluster. An electrode is included if its
        "weight" is at least 0.1x the largest weight of any electrode. 
        The weight is just the rms of the signal in the template. 
        Optional argument THRESH overrides the threshold.'''
        K, W, E = self.tmpls.shape
        if win is None:
            win = range(W)
        if k>=K:
            tmplelcs = np.array([self.elc4clust[k]])
            usedw = np.array([1])
        else:
            chweights = np.sqrt(np.sum(self.tmpls[k,win,:]**2, 0))
            maxweight = np.max(chweights)
            usedchs = chweights > thresh * maxweight
            tmplchans = np.nonzero(usedchs)[0]
            usedw = chweights[usedchs]
            ordr = np.argsort(-usedw)
            tmplchans = tmplchans[ordr]
            usedw = usedw[ordr]
            tmplelcs = self.chan2elc[tmplchans].flatten().tolist()
        if return_weights:
            return tmplelcs, usedw
        else:
            return tmplelcs

    def _isclusterrelevant(self, elcs, k, thresh=.1):
        '''ISCLUSTERRELEVANT - Test whether electrodes participate in cluster
        ISCLUSTERRELEVANT(elcs, k) returns True if any of the electrodes 
        in the list ELCS is involved in cluster K. 
        Optional argument THRESH overrides the inclusion threshold as in
        ELECTRODESFORCLUSTER.'''
        tmplelcs = self.electrodesforcluster(k, thresh)
        for e in elcs:
            if e in tmplelcs:
                return True
        return False

    def relevantclusters(self, elcs, thresh=.1):
        '''RELEVANTCLUSTERS - All clusters that involve a set of electrodes
        RELEVANTCLUSTERS(elcs) returns a vector of booleans that reflect 
        for each cluster  whether any of the electrodes in ELCS is 
        involved in that cluster. 
        Optional argument THRESH is as for ELECTRODESFORCLUSTER.'''
        rel = np.zeros(self.nclusters(), bool)
        for k in range(self.nclusters()):
            rel[k] = self._isclusterrelevant(elcs, k, thresh)
        return rel

    def relevantspikes(self, elcs, thresh=.1, mask=None):
        '''RELEVANTSPIKES - List of spikes that involve given electrodes
        idx = RELEVANTSPIKES(elcs) returns the indices (into the vectors 
        returned by ALLSPIKES) of all the spikes that involve at least 
        one of the electrodes in the list ELCS according to 
        RELEVANTCLUSTERS. 
        Optional argument MASK may be a boolean vector that specifies 
        which clusters may potentially be included.'''
        rel = self.relevantclusters(elcs, thresh)
        if mask is not None:
            rel &= mask
        isrel = rel[self.spk_cls]
        return np.nonzero(isrel)[0]
