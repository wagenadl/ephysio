#!/usr/bin/python3

import numpy as np
import os
import glob

from . import timemachine


def _populate(dct, *args):
    for a in args:
        if a not in dct:
            dct[a] = {}
        dct = dct[a]


def _nodetostr(node):
    if type(node) == str:
        return node
    else:
        return f'Record Node {node}'


def _parselist(s):
    # Takes a string of the form "(xx)(yyy)(zzz)"... and returns a list ["xx", "yyy", "zzz", ...]
    s = ")" + s + "("
    bits = s.split(")(")
    return bits[1:-1]


def _parsescalar(s):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s=="true":
        return True
    if s=="false":
        return False
    return s


def _loadmeta(metafn):
    """
    LOADOEBIN - Load the SpikeGLX meta file.  Meta file is a text file detailing
    channel information, channel metadata and event metadata descriptions.
    Contains a field for each of the recorded elements detailing their folder
    names, samplerate, channel count and other needed information.
    Parameters
    ----------
    Returns
    -------
    oebin : dictionary
        Organized information from meta file into a dictionary for the selected
        subexperiment and recording.
    """

    if not os.path.exists(metafn):
        raise ValueError(f"No meta file at {metafn}")
    with open(metafn) as f:
        mdatList = f.readlines()
        # convert the list entries into key value pairs
        meta = {}
        for m in mdatList:
            key, value = m.strip().split('=', 1)
            if key.startswith("ni"):
                key = key[2:]
            elif key.startswith("im"):
                key = key[2:]
            if key.startswith("~"):
                meta[key[1:]] = _parselist(value)
            else:
                meta[key] = _parsescalar(value)
    return meta


def _compressdigi(digi):
    tt = []
    vv = []
    i0 = 0
    L = 32768
    v0 = 0
    while i0 < len(digi):
        print(f"compressdigi {int(i0*100/len(digi))}%    \r")
        dig = digi[i0:i0+L]
        dd = np.diff(np.concatenate(([v0], dig), 0))
        idx = np.nonzero(dd)[0]
        tt.append(i0 + idx)
        vv.append(dig[idx])
        v0 = dig[-1]
        i0 += L
    tt = np.concatenate(tt)
    vv = np.concatenate(vv)
    return tt,vv


def unpackbits(data, nbits=16):
    res = []

    for i in range(nbits):
        res.append(np.bitwise_and(data, 1<<i))
    return res


def _builddictfromttvv(tt, vv):
    bwd = unpackbits(vv, 8)
    evts = {}
    for k in range(8):
        idx = np.nonzero(np.diff(np.concatenate([[0], bwd[k], [0]])))[0]
        ttrans = tt[idx]
        N = len(ttrans)
        if N:
            evts[k + 1] = ttrans.reshape(N//2, 2)
    return evts


class Loader:
    '''
    This class allows easy access to all of this information.
    '''

    def __init__(self, root, cntlbarcodes=None):
        '''Loader(root) constructs a loader for spikeGLX data.
        Parameters
        ----------
        root : location of data in the file system
        cntlbarcodes: Whether to expect CNTL-style bar codes. (The alternative
            is OpenEphys-style bar codes.) Set to None to autodetect.
        '''
        self.root = root
        self.dtng = root.split("/")[-1]
        self._streams = None
        self._nodemap = None    # node -> list of streams
        self._streammap = None  # stream -> node
        self._sfreqs = {}   # node->expt->rec->stream->Hertz
        self._metas = {}    # stream->expt->rec->metainfo
        self._events = {}   # node->expt->rec->stream->digitalchannel->Nx2
        self._ss0 = {}  # node->expt->rec->stream
        if not os.path.exists(root):
            raise ValueError(f"No data at {root}")

    def _metafilename(self, stream, expt, rec, node):
        '''
        METAFILENAME - Return the full pathname of a .meta file
        fn = METAFILENAME(exptroot, expt, rec, stream) ...
        '''
        if node is None:
            raise ValueError(f"Node cannot be None for stream {stream} {expt} {rec}")
        simplefn = f"{self.root}/{self.dtng}_t{rec}.{node}.meta"
        if os.path.exists(simplefn):
            return simplefn
        deepfn = f"{self.root}/{self.dtng}_{node}/{self.dtng}_t{rec}.{stream}.meta"
        if os.path.exists(deepfn):
            return deepfn
        raise ValueError(f"No meta file for {node} {expt} {rec} {stream}")

    def _meta(self, stream, expt, rec, node):
        node = self._autonode(stream, node)
        _populate(self._metas, stream, expt)
        if rec in self._metas[stream][expt]:
            return self._metas[stream][expt][rec]
        metafn = self._metafilename(stream, expt, rec, node)
        meta = _loadmeta(metafn)
        self._metas[stream][expt][rec] = meta
        return meta

    def _autonode(self, stream, node=None):
        if node is None:
            return self.streammap()[stream][0]
        else:
            return node

    def nodes(self):
        '''
        NODES - List of recording nodes
        NODES() returns a simple list of recording nodes, or [None] for
        older versions of OpenEphys that did not keep track.
        '''
        return list(self.nodemap().keys())

    def nodemap(self):
        '''
        NODEMAP - Map of stream names per node
        NODEMAP() returns a dict mapping node names to lists of the streams
        contained in each node.
        '''
        if self._nodemap is not None:
            return self._nodemap
        root = self.root
        dtng = self.dtng
        metas = glob.glob(f"{root}/{dtng}_*.meta") + glob.glob(f"{root}/{dtng}_*/{dtng}_*.meta")
        print(metas)
        metas = [m.replace("\\", "/") for m in metas]
        metas = [m.split("/")[-1] for m in metas]
        metas = [m[len(dtng):] for m in metas]
        self._nodemap = {}
        self._streammap = {}
        for m in metas:
            bits = m.split(".")
            node = bits[1]
            stream = ".".join(bits[1:-1])
            if node not in self._nodemap:
                self._nodemap[node] = []
            self._nodemap[node].append(stream)
            self._streammap[stream] = [node]
        return self._nodemap

    def streams(self):
        '''
        STREAMS - List of all streams
        STREAMS() returns a list of all stream names, spike, LFP, or
        otherwise.
        '''
        if self._streams is None:
            sss = []
            for ss in self.nodemap().values():
                sss += ss
            sss = list(set(sss))
            sss.sort()
            self._streams = sss
        return self._streams

    def streammap(self):
        '''
        STREAMMAP - Map of stream names to recording nodes
        STREAMMAP() returns a dict mapping stream names to lists of
        recording names that contain that stream.
        '''
        if self._streammap is None:
            self.nodemap()
        return self._streammap

    def nidaqstreams(self):
        '''
        NIDAQSTREAMS - List of all NIDAQ streams
        NIDAQSTREAMS() returns a list of all NIDAQ streams.
        '''
        return [s for s in self.streams() if s.startswith("nidq")]

    def nidaqstream(self, n=0):
        '''
        NIDAQSTREAM - Name of NIDAQ stream
        NIDAQSTREAM() returns the name of the first NIDAQ stream.
        NIDAQSTREAM(n) returns the name of the n-th LFP stream (counting
        from 0).
        Raises exception if the given stream does not exist.
        '''
        nidaqs = self.nidaqstreams()
        if len(nidaqs) <= n:
            raise ValueError("Nonexistent NIDAQ stream")
        return nidaqs[n]

    def _firstexpt(self, node):
        fldr = self.root
        bits = fldr.split("g")
        return int(bits[-1])

    def _firstrec(self, node, expt):
        fldr = self.root
        fns = glob.glob(f"{fldr}/*.{node}.meta")
        if len(fns)==0:
            fns = glob.glob(f"{fldr}/*_{node}/*.{node}.*.meta")
        fns.sort()
        for fn in fns:
            bits = fn.split("_t")
            rec = int(bits[-1].split(".")[0])
            return rec

    def samplingrate(self, stream, expt=None, rec=None, node=None):
        '''
        SAMPLINGRATE - Sampling rate of a stream
        SIMPLINGRATE(stream), where STREAM is one of the items returned
        by STREAMS() or its friends, returns the sampling rate of that
        stream in Hertz. Optional experiments EXPT and REC specify the
        "experiment" and "recording", but those can usually be left out,
        as the sampling rate is generally consistent for a whole session.
        '''
        node = self._autonode(stream, node)
        if expt is None:
            expt = self._firstexpt(node)
        if rec is None:
            rec = self._firstrec(node, expt)
        _populate(self._sfreqs, node, expt, rec)
        if stream not in self._sfreqs[node][expt][rec]:
            info = self._meta(stream, expt, rec, node)
            self._sfreqs[node][expt][rec][stream] = info['SampRate']
        return self._sfreqs[node][expt][rec][stream]

    def spikestreams(self):
        '''
        SPIKESTREAMS - List of all spike streams
        SPIKESTREAMS() returns a list of all spike streams, i.e., those
        streams that are not NIDAQ streams or obviously LFP streams.
        '''
        nidaqs = set(self.nidaqstreams())
        ss = []
        for node, streams in self.nodemap().items():
            fsmax = 0
            fsbystream = {}
            for s in streams:
                if s not in nidaqs:
                    fs = self.samplingrate(s, node=node)
                    fsbystream[s] = fs
                    fsmax = max(fsmax, fs)
            for s in streams:
                if s not in nidaqs:
                    if fsbystream[s] == fsmax:
                        ss.append(s)
        return ss

    def spikestream(self, n=0):
        '''
        SPIKESTREAM - Name of a spike stream
        SPIKESTREAM() returns the name of the first spike stream, if any.
        SPIKESTREAM(n) returns the name of the n-th spike stream (counting
        from 0).
        Raises exception if the given stream does not exist.
        '''
        ss = self.spikestreams()
        if len(ss) <= n:
            raise ValueError("Nonexistent spikestream")
        return ss[n]

    def lfpstreams(self):
        '''
        LFPSTREAMS - List of all LFP streams
        LFPSTREAMS() returns a list of all LFP streams, i.e., those
        streams that are not NIDAQ streams or obviously spike streams.
        '''
        nidaqs = set(self.nidaqstreams())
        ss = []
        for node, streams in self.nodemap().items():
            fsmin = 10_000_000
            fsbystream = {}
            for s in streams:
                if s not in nidaqs:
                    fs = self.samplingrate(s, node=node)
                    fsbystream[s] = fs
                    fsmin = min(fsmin, fs)
            for s in streams:
                if s not in nidaqs:
                    if fsbystream[s] == fsmin:
                        ss.append(s)
        return ss

    def lfpstream(self, n=0):
        '''
        LFPSTREAM - Name of an LFP stream
        LFPSTREAM() returns the name of the first LFP stream, if any.
        LFPSTREAM(n) returns the name of the n-th LFP stream (counting
        from 0).
        Raises exception if the given stream does not exist.
        '''
        ss = self.lfpstreams()
        if len(ss) <= n:
            raise ValueError("Nonexistent LFP stream")
        return ss[n]

    def contfolder(self, stream, expt=0, rec=0, node=None):
        '''
        CONTFOLDER - Folder name where continuous data is stored
        p = CONTFOLDER(stream) returns the full path of the "continuous" folder for the given stream.
        Optional expt, rec, and node further specify.
        '''
        node = self._autonode(stream, node)
        if stream==node:
            return self.root
        else:
            return f"{self.root}/{self.dtng}_{node}"

    def _alldata(self, stream, expt=0, rec=0, node=None, stage='continuous'):
        '''
        DATA - Data for a stream
        DATA(stream) returns the data for the first recording from the
        given stream as a TxC array. Optional arguments EXPT, REC, and
        NODE further specify.
        By default, the file "continuous.dat" is loaded. Use optional
        argument STAGE to specify an alternative. (E.g., stage='salpa'
        for "salpa.dat".)
        '''
        contfn = self.contfolder(stream, expt, rec, node)
        contfn += f"/{self.dtng}_t{rec}.{stream}"
        if stage != "continuous":
            contfn += "." + stage
        contfn += ".bin"
        mm = np.memmap(contfn, dtype=np.int16, mode='r')
        info = self._meta(stream, expt, rec, node)
        C = info['nSavedChans']
        T = len(mm) // C
        return np.reshape(mm, [T, C])

    def channelcounts(self, stream, expt=0, rec=0, node=None):
        '''
        CHANNELCOUNTS - Channel counts for a stream
        nana, ndig = CHANNELNUMBERS(stream) returns a list of channel numbers
        for the given stream. Optional arguments EXPT, REC, and NODE
        further specify.
        '''
        node = self._autonode(stream, node)
        info = self._meta(stream, expt, rec, node)
        if info['typeThis']=='imec':
            ap, lf, sy = [int(x) for x in info['snsApLfSy'].split(",")]
            return ap+lf, sy
        elif info['typeThis']=='nidq':
            mn,mx,xa,dw = [int(x) for x in info['snsMnMaXaDw'].split(",")]
            return xa,dw
        else:
            raise Exception(f"Unknown stream type: {info['typeThis']}")

    def data(self, stream, expt=0, rec=0, node=None, stage='continuous'):
        dat = self._alldata(stream, expt, rec, node, stage)
        C = self.channelcounts(stream, expt, rec, node)[0]
        return dat[:, :C]

    def digidata(self, stream, expt=0, rec=0, node=None, stage='continuous'):
        dat = self._alldata(stream, expt, rec, node, stage)
        C = self.channelcounts(stream, expt, rec, node)[0]
        return dat[:, C:]

    def events(self, stream, expt=0, rec=0, node=None, reconstruct=False):
        fldr = self.contfolder(stream, expt, rec, node)
        fn = f"{fldr}/{self.dtng}_t{rec}.{stream}.events.npy"
        if os.path.exists(fn) and not reconstruct:
            ttvv = np.load(fn)
            tt = ttvv[:,0]
            vv = ttvv[:,1]
        else:
            digi = self.digidata(stream, expt, rec, node)
            print(digi.shape)
            tt, vv = _compressdigi(digi[:,0])
            print(np.max(tt))
            ttvv = np.stack([tt, vv], 1)
            np.save(fn, ttvv)
        evts = _builddictfromttvv(tt, vv)
        return evts


############################################################################

    def barcodes(self, stream, expt=0, rec=0, node=None, channel=1):
        """
        BARCODES - Extract barcodes from a given SpikeGLX stream.
        Parameters are similar to OpenEphysIO with adjustments for SpikeGLX data structure.
        Returns a timemachine. BarCodes object.
        """
        node = self._autonode(stream, node)
        _populate(self._barcodes, node, expt, rec)
        if stream not in self._barcodes[node][expt][rec]:
            # Extract event data from the specified channel
            evts = self.events(stream, expt, rec, node, reconstruct=True)
            if channel in evts:
                event_data = evts[channel]
            else:
                raise ValueError(f"Channel {channel} not found in stream {stream}")
            # Flatten event data and determine barcode type
            ss = event_data.flatten()
            fs = self.samplingrate(stream, expt, rec, node)
            if self.cntlbarcodes is None:
                self.cntlbarcodes = timemachine.CNTLBarCodes.probablyCNTL(ss, fs)
            if self.cntlbarcodes:
                self._barcodes[node][expt][rec][stream] = timemachine.CNTLBarCodes(ss, fs)
            else:
                self._barcodes[node][expt][rec][stream] = timemachine.OpenEphysBarCodes(ss, fs)
        return self._barcodes[node][expt][rec][stream]



    # def shifttime(self, times, sourcestream, deststream, expt=1, rec=1,
    #               sourcenode=None, destnode=None,
    #               sourcebarcode=1, destbarcode=1):
    #     '''SHIFTTIME - Translate event time stamps to other stream
    #     SHIFTTIME(times, source, dest) translates event time stamps
    #     defined relative to the SOURCE stream for use as indices in
    #     the DEST stream. Operates on the first experiment/recording
    #     unless EXPT and REC are specified.
    #     This relies on the existence of "bar codes" in one of the
    #     event channels of both streams.
    #     SOURCEBARCODE and DESTBARCODE specify bar code channels.
    #     Analog channels are supported; see BARCODES.
    #
    #     Perhaps this function should be called TRANSLATEEVENTTIME.'''
    #
    #     bc_source = self.barcodes(sourcestream, expt, rec,
    #                              sourcenode, sourcebarcode)
    #     bc_dest = self.barcodes(deststream, expt, rec,
    #                              destnode, destbarcode)
    #     tm = timemachine.TimeMachine(bc_dest, bc_source)
    #     return tm.translatetimes(times)
    #
    # def translatedata(self, data, t0, sourcestream, deststream,
    #                   expt=1, rec=1,
    #                   sourcenode=None, destnode=None,
    #                   sourcebarcode=1, destbarcode=1):
    #     '''TRANSLATEDATA - Translate a chunk of data from one timezone to another.
    #     datad, t0d = TRANSLATEDATA(data, t0, source, dest) takes a chunk of data (vector of
    #     arbitrary length N) that lives in the timezone of the SOURCE stream with time stamps
    #     T0 up to T1 = T0+N, and reinterpolates it to the timezone of the DEST stream. The
    #     function figures out the translations of T0 and T1 in the DEST stream. We call those T0D
    #     and T1D. The function returns a vector of length M = T1D - T0D as well as T0D. Note that M
    #     and T1D are not returned but can be directly inferred from T0D and the length of DATAD.
    #     Note that Txx are sample time stamps, i.e., expressed in samples, not seconds.
    #     See SHIFTTIME for other parameters and conditions.
    #     '''
    #     bc_source = self.barcodes(sourcestream, expt, rec,
    #                               sourcenode, sourcebarcode)
    #     bc_dest = self.barcodes(deststream, expt, rec,
    #                             destnode, destbarcode)
    #     tm = timemachine.TimeMachine(bc_dest, bc_source)
    #     return tm.translatedata(data, t0)
    #
    # def nidaqevents(self, stream, expt=1, rec=1, node=None,
    #                 nidaqstream=None, nidaqbarcode=1, destbarcode=1,
    #                 glitch_ms=None):
    #     '''NIDAQEVENTS - NIDAQ events translated to given stream
    #     NIDAQEVENTS is a convenience function that first calls EVENTS
    #     on the NIDAQ stream, then SHIFTTIME to convert those events
    #     to the time base of the given STREAM.
    #     NIDAQBARCODE and DESTBARCODE are the digital (or analog)
    #     channel that contain bar codes.
    #     Optional argument GLITCH_MS specifies that glitches shorter than
    #     the given duration should be removed.'''
    #     if nidaqstream is None:
    #         nidaqstream = self.nidaqstream()
    #     events = self.events(nidaqstream, expt, rec)
    #     fs = self.samplingrate(nidaqstream, expt, rec)
    #     if stream == nidaqstream:
    #         return events
    #     nevents = {}
    #     for ch, evts in events.items():
    #         if ch != nidaqbarcode:
    #             if glitch_ms is not None:
    #                 evts = dropglitches(evts, glitch_ms * fs / 1000)
    #             nevents[ch] = self.shifttime(evts, nidaqstream, stream,
    #                                          expt, rec,
    #                                          sourcebarcode=nidaqbarcode,
    #                                          destbarcode=destbarcode).astype(int)
    #     return nevents
    #
    # def inferblocks(self, ss, stream, split_s=5.0, dropshort_ms=None, minblocklen=None):
    #     '''INFERBLOCKS - Infer blocks in lists of sample time stamps
    #     sss = INFERBLOCKS(ss, stream) splits events into inferred blocks based
    #     on lengthy pauses.
    #
    #     Parameters
    #     ----------
    #     ss : numpy.ndarray
    #         The samplestamps of events (in samples) relative to the recording.
    #     stream : the stream from which those events originate (to retrieve sampling rate)
    #     t_split_s : threshold for splitting events, default is 5.0 seconds
    #     dropshort_ms: events that happen less than given time after previous are dropped
    #     minblocklen: if given, blocks with fewer events than this are dropped
    #
    #     Returns
    #     -------
    #     ss_block : list
    #         List of numpy arrays samplestamps, one per block.
    #     '''
    #
    #     fs = self.samplingrate(stream)
    #     return timemachine.inferblocks(ss, fs, split_s=split_s, dropshort_ms=dropshort_ms, minblocklen=minblocklen)







