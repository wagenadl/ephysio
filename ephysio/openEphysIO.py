#!/usr/bin/python3

import numpy as np
import ast
import os
import glob
from . import timeMachine

def _nodetostr(node):
    if type(node) == str:
        return node
    else:
        return f'Record Node {node}'


def loadoebin(exptroot, expt=1, rec=1, node=None):
    """
    LOADOEBIN - Load the Open Ephys oebin file.  Oebin is a JSON file detailing
    channel information, channel metadata and event metadata descriptions.
    Contains a field for each of the recorded elements detailing their folder
    names, samplerate, channel count and other needed information.

    Parameters
    ----------
    exptroot : string
        Path to the folder of the general experiment.
    expt : integer, default is 1
        The subexperiment number.
    rec : integer, default is 1
        The recording number.
    node: None or a recording node number.

    Returns
    -------
    oebin : dictionary
        Organized information from oebin file into a dictionary for the selected
        subexperiment and recording.
    """

    fldr = f'{exptroot}'
    if node is not None:
        fldr += '/' + _nodetostr(node)
    fldr += f'/experiment{expt}/recording{rec}'
    oebin = ast.literal_eval(open(f'{fldr}/structure.oebin').read())
    return oebin


def findnode(exptroot, expt, rec, stream, node=None):
    if not os.path.exists(exptroot):
        raise ValueError(f'Experiment not found: {exptroot}')

    if node is not None:
        streampaths = glob.glob(f'{exptroot}/{node}/experiment{expt}/recording{rec}/continuous/{stream}')
        if len(streampaths) == 1:
            streampath = streampaths[0].replace('\\', '/')
            stream = streampath.split('/')[-1]
        else:
            raise ValueError(f'Stream {stream} not found')
        return node, stream

    if os.path.exists(f'{exptroot}/experiment{expt}'):
        streampaths = glob.glob(f'{exptroot}/experiment{expt}/recording{rec}/continuous/{stream}')
        if len(streampaths) == 1:
            streampath = streampaths[0].replace('\\', '/')
            stream = streampath.split('/')[-1]
        else:
            raise ValueError(f'Stream {stream} not found')
        return '', stream

    nodepaths = glob.glob(f'{exptroot}/Record Node *')
    for nodepath in nodepaths:
        nodepath = nodepath.replace('\\', '/')
        streampaths = glob.glob(f'{nodepath}/experiment{expt}/recording{rec}/continuous/{stream}')
        if len(streampaths) == 1:
            streampath = streampaths[0].replace('\\', '/')
            return nodepath.split('/')[-1], streampath.split('/')[-1]
    raise ValueError('No node found for given expt/rec/stream')


def streaminfo(exptroot, expt=1, rec=1, section='continuous', stream=0, ttl=None, node=None):
    """
    STREAMINFO - Return a dictionary containing the information session
    about the selected stream from the oebin file.

    Parameters
    ----------
    exptroot : string
        Path to the folder of the general experiment.
    expt : integer, default is 1
        The subexperiment number.
    rec : integer, default is 1
        The recording number.
    section : string, default is 'continuous'
        The name of the file
    stream : integer or string, default is 0. If numeric, node must be specified if using modern openephys
        The continuous data source we are getting the filename for.
    node : The recording node to use. Usually left as None, in which case STREAM is used to find the node

    Returns
    -------
        Return a dictionary containing the information session about the
        indicated stream from the oebin file.
    Notes
    -----
    Optional argument STREAM specifies which continuous data to load. Default is
        index 0. Alternatively, STREAM may be a string, specifying the
        subdirectory, e.g., "Neuropix-PXI-126.1/". This method is preferred,
        because it is more robust. Who knows whether those stream numbers are
        going to be preserved from one day to the next.
        (The final slash is optional.)
    """

    node, stream = findnode(exptroot, expt, rec, stream, node)
    oebin = loadoebin(exptroot, expt, rec, node)
    if type(stream) != str:
        return oebin[section][stream]
    stream = stream.split('/')[0]
    for n in range(len(oebin[section])):
        folder = oebin[section][n]['folder_name'].split('/')
        if folder[0].lower() == stream.lower():
            if section != 'events' or ttl is None or folder[1].lower() == ttl.lower():
                return oebin[section][n]
    raise ValueError(f'Could not find {section} stream "{stream}" ttl {ttl}')


def recordingpath(exptroot, expt, rec, stream, node):
    node, stream = findnode(exptroot, expt, rec, stream, node)
    fldr = f'{exptroot}/{node}/experiment{expt}/recording{rec}'
    return fldr

def _continuousmetadata(tsfn, continfo):
    tms = np.load(tsfn, mmap_mode='r')
    s0 = tms[0]
    chlist = continfo['channels']
    f_Hz = continfo['sample_rate']
    return s0, f_Hz, chlist


def _doloadcontinuous(contfn, tsfn, continfo):
    """
    _DOLOADCONTINUOUS - Load continuous data from a file.
    dat, s0, f_Hz, chinfo = DOLOADCONTINOUS(contfn, tsfn, continfo)
    performs the loading portion of the LOADCONTINUOUS function.
    """

    mm = np.memmap(contfn, dtype=np.int16, mode='r')  # in Windows os "mode=c" do not work
    C = continfo['num_channels']
    N = len(mm) // C
    dat = np.reshape(mm, [N, C])
    del mm
    s0, f_Hz, chlist = _continuousmetadata(tsfn, continfo)
    return (dat, s0, f_Hz, chlist)


def continuousmetadata(exptroot, expt=1, rec=1, stream=0, infix='continuous', contfn=None, node=None):
    '''Like LOADCONTINUOUS, except it doesn't load data.
    Return is (s0, f_Hz, chlist). '''
    ourcontfn, tsfn, continfo = contfilename(exptroot, expt, rec, stream, infix, node)
    return _continuousmetadata(tsfn, continfo)


def loadcontinuous(exptroot, expt=1, rec=1, stream=0, infix='continuous', contfn=None, node=None):
    """
    LOADCONTINUOUS - Load continuous data from selected data source in an
    Open Ephys experiment.

    Parameters
    ----------
    exptroot : string
        Path to the folder of the general experiment.
    expt : integer, default is 1
        The subexperiment number.
    rec : integer, default is 1
        The recording number.
    stream : integer or string, default is 0
        The continuous data source we are getting the filename for.
    contfn : null or string, default is None

    Returns
    -------
        Returns the outputs of _doloadcontinuous, i.e.
    dat : numpy.ndarray
        Data for the selected experiment and recording.
    s0 : numeric
        Sample number relative to the start of the experiment of the start of
        this recording.
    f_Hz : integer
        The sampling rate (in Hz) of the data set.
    chlist : list
        Channel information dicts, straight from the oebin file.

    Notes
    -----
    The returned value S0 is important because event timestamps are relative
        to the experiment rather than to a recording.
    Optional argument STREAM specifies which continuous data to load. Default is
        index 0. Alternatively, STREAM may be a string, specifying the
        subdirectory, e.g., "Neuropix-PXI-126.1/". This method is preferred,
        because it is more robust. Who knows whether those stream numbers are
        going to be preserved from one day to the next.
        (The final slash is optional.)
    Optional argument CONTFN overrides the name of the continuous.dat file,
        which is useful if you want to load the output of a preprocessed file
        (see, e.g., applyCAR).
    """

    ourcontfn, tsfn, continfo = contfilename(exptroot, expt, rec, stream, infix, node)
    if contfn is not None:
        ourcontfn = contfn
    return _doloadcontinuous(ourcontfn, tsfn, continfo)


def _lameschmitt(dat, thr1, thr0):
    high = dat >= thr1
    high[0] = False
    low = dat <= thr0
    iup = []
    idn = []
    idx = 0
    siz = len(dat)
    while idx < siz:
        didx = np.argmax(high[idx:])
        if didx == 0:
            break
        idx += didx
        iup.append(idx)
        didx = np.argmax(low[idx:])
        if didx == 0:
            break
        idx += didx
        idn.append(idx)
    if len(iup) > len(idn):
        iup = iup[:-1]
    return np.array(iup), np.array(idn)


def loadanalogevents(exptroot, expt=1, rec=1, stream=0, node=None, channel=1):
    dat, _, _, _ = loadcontinuous(exptroot, expt, rec, stream, node=node)
    dat = dat[:, channel]
    thr = (np.min(dat) + np.max(dat)) / 2
    iup, idn = _lameschmitt(dat, 1.1 * thr, .9 * thr)

    ss = np.stack((iup, idn), 1).flatten()
    cc = channel + np.zeros(ss.shape, dtype=int)
    st = np.stack((1 + 0 * iup, -1 + 0 * idn), 1).flatten()
    return ss, cc, st



def filterevents(ss_trl, bnc_cc, bnc_states, channel=1, updown=1):
    """
    FILTEREVENTS - Return only selected events from an event stream.

    Parameters
    ----------
    ss_trl : numpy.ndarray
        The samplestamps of events (in samples) relative to the recording.
    bnc_cc : numpy.ndarray
        The event channel numbers associated with each event.
    bnc_states : numpy.ndarray
        Contains +/-(ch) indicating whether the channel went up or down.
    channel : integer, default is 1
        The channel to use
    updown : integer, default is 1
        Set to -1 to extract the down events.
        Set to 0 to extract both the up and down events.

    Returns
    -------
    numpy.ndarray : The extracted timestamps for the up or down or both events for the
        selected channel.
        If updown is set to 0 also return a numpy.ndarray which is the extracted positive or negative answer.
    """

    if updown == 1:
        return ss_trl[np.logical_and(bnc_cc == channel, bnc_states > 0)]
    elif updown == -1:
        return ss_trl[np.logical_and(bnc_cc == channel, bnc_states < 0)]
    elif updown == 0:
        return ss_trl[bnc_cc == channel], np.sign(bnc_states[bnc_cc == channel])
    else:
        raise ValueError('Bad value for updown')



def _probablycntlbarcodes(sss, uds, f_Hz):
    # Guess whether sss, uds represent new or old style bar codes
    isnew = []
    for ss, ud in zip(sss, uds):
        if len(ss) > 4 and ud[0]:
            if len(ss) == 18 and ss[1] - ss[0] < 450 * f_Hz / 30e3:
                isnew.append(1)
            else:
                isnew.append(0)
    return np.mean(isnew) > .5


def _cntlbarcodes(sss, uds):
    # This decodes bar codes from arduino code "stimbarcoduino"
    codes = []
    times = []
    nbar = 0
    noth = 0
    for ss, ud in zip(sss, uds):
        if len(ss) == 18 and ud[0] == 1 and not np.any(np.diff(ud) == 0):
            # Potential barcode
            s0 = ss[0]
            dss = np.diff(ss)
            code = 0
            onems = dss[0] / 10
            thr = dss[0] * 3 // 4
            if np.any(dss < 3 * onems) or np.any(dss > 14 * onems):
                noth += 1
            else:
                nbar += 1
                for ds in dss[1:]:
                    code *= 2
                    if ds > thr:
                        code += 1
                codes.append(code)
                times.append(s0)
        elif len(ss) > 5:
            noth += 1
    print(f"(Found {nbar} legit bar codes and {noth} other groups)")

    return times, codes


def _openephysbarcodes(sss, uds, f_Hz):
    sss_on = []
    sss_off = []
    for ss, ud in zip(sss, uds):
        s0 = ss[0]
        drop = np.nonzero(np.diff(ud) == 0)[0]
        keep = np.ones(np.shape(ss), bool)
        keep[drop] = False
        ss = ss[keep]
        ud = ud[keep]
        ss_on = ss[ud > 0]
        ss_off = ss[ud < 0]
        while len(ss_off) > 0 and len(ss_on) > 0 and ss_off[0] < ss_on[0]:
            ss_off = ss_off[1:]
        while len(ss_on) > 0 and len(ss_off) > 0 and ss_on[-1] > ss_off[-1]:
            ss_on = ss_on[:-1]
        if len(ss_on) > 0 and len(ss_off) > 0:
            sss_on.append(ss_on)
            sss_off.append(ss_off)
        else:
            print(f'Bar code dropped at {s0}')

    codes = []
    times = []
    PREDURATION_MS = 20
    INTER_BARCODE_INTERVAL_S = 30
    BARCODE_BITS = 32
    BITDURATION_MS = (INTER_BARCODE_INTERVAL_S - 1) * 32 / BARCODE_BITS
    # See https://github.com/open-ephys/sync-barcodes/blob/main/arduino-barcodes/arduino-barcodes.ino line 37.
    PERIOD = BITDURATION_MS * f_Hz / 1000

    N = len(sss_on)
    if N > 10:
        # Hack for early version of Frank Lanfranchi's barcode generator
        ss1 = [ss[0] for ss in sss_on]
        ds1 = np.diff(ss1)
        if np.median(ds1) < f_Hz * 10.0:
            PERIOD = 120

    for n in range(N):  # Loop over all the codes
        dton = sss_off[n][1:] - sss_on[n][1:]  # Skip first pulse
        dtoff = sss_on[n][1:] - sss_off[n][:-1]
        dtoff[0] -= PREDURATION_MS * f_Hz / 1000  # First interval start marker
        dton = np.round(dton / PERIOD).astype(int)
        dtoff = np.round(dtoff / PERIOD).astype(int)
        value = 0
        K = len(dton)
        bit = 1
        for k in range(K):
            for q in range(dtoff[k]):
                bit *= 2
            for q in range(dton[k]):
                value += bit
                bit *= 2
        codes.append(value)
        times.append(sss_on[n][0])
    return times, codes


def getbarcodes(ss_trl, bnc_cc, bnc_states, f_Hz, channel=1, newstyle=None):
    '''
    GETBARCODES - Obtain barcodes from samplestamped rising and falling edges.

    Parameters
    ----------
    ss_trl : numpy.ndarray
        The sample stamps of an event (in samples) relative to the recording.
    bnc_cc : numpy.ndarray
        The event channel numbers associated with each event.
    bnc_states : numpy.ndarray
        Contains +/-(channel) indicating whether the channel went up or down.
    f_Hz : integer
        Frequency (in Hz) of recording sampling rate.
    channel : integer, default is 1
        The channel to use.
    newstyle: Bool: True for new style bar codes, False for old style, or None
        for use heuristic

    Returns
    -------
    times : time stamps of starts of bar codes, in same units as ss_trl
    codes : decoded bar codes (16-bit integers)
    '''

    ss, ud = filterevents(ss_trl, bnc_cc, bnc_states, channel=channel, updown=0)
    sss, uds = inferblocks(ss, t_split_s=.08, f_Hz=f_Hz, extra=ud)
    if newstyle is None:
        newstyle = _probablycntlbarcodes(sss, uds, f_Hz)

    if newstyle:
        print('New bar codes!')
        return _cntlbarcodes(sss, uds)
    else:
        print('Old bar codes!')
        return _openephysbarcodes(sss, uds, f_Hz)


def matchbarcodes(ss1, bb1, ss2, bb2):
    '''
    MATCHBARCODES -
    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    '''
    sss1 = []
    sss2 = []
    N1 = len(ss1)
    for n in range(N1):
        b = bb1[n]
        try:
            idx = bb2.index(b)
            sss1.append(ss1[n])
            sss2.append(ss2[idx])
        except:
            print(f'Caution: no match for barcode #{n}')
            pass  # Barcode not matched
    return (sss1, sss2)



def dropglitches(ss, ds0):
    '''DROPGLITCHES - Drop glitches from event streams
    ss = DROPGLITCHES(ss, ds0), where SS is an Nx2 array of on and off times of events,
    hunts for glitches (events or interevent times shorter than DS0) and removes them.
    The result is an N'x2 array, where N' is less than N by the number of removed glitches.'''
    ss = ss.flatten()
    glitch = np.nonzero(np.diff(ss) < ds0)[0]
    # 0 5   20 25    26 27   50 55    => 0 5    20 25    50 55
    #                 *  *

    # 0 5   20 25    36 37   50 55    => 0 5    20 25    50 55
    #                    *

    # 0 5   20 25    26 37   50 55    => 0 5    20 37    50 55
    #                 *
    while len(glitch):
        ss = np.delete(ss, [glitch[0], glitch[0] + 1])
        glitch = np.nonzero(np.diff(ss) < ds0)[0]
    return np.reshape(ss, [len(ss) // 2, 2])



def _populate(dct, *args):
    for a in args:
        if a not in dct:
            dct[a] = {}
        dct = dct[a]


def _quickglob(pattern):
    # Replace "//" by "/" except at beginning or after ":"
    bits = pattern.split(":")
    nb = []
    for bit in bits:
        nb.append(bit[:1] + bit[1:].replace("//", "/"))
    pattern = ":".join(nb)

    if pattern[-1] == '/':
        pattern = pattern[:-1]
    idx = None
    for k, bit in enumerate(pattern.split("/")):
        if "*" in bit:
            idx = k
    if idx is None:
        raise ValueError("Bad glob")
    paths = [path.replace("\\", "/") for path in glob.glob(pattern)]
    return [p.split("/")[idx] for p in paths]



def _deglitch(ss, thresh):
    ss = ss.flatten()
    while True:
        ds = np.diff(ss)
        drop = np.nonzero(ds<thresh)[0]
        if len(drop)>0:
            ss = np.delete(ss, [drop[0], drop[0]+1])
        else:
            break
    N = len(ss)
    return ss.reshape(N//2, 2)


class Loader:
    '''OpenEphys organizes a recording session into "experiments" which
    contain "recordings" which contain "streams" of continuous data
    with associated "events". In recent versions, the hierarchy on disk
    also keeps track of the recording "node" that saved each stream.
    This class allows easy access to all of this information.
    '''

    def __init__(self, root, cntlbarcodes=None):
        '''Loader(root) constructs a loader for OpenEphys data.

        Parameters
        ----------
        root : location of data in the file system
        cntlbarcodes: Whether to expect CNTL-style bar codes. (The alternative
            is OpenEphys-style bar codes.) Set to None to autodetect.

        Notes
        -----
        root must be specified with forward slashes, even on Windows.
        '''

        self.root = root
        self._expts = None
        self._recs = {}  # expt -> list of recs
        self._streams = None
        self._nodemap = None  # node -> list of streams
        self._streammap = None  # stream -> node
        self._sfreqs = {}  # node->expt->rec->stream->Hertz
        self._oebins = {}  # node->expt->rec
        self._events = {}  # node->expt->rec->stream->digitalchannel->Nx2
        self._ss0 = {}  # node->expt->rec->stream
        self._barcodes = {}  # node->expt->rec->stream->(times, ids)
        self.cntlbarcodes = cntlbarcodes
        if not os.path.exists(root):
            raise ValueError(f"No data at {root}")

    def _oebin(self, node, expt, rec):
        _populate(self._oebins, node, expt)
        if rec not in self._oebins[node][expt]:
            self._oebins[node][expt][rec] = loadoebin(self.root,
                                                      expt, rec, node)
        return self._oebins[node][expt][rec]

    def _oebinsection(self, expt, rec, stream, section='continuous', node=None):
        node = self._autonode(stream, node)
        oebin = self._oebin(node, expt, rec)[section]
        N = len(oebin)
        for n in range(N):
            if oebin[n]['folder_name'].lower() == stream.lower() + "/":
                return oebin[n]
        raise ValueError(f"No oebin for {node} {expt} {rec} {stream} {section}")

    def _autonode(self, stream, node=None):
        if node is None:
            return self.streammap()[stream][0]
        else:
            return node

    def streams(self):
        '''STREAMS - List of all streams
        STREAMS() returns a list of all stream names, spike, LFP, or
        otherwise.'''
        if self._streams is None:
            sss = []
            for ss in self.nodemap().values():
                sss += ss
            sss = list(set(sss))
            sss.sort()
            self._streams = sss
        return self._streams

    def spikestreams(self):
        '''SPIKESTREAMS - List of all spike streams
        SPIKESTREAMS() returns a list of all spike streams, i.e., those
        streams that are not NIDAQ streams or obviously LFP streams.'''
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
        '''SPIKESTREAM - Name of a spike stream
        SPIKESTREAM() returns the name of the first spike stream, if any. 
        SPIKESTREAM(n) returns the name of the n-th spike stream (counting 
        from 0).
        Raises exception if the given stream does not exist.'''
        ss = self.spikestreams()
        if len(ss)<=n:
            raise ValueError("Nonexistent spikestream")
        return ss[n]

    def lfpstreams(self):
        '''LFPSTREAMS - List of all LFP streams
        LFPSTREAMS() returns a list of all LFP streams, i.e., those
        streams that are not NIDAQ streams or obviously spike streams.'''
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
        '''LFPSTREAM - Name of an LFP stream
        LFPSTREAM() returns the name of the first LFP stream, if any. 
        LFPSTREAM(n) returns the name of the n-th LFP stream (counting 
        from 0).
        Raises exception if the given stream does not exist.'''
        ss = self.lfpstreams()
        if len(ss)<=n:
            raise ValueError("Nonexistent LFP stream")
        return ss[n]

    def nidaqstreams(self):
        '''NIDAQSTREAMS - List of all NIDAQ streams
        NIDAQSTREAMS() returns a list of all NIDAQ streams.'''
        return [s for s in self.streams() if s.startswith("NI-DAQ")]

    def nidaqstream(self, n=0):
        '''NIDAQSTREAM - Name of NIDAQ stream
        NIDAQSTREAM() returns the name of the first NIDAQ stream.
        NIDAQSTREAM(n) returns the name of the n-th LFP stream (counting
        from 0). 
        Raises exception if the given stream does not exist.'''
        nidaqs = self.nidaqstreams()
        if len(nidaqs) <= n:
            raise ValueError("Nonexistent NIDAQ stream")
        return nidaqs[n]

    def experiments(self):
        '''EXPERIMENTS - List of "experiments"
        EXPERIMENTS() returns a list of all experiments in the session.'''
        if self._expts is None:
            node = self.nodes()[0]
            fldr = self.root
            if node is not None:
                fldr += f"/{node}"
            pattern = f"{fldr}/experiment*"
            expts = _quickglob(pattern)
            self._expts = [int(expt[len("experiment"):]) for expt in expts]
        return self._expts

    def recordings(self, expt):
        '''RECORDINGS - List of "recordings"
        RECORDINGS(expt) returns a list of all recordings in the given
        "experiment" (which must be one of the items in the list returned
        by EXPERIMENTS()).'''
        if expt not in self._recs:
            node = self.nodes()[0]
            fldr = self.root
            if node is not None:
                fldr += f"/{node}"
            pattern = f"{fldr}/experiment{expt}/recording*"
            recs = _quickglob(pattern)
            self._recs[expt] = [int(rec[len("recording"):]) for rec in recs]
        return self._recs[expt]

    def nodes(self):
        '''NODES - List of recording nodes
        NODES() returns a simple list of recording nodes, or [None] for
        older versions of OpenEphys that did not keep track.'''
        return list(self.nodemap().keys())

    def _firstexpt(self, node):
        fldr = self.root
        if node is not None:
            fldr += f"/{node}"
        xpts = [x for x in os.listdir(fldr) if x.startswith("experiment")]
        if not xpts:
            raise Exception("No experiments")
        xpts.sort()
        return int(xpts[0][10:])

    def _firstrec(self, node, expt):
        fldr = self.root
        if node is not None:
            fldr += f"/{node}"
        fldr += f"/experiment{expt}"
        recs = [x for x in os.listdir(fldr) if x.startswith("recording")]
        if not recs:
            raise Exception("No recordings")
        recs.sort()
        return int(recs[0][9:])
    
    def _recfolder(self, node, expt=1, rec=1):
        fldr = self.root
        if node is not None:
            fldr += f"/{node}"
        if expt is None:
            expt = self._firstexpt(node)
        if rec is None:
            rec = self._firstrec(node, expt)
        fldr += f"/experiment{expt}"
        fldr += f"/recording{rec}"
        return fldr

    def contfolder(self, stream, expt=1, rec=1, node=None):
        '''CONTFOLDER - Folder name where continuous data is stored
        p = CONTFOLDER(stream) returns the full path of the "continuous" folder for the given stream.
        Optional expt, rec, and node further specify.'''
        node = self._autonode(stream, node)
        return self._recfolder(node, expt, rec) + f"/continuous/{stream}"

    def _contsamplestampfile(self, stream, expt=1, rec=1, node=None):
        fldr = self.contfolder(stream, expt, rec, node)
        for fn in ["sample_numbers", "timestamps"]:
            if os.path.exists(f"{fldr}/{fn}.npy"):
                return f"{fldr}/{fn}.npy"
        raise Exception(f"No sample stamp file found for {stream} {expt}:{rec}")

    def _eventfolder(self, stream, expt=1, rec=1, node=None):
        node = self._autonode(stream, node)
        fldr = self._recfolder(node, expt, rec) + f"/events/{stream}"
        ttl = _quickglob(f"{fldr}/TTL*")
        return f"{fldr}/{ttl[0]}"

    def _eventsamplestampfile(self, stream, expt=1, rec=1, node=None):
        fldr = self._eventfolder(stream, expt, rec, node)
        for fn in ["sample_numbers", "timestamps"]:
            if os.path.exists(f"{fldr}/{fn}.npy"):
                return f"{fldr}/{fn}.npy"
        raise Exception(f"No sample stamp file found for {stream} {expt}:{rec}")

    def _eventstatesfile(self, stream, expt=1, rec=1, node=None):
        fldr = self._eventfolder(stream, expt, rec, node)
        for fn in ["states", "channel_states"]:
            if os.path.exists(f"{fldr}/{fn}.npy"):
                return f"{fldr}/{fn}.npy"
        raise Exception(f"No sample stamp file found for {stream} {expt}:{rec}")

    def nodemap(self):
        '''NODEMAP - Map of stream names per node
        NODEMAP() returns a dict mapping node names to lists of the streams
        contained in each node.'''

        def explorenodes(node, timestamps_optional=False):
            pattern = self._recfolder(node, None, None) + "/continuous/*/timestamps.npy"
            streams = _quickglob(pattern)
            if timestamps_optional:
                # Tolerate lack of timestamps.npy files
                # This can be OK if you only want to read continuous data.
                # It does not work if you need to access events.
                pattern = self._recfolder(node, None, None) + "/continuous/*/continuous.dat"
                streams += _quickglob(pattern)
                streams = list(set(streams))
                streams.sort()
            return streams

        if self._nodemap is None:
            nodes = _quickglob(f"{self.root}/*")
            nodemap = {}
            if any([node.startswith("experiment") for node in nodes]):
                # Old style without explicit nodes
                nodemap[None] = explorenodes(None)
            else:
                for node in nodes:
                    if os.path.exists(f"{self.root}/{node}/settings.xml"):
                        probes = explorenodes(node)
                        if len(probes):
                            nodemap[node] = probes
            streammap = {}
            for node, streams in nodemap.items():
                for stream in streams:
                    if stream not in streammap:
                        streammap[stream] = []
                    streammap[stream].append(node)
            self._nodemap = nodemap
            self._streammap = streammap
        return self._nodemap

    def streammap(self):
        '''STREAMMAP - Map of stream names to recording nodes
        STREAMMAP() returns a dict mapping stream names to lists of
        recording names that contain that stream.'''
        if self._streammap is None:
            self.nodemap()
        return self._streammap

    def samplingrate(self, stream, expt=None, rec=None, node=None):
        '''SAMPLINGRATE - Sampling rate of a stream
        SIMPLINGRATE(stream), where STREAM is one of the items returned
        by STREAMS() or its friends, returns the sampling rate of that
        stream in Hertz. Optional experiments EXPT and REC specify the
        "experiment" and "recording", but those can usually be left out,
        as the sampling rate is generally consistent for a whole session.'''
        node = self._autonode(stream, node)
        if expt is None:
            expt = self._firstexpt(node)
        if rec is None:
            rec = self._firstrec(node, expt)
        _populate(self._sfreqs, node, expt, rec)
        if stream not in self._sfreqs[node][expt][rec]:
            info = self._oebinsection(expt, rec, stream=stream, node=node)
            self._sfreqs[node][expt][rec][stream] = info['sample_rate']
        return self._sfreqs[node][expt][rec][stream]

    def data(self, stream, expt=1, rec=1, node=None, stage='continuous'):
        '''DATA - Data for a stream
        DATA(stream) returns the data for the first recording from the
        given stream as a TxC array. Optional arguments EXPT, REC, and
        NODE further specify.
        By default, the file "continuous.dat" is loaded. Use optional
        argument STAGE to specify an alternative. (E.g., stage='salpa'
        for "salpa.dat".)'''
        contfn = self.contfolder(stream, expt, rec, node)
        contfn += f"/{stage}.dat"
        mm = np.memmap(contfn, dtype=np.int16, mode='r')
        info = self._oebinsection(expt, rec, stream=stream, node=node)
        C = info['num_channels']
        T = len(mm) // C
        return np.reshape(mm, [T, C])

    def bitvolts(self, stream, expt=1, rec=1, node=None):
        '''BITVOLTS - Scale factor for all the channels for a data stream
        BITVOLTS(stream) returns the scale factors to convert DATA from
        binary scale to volts as a C-length array. Optional arguments EXPT, REC, and
        NODE further specify.'''
        info = self._oebinsection(expt, rec, stream=stream, node=node)
        return [ch['bit_volts'] for ch in info['channels']]

    def channellist(self, stream, expt=1, rec=1, node=None):
        '''CHANNELLIST - List of channels for a stream
        CHANNELLIST(stream) returns the list of channels for that stream.
        Each entry in the list is a dict with channel name and other
        information straight from the OEBIN file.
        Optional arguments EXPT, REC, and NODE further specify.'''
        info = self._oebinsection(expt, rec, stream=stream, node=node)
        return info['channels']

    def events(self, stream, expt=1, rec=1, node=None):
        '''EVENTS - Events for a stream
        EVENTS(stream) returns the events associated with the given
        stream as a dict of event channel numbers (typically counted 1
        to 8) to Nx2 arrays of on and off sample times, measured from
        the beginning of the recording, so that they can be used
        directly as indices into the corresponding DATA().
        Ab initio, only conventional digital events are returned, but
        after you call ANALOGEVENTS() on the stream, those are included
        in the dict returned by EVENTS as well.'''
        node = self._autonode(stream, node)
        
        _populate(self._events, node, expt, rec)
        _populate(self._ss0, node, expt, rec)
        if stream not in self._events[node][expt][rec]:
            tms = np.load(self._contsamplestampfile(stream, expt, rec, node))
            self._ss0[node][expt][rec][stream] = tms[0]
            ss_abs = np.load(self._eventsamplestampfile(stream, expt, rec, node))
            fldr = self._eventfolder(stream, expt, rec, node)
            ss = ss_abs - self._ss0[node][expt][rec][stream]
            delta = np.load(self._eventstatesfile(stream, expt, rec, node))
            cc = np.abs(delta)
            self._events[node][expt][rec][stream] = {}
            channels = np.unique(cc)
            for c in channels:
                if c<=0:
                    print("Warning: negative channel in event extraction", c)
                    continue
                idx = np.nonzero(cc == c)[0]
                N = len(idx)
                myss = ss[idx]
                mydelta = delta[idx]
                if len(myss) and mydelta[0] < 0:
                    myss = myss[1:]
                    mydelta = mydelta[1:]
                if len(myss) and mydelta[-1] > 0:
                    myss = myss[:-1]
                    mydelta = mydelta[:-1]
                if np.any(np.diff(mydelta) == 0):
                    raise Exception("Event edges should alternate")
                N = len(myss)
                myss = myss.reshape(N // 2, 2)
                self._events[node][expt][rec][stream][c] = myss
        return self._events[node][expt][rec][stream]

    def analogevents(self, stream, channel="A0", expt=1, rec=1, node=None):
        '''ANALOGEVENTS - Return virtual events from analog channel
        ss = ANALOGEVENTS(stream, channel) treats the given channel (specified
        in string form, e.g., "A0", or "A1", etc.) as if it were a digital
        channel and returns an Nx2 array of on/off event time stamps.
        '''
        node = self._autonode(stream, node)
        self.events(stream, expt, rec, node) # just to populate the dict
        if channel not in self._events[node][expt][rec][stream]:
            ss, cc, st = loadanalogevents(self.root, expt, rec, stream, node, int(channel[1:]))
            N = len(ss)
            self._events[node][expt][rec][stream][channel] = np.reshape(ss, (N // 2, 2))
        return self._events[node][expt][rec][stream]

    def barcodes(self, stream, expt=1, rec=1, node=None, channel=1):
        '''BARCODES - Extract bar codes from a given stream
        times, codes = BARCODES(stream) returns the time stamps and codes of the
        bar codes associated with the given stream. Optional arguments EXPT, REC,
        NODE further specify.
        Optional argument CHANNEL specifies the digital channel from which the
        bar codes are to be read. If CHANNEL is a string like "A0", the bar codes
        are read from the given analog channel.'''
        node = self._autonode(stream, node)
        _populate(self._barcodes, node, expt, rec)
        if stream not in self._barcodes[node][expt][rec]:
            if type(channel) == str:
                self.analogevents(stream, channel, expt, rec, node)
            evts = self.events(stream, expt, rec, node)[channel]
            fs = self.samplingrate(stream, expt, rec, node)
            ss_on = evts[:, 0]
            ss_off = evts[:, 1]
            sss_on, sss_off = inferblocks(ss_on, fs, t_split_s=1, extra=ss_off)
            sss = [np.stack((son, sof), 1).flatten()
                   for son, sof in zip(sss_on, sss_off)]
            uds = [np.stack((np.ones(son.shape),
                             -np.ones(son.shape)), 1).flatten()
                   for son in sss_on]

            if self.cntlbarcodes is None:
                self.cntlbarcodes = _probablycntlbarcodes(sss, uds, fs)
            if self.cntlbarcodes:
                tt, vv = _cntlbarcodes(sss, uds)
            else:
                tt, vv = _openephysbarcodes(sss, uds, fs)
            if len(tt) < 5:
                raise Exception("Not enough bar codes - Did you specify the right kind in the Loader constructor?")
            self._barcodes[node][expt][rec][stream] = (tt, vv)
        return self._barcodes[node][expt][rec][stream]

    def shifttime(self, times, sourcestream, deststream, expt=1, rec=1,
                  sourcenode=None, destnode=None,
                  sourcebarcode=1, destbarcode=1):
        '''SHIFTTIME - Translate event time stamps to other stream
        SHIFTTIME(times, source, dest) translates event time stamps
        defined relative to the SOURCE stream for use as indices in
        the DEST stream. Operates on the first experiment/recording
        unless EXPT and REC are specified.
        This relies on the existence of "bar codes" in one of the
        event channels of both streams.
        SOURCEBARCODE and DESTBARCODE specify bar code channels.
        Analog channels are supported; see BARCODES.

        Perhaps this function should be called TRANSLATEEVENTTIME.'''

        ss1, bb1 = self.barcodes(sourcestream, expt, rec,
                                 sourcenode, sourcebarcode)
        ss2, bb2 = self.barcodes(deststream, expt, rec,
                                 destnode, destbarcode)
        ss1_matched, ss2_matched = matchbarcodes(ss1, bb1, ss2, bb2)
        if len(ss1_matched) < 2 + .2 * (len(ss1) + len(ss2)) / 2:
            raise Exception("Not enough matched bar codes")

        # Interpolate
        result = np.interp(times, ss1_matched, ss2_matched)

        # Extrapolate times before the first barcode and after the last
        isbefore = np.nonzero(times < ss1[0])[0]
        if len(isbefore):
            print(f"Caution: Extrapolating {len(isbefore)} event(s) before start of bar codes")
            a_before, b_before = np.polyfit(ss1[:2], ss2[:2], 1)
            result[isbefore] = a_before * times[isbefore] + b_before

        isafter = np.nonzero(times > ss1[-1])[0]
        if len(isafter):
            print(f"Caution: Extrapolating {len(isafter)} event(s) after end of bar codes")
            a_after, b_after = np.polyfit(ss1[-2:], ss2[-2:], 1)
            result[isafter] = a_after * times[isafter] + b_after

        return result

    def translatedata(self, data, t0, sourcestream, deststream,
                      expt=1, rec=1,
                      sourcenode=None, destnode=None,
                      sourcebarcode=1, destbarcode=1):
        '''TRANSLATEDATA - Translate a chunk of data from one timezone to another.
        datad, t0d = TRANSLATEDATA(data, t0, source, dest) takes a chunk of data (vector of
        arbitrary length N) that lives in the timezone of the SOURCE stream with time stamps
        T0 up to T1 = T0+N, and reinterpolates it to the timezone of the DEST stream. The
        function figures out the translations of T0 and T1 in the DEST stream. We call those T0D
        and T1D. The function returns a vector of length M = T1D - T0D as well as T0D. Note that M
        and T1D are not returned but can be directly inferred from T0D and the length of DATAD.
        Note that Txx are sample time stamps, i.e., expressed in samples, not seconds.
        See SHIFTTIME for other parameters and conditions.
        '''
        N = len(data)
        t1 = t0 + N
        # Figure out edges of interval in destination time zone
        t01d = self.shifttime(np.array([t0, t1]), sourcestream, deststream, expt, rec,
                              sourcenode, destnode, sourcebarcode, destbarcode)
        t0d = t01d[0]
        t1d = t01d[1]
        ttd = np.arange(t0d, t1d)
        # Figure out timepoints in source time zone corresponding to interval in dest.
        # Note backward translation
        tts = self.shifttime(ttd, deststream, sourcestream, expt, rec,
                             destnode, sourcenode, destbarcode, sourcebarcode)
        # Interpolate the data
        datad = np.interp(tts, np.arange(t0, t1), data)
        return datad, t0d

    def nidaqevents(self, stream, expt=1, rec=1, node=None,
                    nidaqstream=None, nidaqbarcode=1, destbarcode=1,
                    glitch_ms=None):
        '''NIDAQEVENTS - NIDAQ events translated to given stream
        NIDAQEVENTS is a convenience function that first calls EVENTS
        on the NIDAQ stream, then SHIFTTIME to convert those events
        to the time base of the given STREAM.
        NIDAQBARCODE and DESTBARCODE are the digital (or analog)
        channel that contain bar codes.
        Optional argument GLITCH_MS specifies that glitches shorter than
        the given duration should be removed.'''
        if nidaqstream is None:
            nidaqstream = self.nidaqstream()
        events = self.events(nidaqstream, expt, rec)
        fs = self.samplingrate(nidaqstream, expt, rec)
        if stream == nidaqstream:
            return events
        nevents = {}
        for ch, evts in events.items():
            if ch != nidaqbarcode:
                if glitch_ms is not None:
                    evts = dropglitches(evts, glitch_ms * fs / 1000)
                nevents[ch] = self.shifttime(evts, nidaqstream, stream,
                                             expt, rec,
                                             sourcebarcode=nidaqbarcode,
                                             destbarcode=destbarcode).astype(int)
        return nevents

    def inferblocks(self, ss, stream, split_s=5.0, dropshort_ms=None, minblocklen=None):
        '''INFERBLOCKS - Infer blocks in lists of sample time stamps
        sss = INFERBLOCKS(ss, stream) splits events into inferred blocks based
        on lengthy pauses.

        Parameters
        ----------
        ss : numpy.ndarray
            The samplestamps of events (in samples) relative to the recording.
        stream : the stream from which those events originate (to retrieve sampling rate)
        t_split_s : threshold for splitting events, default is 5.0 seconds
        dropshort_ms: events that happen less than given time after previous are dropped
        minblocklen: if given, blocks with fewer events than this are dropped

        Returns
        -------
        ss_block : list
            List of numpy arrays samplestamps, one per block.
        '''

        fs = self.samplingrate(stream)
        if dropshort_ms:
            ss = _deglitch(ss, dropshort_ms * fs/1e3)
        ssb, sse = inferblocks(ss[:, 0], fs, split_s, ss[:, 1])
        blks = [np.stack((sb, se), 1) for sb, se in zip(ssb, sse)]
        if minblocklen is not None:
            blks = [blk for blk in blks if len(blk)>=minblocklen]
        return blks
