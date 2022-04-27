#!/usr/bin/python3

import numpy as np
import ast
import os
import glob


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


def streaminfo(exptroot, expt=1, rec=1, section='continuous', stream=0, ttl='TTL_1', node=None):
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
    if not stream.endswith('/'):
        stream += '/'
    if section=='events':
        stream += ttl + '/'
    for n in range(len(oebin[section])):
        if oebin[section][n]['folder_name'].lower() == stream.lower():
            return oebin[section][n]
    raise ValueError(f'Could not find {section} stream "{stream}"')


def recordingpath(exptroot, expt, rec, stream, node):
    node, stream = findnode(exptroot, expt, rec, stream, node)
    fldr = f'{exptroot}/{node}/experiment{expt}/recording{rec}'
    return fldr


def contfilename(exptroot, expt=1, rec=1, stream=0, infix='continuous', node=None):
    """
    CONTFILENAME - Return the filename of the continuous data for a given recording.
    
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

    Returns
    -------
    A tuple (ifn, tsfn, info) comprising:
    ifn : string
        Full filename of the continuous.dat file.
    tsfn : numpy.ndarray
        Timestamps for the slected recording in the selected subexperiment.
    info : string
        The corresponding "continfo" section of the oebin file.

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
    fldr = recordingpath(exptroot, expt, rec, stream, node)
    continfo = streaminfo(exptroot, expt, rec, 'continuous', stream, node=node)
    subfldr = continfo['folder_name']
    if subfldr.endswith('/'):
        subfldr = subfldr[:-1]
    ifn = f'{fldr}/continuous/{subfldr}/{infix}.dat'
    tsfn = f'{fldr}/continuous/{subfldr}/timestamps.npy'
    return ifn, tsfn, continfo

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
    high = dat>=thr1
    high[0] = False
    low = dat<=thr0
    iup = []
    idn = []
    idx = 0
    siz = len(dat)
    while idx<siz:
        didx = np.argmax(high[idx:])
        if didx==0:
            break
        idx += didx
        iup.append(idx)
        didx = np.argmax(low[idx:])
        if didx==0:
            break
        idx += didx
        idn.append(idx)
    if len(iup)>len(idn):
        iup = iup[:-1]
    return np.array(iup), np.array(idn)


def loadanalogevents(exptroot, expt=1, rec=1, stream=0, node=None, channel=1):
    dat, _, _, _ = loadcontinuous(exptroot, expt, rec, stream, node=node)
    dat = dat[:,channel]
    thr = (np.min(dat) + np.max(dat)) / 2
    iup, idn = _lameschmitt(dat, 1.1*thr, .9*thr)
    
    ss = np.stack((iup, idn), 1).flatten()
    cc = channel + np.zeros(ss.shape, dtype=int)
    st = np.stack((1 + 0*iup, -1 + 0*idn), 1).flatten()
    return ss, cc, st


def loadevents(exptroot: str, s0: int=0, expt: int=1, rec: int=1, stream: int=0, ttl: str='TTL_1', node: int=None):
    """
    LOADEVENTS - Load events associated with a continuous data stream.

    Parameters
    ----------
    exptroot : string
        Path to the folder of the general experiment.
    s0 :  integer, default is 0
        The first timestamp when hit play on Open Ephys gui. It  must
        be obtained from LOADCONTINUOUS.
    expt : integer, default is 1
        The subexperiment number.
    rec : integer, default is 1
        The recording number.
    stream : integer or string, default is 0
        The continuous data source we are getting the events for, either as an integer index or
        as a direct folder name.
    ttl : string, default is 'TTL_1'
        The TTL event stream that we are loading

    Returns
    -------
    ss_trl - s0 : numpy.ndarray
        The timestamps (samplestamps) of events (in samples) relative to the recording.
    bnc_cc : numpy.ndarray
        The event channel numbers associated with each event.
    bnc_states : numpy.ndarray
        Contains +/-1 indicating whether the channel went up or down.
    fw : numpy.ndarray
        The full_words 8-bit event states for the collection of events.

    Notes
    -----
    Optional argument STREAM specifies which continuous data to load. Default is
        index 0. Alternatively, STREAM may be a string, specifying the
        subdirectory, e.g., "Neuropix-PXI-126.1/". This method is preferred,
        because it is more robust. Who knows whether those stream numbers are
        going to be preserved from one day to the next.
        (The final slash is optional.)
    Note that SS_TRL can be used directly to index continuous data: Even though
        timestamps are stored on disk relative to start of experiment, this
        function subtracts the timestamp of the start of the recording to make life
        a little easier.
    """

    node, stream = findnode(exptroot, expt, rec, stream, node)
    if s0 is None:
        s0, f_Hz, chlist = continuousmetadata(exptroot, expt, rec, stream, node)
    fldr = f'{exptroot}/{node}/experiment{expt}/recording{rec}'
    evtinfo = streaminfo(exptroot, expt, rec, 'events', stream, ttl, node)
    subfldr = evtinfo['folder_name']
    ss_trl = np.load(f'{fldr}/events/{subfldr}/timestamps.npy')
    bnc_cc = np.load(f'{fldr}/events/{subfldr}/channels.npy')
    bnc_states = np.load(f'{fldr}/events/{subfldr}/channel_states.npy')
    fw = np.load(f'{fldr}/events/{subfldr}/full_words.npy')
    return (ss_trl - s0, bnc_cc, bnc_states, fw)


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
        Contains +/-1 indicating whether the channel went up or down.
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


def inferblocks(ss_trl, f_Hz, t_split_s=5.0, extra=None, dropshort_ms=None):
    """
    INFERBLOCKS - Split events into inferred stimulus blocks based on
    lengthy pauses.

    Parameters
    ----------
    ss_trl : numpy.ndarray
        The samplestamps of events (in samples) relative to the recording.
        (obtained from LOADEVENTS or FILTEREVENTS)
    f_Hz : integer
        Frequency (in Hz) of recording sampling rate.
    t_split_s : numeric, default is 5.0
    dropshort_ms: events that happen less than given time after previous are dropped

    Returns
    -------
    ss_block : list
        List of numpy arrays samplestamps, one per block.
    Notes
    -----
    ss_block = INFERBLOCKS(ss_trl, f_Hz) splits the event time stamps SS_TRL (from LOADEVENTS
    or FILTEREVENTS) into blocks with cuts when adjacent events are more than 5 seconds
    apart. Optional argument T_SPLIT_S overrides that threshold.
    """

    if dropshort_ms is not None:
        dt = np.diff(ss_trl)
        drop = np.nonzero(dt < dropshort_ms*f_Hz/1000)[0] + 1
        ss_trl = np.delete(ss_trl, drop)
    idx = np.nonzero(np.diff(ss_trl) >= t_split_s * f_Hz)[0] + 1
    N = len(ss_trl)
    idx = np.hstack((0, idx, N))
    ss_block = []
    for k in range(len(idx) - 1):
        ss_block.append(ss_trl[idx[k]:idx[k + 1]])
    if extra is None:
        return ss_block
    ex_block = []
    for k in range(len(idx) - 1):
        ex_block.append(extra[idx[k]:idx[k + 1]])
    return ss_block, ex_block


def extractblock(dat, ss_trl, f_Hz, margin_s=10.0):
    '''
    EXTRACTBLOCK - Extract ephys data for a block of vis_stimuli identified by SS_TRL
    which must be one of the items in the list returned by INFERBLOCKS.

    Parameters
    ----------
    dat : numpy.ndarray
        Ephys data from where we want to extract from.
    ss_trl : numpy.ndarray
        The samplestamps of event (in samples) relative to the recording which
        should be one of the items in the list returned by INFERBLOCKS.
    f_Hz : integer
        Frequency (in Hz) of recording sampling rate.
    margin_s : numeric, default is 10.0
        Length of the margin (in seconds) included at the beginning and end of
        the block (unless of course the block starts less than 10 s from the
        beginning of the file or analogously at the end).

    Returns
    -------
    dat[s0:s1,:] : numpy.ndarray
        Extracted portion of ephys data.
    ss_trl - s0 : numpy.ndarray
        Shifted timestamps of events (relative to the extracted portion of data).
    '''

    s0 = ss_trl[0] - int(margin_s * f_Hz)
    s1 = ss_trl[-1] + int(margin_s * f_Hz)
    S, C = dat.shape
    if s0 < 0:
        s0 = 0
    if s1 > S:
        s1 = S
    return dat[s0:s1, :], ss_trl - s0


def _probablycntlbarcodes(sss, uds, f_Hz):
    # Guess whether sss, uds represent new or old style bar codes
    isnew = []
    for ss, ud in zip(sss, uds):
        if len(ss) > 4 and ud[0]:
            if len(ss)==18 and ss[1]-ss[0] < 450*f_Hz/30e3:
                isnew.append(1)
            else:
                isnew.append(0)
    return np.mean(isnew) > .5


def _cntlbarcodes(sss, uds):
    # This decodes bar codes from arduino code "stimbarcoduino"
    codes = []
    times = []

    for ss, ud in zip(sss, uds):
        if len(ss)==18 and ud[0]==1 and not np.any(np.diff(ud)==0):
            # Potential barcode
            s0 = ss[0]
            dss = np.diff(ss)
            code = 0
            onems = dss[0]/10
            thr = dss[0]*3//4
            if np.any(dss < 3*onems) or np.any(dss>12*onems):
                print(f'Likely faulty bar code of length {len(ss)} at {ss[0]}')
            else:
                for ds in dss[1:]:
                    code *= 2
                    if ds > thr:
                        code += 1
                codes.append(code)
                times.append(s0)
        elif len(ss)>5:
            print(f'Likely faulty bar code of length {len(ss)} at {ss[0]}')

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
    PERIOD = 120 * f_Hz / 30000
    N = len(sss_on)
    for n in range(N):
        dton = sss_off[n] - sss_on[n]
        dtoff = sss_on[n][1:] - sss_off[n][:-1]
        dton = np.round(dton / PERIOD).astype(int)
        dtoff = np.round(dtoff / PERIOD).astype(int)
        value = 0
        K = len(dton)
        bit = 1
        for k in range(K):
            for q in range(dton[k]):
                value += bit
                bit *= 2
            if k < K - 1:
                for q in range(dtoff[k]):
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

    ss_barc : decoded bar codes (16-bit integers)
    '''
    
    ss, ud = filterevents(ss_trl, bnc_cc, bnc_states, channel=channel, updown=0)
    sss, uds = inferblocks(ss, t_split_s=1, f_Hz=f_Hz, extra=ud)
    if newstyle is None:
        newstyle = _probablycntlbarcodes(sss, uds, f_Hz)

    if newstyle:
        print('New bar codes!')
        return _cntlbarcodes(sss, uds)
    else:
        print('Old bar codes!')
        return _openephysbarcodes(sss, uds, f_Hz)


def matchbarcodes(ss1, bb1, ss2, bb2):
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


def loadtranslatedevents(exptroot, expt=1, rec=1,
                         sourcestream='NI-DAQmx-142.0',
                         targetstream='Neuropix-PXI-126.0',
                         targetttl='TTL_1',
                         sourcenode=None,
                         targetnode=None,
                         newstylebarcodes=None,
                         sourcebarcodechannel=1):
    '''
    LOADTRANSLATEDEVENTS - As LOADEVENTS, but ss_trl is translated to 
    samples in the target stream.
    
    Parameters
    ----------
    exptroot, expt, rec : as for loadevents
    sourcestream: the stream from which the events will be loaded
    targetstream: the stream into which the timestamps will be translated
    newstylebarcodes: set to True to expect CNTL-style bar codes, False to
        expect OpenEphys-style bar codes, or None to auto-detect.
    sourcebarcodechannel: the digital input that the barcode generate is
        connected on the source. Set to "A0" to "A7" to extract bar codes
        from analog channels instead.

    In addition, the following parameters have defaults that are usually sufficient:
    sourcenode, targetnode: node identifiers for source and target streams
    targetttl: the TTL group number for the target stream

    '''
    
    _, fs_Hz_src, _ = continuousmetadata(exptroot, expt, rec, stream=sourcestream)

    if type(sourcebarcodechannel)==str and sourcebarcodechannel.startswith("A"):
        sourcebarcodechannel=int(sourcebarcodechannel[1:])
        ss_trl, bnc_cc, bnc_states = loadanalogevents(exptroot, expt=expt, rec=rec, stream=sourcestream,
                                                      node=sourcenode, channel=sourcebarcodechannel)
        fw = bnc_states
    else:
        ss_trl, bnc_cc, bnc_states, fw = loadevents(exptroot, s0=0, expt=expt, rec=rec, stream=sourcestream,
                                                    node=sourcenode)
    t_ni, bc_ni = getbarcodes(ss_trl, bnc_cc, bnc_states, fs_Hz_src, newstyle=newstylebarcodes,
                              channel=sourcebarcodechannel)

    _, fs_Hz_tgt, _ = continuousmetadata(exptroot, expt, rec, stream=targetstream)
    (ss1, cc1, vv1, fw1) = loadevents(exptroot, s0=None, expt=expt, rec=rec, stream=targetstream, ttl=targetttl,
                                      node=targetnode)
    t_np, bc_np = getbarcodes(ss1, cc1, vv1, fs_Hz_tgt, newstyle=newstylebarcodes)

    ss_ni, ss_np = matchbarcodes(t_ni, bc_ni, t_np, bc_np)
    ss_trl = np.interp(ss_trl, ss_ni, ss_np)
    idx = np.nonzero(bnc_cc != 1)
    ss_trl = ss_trl[idx]
    bnc_cc = bnc_cc[idx]
    bnc_states = bnc_states[idx]
    fw = fw[idx]
    return ss_trl.astype(int), bnc_cc, bnc_states, fw


class EventTranslator:
    '''EventTranslator transforms timestamps from one stream (e.g., a probe)
    to another stream (either another probe or the NIDAQ.'''
    def __init__(self, exptroot, expt=1, rec=1,
                         sourcestream='NI-DAQmx-142.0',
                         sourcettl='TTL_1',
                         targetstream='Neuropix-PXI-126.0',
                         targetttl='TTL_1',
                         sourcenode=None,
                         targetnode=None,
                         sourcebarcodechannel=1):

        (s0, f_Hz, chlist) = continuousmetadata(exptroot, expt, rec, stream=sourcestream, node=sourcenode)
        if type(sourcebarcodechannel)==str and sourcebarcodechannel.startswith("A"):
            sourcebarcodechannel=int(sourcebarcodechannel[1:])
            ss_trl, bnc_cc, bnc_states = loadanalogevents(exptroot, expt=expt, rec=rec, stream=sourcestream,
                                                          node=sourcenode, channel=sourcebarcodechannel)
            fw = bnc_states
        else:
            ss_trl, bnc_cc, bnc_states, fw = loadevents(exptroot, s0=s0, expt=expt, rec=rec, stream=sourcestream,
                                                          ttl=sourcettl, node=sourcenode)
        t_ni, bc_ni = getbarcodes(ss_trl, bnc_cc, bnc_states, f_Hz, channel=sourcebarcodechannel)

        (s0, f_Hz, chlist) = continuousmetadata(exptroot, expt, rec, stream=targetstream, node=targetnode)
        (ss1, cc1, vv1, fw1) = loadevents(exptroot, s0=s0, expt=expt, rec=rec, stream=targetstream, ttl=targetttl,
                                          node=targetnode)
        t_np, bc_np = getbarcodes(ss1, cc1, vv1, f_Hz)

        self.ss_ni, self.ss_np = matchbarcodes(t_ni, bc_ni, t_np, bc_np)

    def translate(self, ss):
        '''Translates timestamps from the SOURCESTREAM to the TARGETSTREAM'''
        return np.interp(ss, self.ss_ni, self.ss_np)


#%%
def read_broken_array(fp, allow_pickle=False, pickle_kwargs=None):
    """
    Read an array from an NPY file without reshaping.
    This is copied from numpy.lib.format to deal with files that got truncated because of disk-full errors.
    Be very, very careful. Always check that the results make sense before trusting them. You may be loading
    garbage.
    FP must be an opened .npy file.

    Parameters
    ----------
    fp : file_like object
        If this is not a real file object, then this may take extra memory
        and time.
    allow_pickle : bool, optional
        Whether to allow writing pickled data. Default: False

        .. versionchanged:: 1.16.3
            Made default False in response to CVE-2019-6446.

    pickle_kwargs : dict
        Additional keyword arguments to pass to pickle.load. These are only
        useful when loading object arrays saved on Python 2 when using
        Python 3.

    Returns
    -------
    array : ndarray
        The array from the data on disk.
    shape: tuple
        The shape that array should have had
    isfortran: bool
        Is the data stored in Fortran rather than C order (?)

    Raises
    ------
    ValueError
        If the data is invalid, or allow_pickle=False and the file contains
        an object array.

    """
    
    from numpy.lib.format import read_magic, _check_version, _read_array_header, isfileobj
    import numpy

    version = read_magic(fp)
    _check_version(version)
    shape, fortran_order, dtype = _read_array_header(fp, version)
    if len(shape) == 0:
        count = 1
    else:
        count = numpy.multiply.reduce(shape, dtype=numpy.int64)

    # Now read the actual data.
    if dtype.hasobject:
        # The array contained Python objects. We need to unpickle the data.
        if not allow_pickle:
            raise ValueError("Object arrays cannot be loaded when "
                             "allow_pickle=False")
        if pickle_kwargs is None:
            pickle_kwargs = {}
        try:
            array = pickle.load(fp, **pickle_kwargs)
        except UnicodeError as err:
            # Friendlier error message
            raise UnicodeError("Unpickling a python object failed: %r\n"
                               "You may need to pass the encoding= option "
                               "to numpy.load" % (err,))
    else:
        if isfileobj(fp):
            # We can use the fast fromfile() function.
            array = numpy.fromfile(fp, dtype=dtype, count=count)
        else:
            # This is not a real file. We have to read it the
            # memory-intensive way.
            # crc32 module fails on reads greater than 2 ** 32 bytes,
            # breaking large reads from gzip streams. Chunk reads to
            # BUFFER_SIZE bytes to avoid issue and reduce memory overhead
            # of the read. In non-chunked case count < max_read_count, so
            # only one read is performed.

            # Use np.ndarray instead of np.empty since the latter does
            # not correctly instantiate zero-width string dtypes; see
            # https://github.com/numpy/numpy/pull/6430
            array = numpy.ndarray(count, dtype=dtype)

            if dtype.itemsize > 0:
                # If dtype.itemsize == 0 then there's nothing more to read
                max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, dtype.itemsize)

                for i in range(0, count, max_read_count):
                    read_count = min(max_read_count, count - i)
                    read_size = int(read_count * dtype.itemsize)
                    data = _read_bytes(fp, read_size, "array data")
                    array[i:i+read_count] = numpy.frombuffer(data, dtype=dtype,
                                                             count=read_count)

    return array, shape, fortran_order


def _populate(dct, *args):
    for a in args:
        if a not in dct:
            dct[a] = {}
        dct = dct[a]


def _quickglob(pattern):
    idx = None
    for k, bit in enumerate(pattern.split("/")):
        if "*" in bit:
            idx = k
    if idx is None:
        raise ValueError("Bad glob")
    paths = [path.replace("\\", "/") for path in glob.glob(pattern)]
    return [p.split("/")[idx] for p in paths]


class Loader: 
    '''OpenEphys organizes a recording session into "experiments" which
    contain "recordings" which contain "streams" of continuous data
    with associated "events". In recent versions, the hierarchy on disk
    also keeps track of the recording "node" that saved each stream. 
    This class allows easy access to all of this information.
    '''

    def __init__(self, root, cntlbarcodes=True):
        '''Loader(root) constructs a loader for OpenEphys data.

        Parameters
        ----------
        root : location of data in the file system
        cntlbarcodes: Whether to expect CNTL-style bar codes. (The alternative
            is OpenEphys-style bar codes.)
        '''

        self.root = root
        self._expts = None
        self._recs = {} # expt -> list of recs
        self._streams = None
        self._nodemap = None # node -> list of streams
        self._streammap = None # stream -> node
        self._sfreqs = {} # node->expt->rec->stream->Hertz
        self._oebins = {} # node->expt->rec
        self._events = {} # node->expt->rec->stream->digitalchannel->Nx2
        self._ss0 = {} # node->expt->rec->stream
        self._barcodes = {} # node->expt->rec->stream->(times, ids)
        self.cntlbarcodes = cntlbarcodes

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
            if oebin[n]['folder_name'].lower()==stream.lower() + "/":
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
            self._streams = []
            for ss in self.nodemap().values():
                self._streams += ss
            self._streams = list(set(self._streams))
            self._streams.sort()
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

    def nidaqstreams(self):
        '''NIDAQSTREAMS - List of all NIDAQ streams
        NIDAQSTREAMS() returns a list of all NIDAQ streams.'''
        return [s for s in self.streams() if s.startswith("NI-DAQ")]
    
    def nidaqstream(self):
        '''NIDAQSTREAM - Name of only NIDAQ stream
        NIDAQSTREAM() returns the name of the NIDAQ stream if there is
        precisely one. Otherwise, raises an exception.'''
        nidaqs = self.nidaqstreams()
        if len(nidaqs)==1:
            return nidaqs[0]
        elif len(nidaqs)==0:
            raise ValueError("No NIDAQ streams")
        else:
            raise ValueError("Multiple NIDAQ streams")
    
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

    def _recfolder(self, node, expt=1, rec=1):
        fldr = self.root
        if node is not None:
            fldr += f"/{node}"
        fldr += f"/experiment{expt}/recording{rec}"
        return fldr

    def _contfolder(self, stream, expt=1, rec=1, node=None):
        node = self._autonode(stream, node)
        return self._recfolder(node, expt, rec) + f"/continuous/{stream}"

    def _eventfolder(self, stream, expt=1, rec=1, node=None):
        node = self._autonode(stream, node)
        fldr = self._recfolder(node, expt, rec) + f"/events/{stream}"
        ttl = _quickglob(f"{fldr}/TTL_*")
        return f"{fldr}/{ttl[0]}"

    def nodemap(self):
        '''NODEMAP - Map of stream names per node
        NODEMAP() returns a dict mapping node names to lists of the streams
        contained in each node.'''
        def explorenodes(node):
            pattern = self._recfolder(node) + "/continuous/*/timestamps.npy"
            streams = _quickglob(pattern)
            return streams
        
        if self._nodemap is None:
            nodes = _quickglob(f"{self.root}/*")
            self._nodemap = {}
            if any([node.startswith("experiment") for node in nodes]):
                # Old style without explicit nodes
                self._nodemap[None] = explorenodes(None)
            else:
                for node in nodes:
                    probes = explorenodes(node)
                    if len(probes):
                        self._nodemap[node] = probes
            self._streammap = {}
            for node, streams in self._nodemap.items():
                for stream in streams:
                    if stream not in self._streammap:
                        self._streammap[stream] = []
                    self._streammap[stream].append(node)
                    
        return self._nodemap

    def streammap(self):
        '''STREAMMAP - Map of stream names to recording nodes
        STREAMMAP() returns a dict mapping stream names to lists of 
        recording names that contain that stream.'''
        if self._streammap is None:
            self.nodemap()
        return self._streammap
    
    def samplingrate(self, stream, expt=1, rec=1, node=None):
        '''SAMPLINGRATE - Sampling rate of a stream
        SIMPLINGRATE(stream), where STREAM is one of the items returned
        by STREAMS() or its friends, returns the sampling rate of that
        stream in Hertz. Optional experiments EXPT and REC specify the
        "experiment" and "recording", but those can usually be left out,
        as the sampling rate is generally consistent for a whole session.'''
        node = self._autonode(stream, node)
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
        contfn = self._contfolder(stream, expt, rec, node)
        contfn += f"/{stage}.dat"
        mm = np.memmap(contfn, dtype=np.int16, mode='r')
        info = self._oebinsection(expt, rec, stream=stream, node=node)
        C = info['num_channels']
        T = len(mm) // C
        return np.reshape(mm, [T, C])

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
        directly as indices into the corresponding DATA().'''
        node = self._autonode(stream, node)
        _populate(self._events, node, expt, rec)
        _populate(self._ss0, node, expt, rec)
        if stream not in self._events[node][expt][rec]:
            tms = np.load(self._contfolder(stream, expt, rec, node)
                          + "/timestamps.npy")
            self._ss0[node][expt][rec][stream] = tms[0]
            fldr = self._eventfolder(stream, expt, rec, node)
            ss_abs = np.load(f"{fldr}/timestamps.npy")
            ss = ss_abs - self._ss0[node][expt][rec][stream]
            cc = np.load(f"{fldr}/channels.npy")
            delta = np.load(f"{fldr}/channel_states.npy")
            self._events[node][expt][rec][stream] = {}
            channels = np.unique(cc)
            for c in channels:
                idx = np.nonzero(cc==c)[0]
                N = len(idx)
                myss = ss[idx]
                mydelta = delta[idx]
                if mydelta[0]<0:
                    raise Exception("First event edge should be up")
                if mydelta[-1]>0:
                    raise Exception("Last event edge should be down")
                if np.any(np.diff(mydelta)==0):
                    raise Exception("Event edges should alternate")
                myss = myss.reshape(N//2, 2) 
                self._events[node][expt][rec][stream][c] = myss
        return self._events[node][expt][rec][stream]

    def _ensureanalogevents(self, stream, expt=1, rec=1, node=None, channel="A0"):
        self.events(stream, expt, rec, node) # just to populate it
        if channel in self._events[node][expt][rec][stream]:
            return
        ss, cc, st = loadanalogevents(self.root, expt, rec, stream, node, int(channel[1:]))
        N = len(ss)
        self._events[node][expt][rec][stream][channel] = np.reshape(ss, (N//2, 2))
    
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
            if type(channel)==str:
                self._ensureanalogevents(stream, expt, rec, node, channel)
            evts = self.events(stream, expt, rec, node)[channel]
            fs = self.samplingrate(stream, expt, rec, node)
            ss_on = evts[:,0]
            ss_off = evts[:,1]
            sss_on, sss_off = inferblocks(ss_on, fs, t_split_s=1, extra=ss_off)
            sss = [np.stack((son,sof),1).flatten()
                   for son,sof in zip(sss_on, sss_off)]
            uds = [np.stack((np.ones(son.shape),
                             -np.ones(son.shape)),1).flatten()
                   for son in sss_on]
            if self.cntlbarcodes:
                tt, vv = _cntlbarcodes(sss, uds)
            else:
                tt, vv = _openephysbarcodes(sss, uds, fs)
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
        Analog channels are supported.'''
        ss1, bb1 = self.barcodes(sourcestream, expt, rec,
                                 sourcenode, sourcebarcode)
        ss2, bb2 = self.barcodes(deststream, expt, rec,
                                 destnode, destbarcode)
        ss1, ss2 = matchbarcodes(ss1, bb1, ss2, bb2)
        return np.interp(times, ss1, ss2)
        
    def nidaqevents(self, stream, expt=1, rec=1, node=None,
                    nidaqstream=None, nidaqbarcode=1, destbarcode=1):
        '''NIDAQEVENTS - NIDAQ events translated to given stream
        NIDAQEVENTS is a convenience function that first calls EVENTS
        on the NIDAQ stream, then SHIFTTIME to convert those events
        to the time base of the given STREAM.
        NIDAQBARCODE and DESTBARCODE are the digital (or analog)
        channel that contain bar codes.'''
        if nidaqstream is None:
            nidaqstream = self.nidaqstream()
        events = self.events(nidaqstream, expt, rec)
        if stream==nidaqstream:
            return events
        nevents = {}
        for ch, evts in events.items():
            if ch != nidaqbarcode:
                nevents[ch] = self.shifttime(evts, nidaqstream, stream,
                                             expt, rec,
                                             sourcebarcode=nidaqbarcode,
                                             destbarcode=destbarcode).astype(int)
        return nevents

    def inferblocks(self, ss, stream, split_s=5.0, dropshort_ms=None):
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

        Returns
        -------
        ss_block : list
            List of numpy arrays samplestamps, one per block.
        '''

        fs = self.samplingrate(stream)
        ssb, sse = inferblocks(ss[:,0], fs, split_s, ss[:,1], dropshort_ms)
        return [np.stack((sb, se), 1) for sb, se in zip(ssb, sse)]
