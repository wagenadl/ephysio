import numpy as np


def inferblocks(ss, f_Hz, t_split_s=1.0):
    """
    INFERBLOCKS - Split events into inferred stimulus blocks based on
    lengthy pauses.

    Parameters
    ----------
    ss : numpy.ndarray
        The samplestamps of events (in samples)
    f_Hz : integer
        Frequency (in Hz) of recording sampling rate.
    t_split_s : numeric, default is 1.0

    Returns
    -------
    ss_block : list
        List of numpy arrays samplestamps, one per block.
    Notes
    -----
    ss_block = INFERBLOCKS(ss, f_Hz) splits the event time stamps SS into
    blocks with cuts when adjacent events are more than 1 second apart.
    Optional argument T_SPLIT_S overrides that threshold.
    """

    ds = np.diff(ss)
    thresh = int(t_split_s * f_Hz)
    ds[np.isnan(ss[1:])] = thresh + 1
    ds[np.isnan(ss[:-1])] = thresh + 1
    idx = np.nonzero(ds >= thresh)[0] + 1
    N = len(ss)
    idx = np.hstack((0, idx, N))
    ss_block = []
    for k in range(len(idx) - 1):
        ss_block.append(ss[idx[k]:idx[k + 1]])
    return ss_block


class BarCodes:
    def __init__(self, ss, fs_Hz):
        self.ss = ss # time stamps of up and down transitions as a single numpy array
        self.fs_Hz = fs_Hz
        self.trivial = True # true if the barcodes are trivial, i.e., there are no codes, just time intervals
        self.codes = {} # Map of timestamp to code ID, in order of time

    def match(self, other):
        """
        Return ss_self, ss_other: Match the barcodes from two different streams.
        """
        if self.trivial != other.trivial:
            raise ValueError("Mismatched triviality")

        if self.trivial:
            return self._matchtrivial(other)
        else:
            return self._matchcodes(other)

    def _matchtrivial(self, other):
        raise NotImplementedError("Trivial matching not implemented yet")

    def _matchcodes(self, other):
        ss_self = []
        ss_other = []
        revmap = { c: s for s, c in other.codes.items() }
        for s, c in self.codes.items():
            if c in revmap:
                ss_self.append(s)
                ss_other.append(revmap[c])
            else:
                print(f'Caution: no match for code {c}')
        return ss_self, ss_other


class KofikoBarCodes(BarCodes):
    def __init__(self, ss, fs_Hz):
        super().__init__(ss, fs_Hz)
        self.trivial = True
        self.codes = { s: 0 for s in ss[::2] }


class CNTLBarCodes(BarCodes):
    def __init__(self, ss, fs_Hz):
        super().__init__(ss, fs_Hz)
        sss = inferblocks(ss, fs_Hz)
        self.trivial = False
        self.codes = {}
        nbar = 0
        noth = 0
        for ss in sss:
            if len(ss) == 18:
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
                    self.codes[s0] = code
            elif len(ss) > 5:
                noth += 1
        print(f"(Found {nbar} legit bar codes and {noth} other groups)")

    @staticmethod
    def probablyCNTL(ss, fs_Hz):
        # Guess whether ss represent CNTL-style bar codes as opposed to OpenEphys style
        sss = inferblocks(ss, fs_Hz)
        balance = 0
        for ss in sss:
            if len(ss) > 4:
                if len(ss) == 18 and ss[1] - ss[0] < 450 * fs_Hz / 30e3:
                    balance += 1
                else:
                    balance -= 1
        return balance > 0


class OpenEphysBarCodes(BarCodes):
    def __init__(self, ss, fs_Hz):
        super().__init__(ss, fs_Hz)
        sss = inferblocks(ss, fs_Hz)
        self.trivial = False
        self.codes = {}

        PREDURATION_MS = 20
        INTER_BARCODE_INTERVAL_S = 30
        BARCODE_BITS = 32
        BITDURATION_MS = (INTER_BARCODE_INTERVAL_S - 1) * 32 / BARCODE_BITS
        # See https://github.com/open-ephys/sync-barcodes/blob/main/arduino-barcodes/arduino-barcodes.ino line 37.
        PERIOD = BITDURATION_MS * fs_Hz / 1000

        for ss in sss:  # Loop over all the codes
            if len(ss) < 4:
                continue
            if len(ss) % 2 == 1:
                continue
            dton = ss[3::2] - ss[2::2] # skip first pulse
            dtoff = ss[2::2] - ss[1:-2:2]
            dtoff[0] -= PREDURATION_MS * fs_Hz / 1000  # First interval start marker
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
            self.codes[ss[0]] = value


class TimeMachine:
    """
    A class to manage the time alignment of the Nidaq and Imec streams.
    """
    def __init__(self, barcodes_dest=None, barcodes_source=None):
        '''Do not call without barcodes_dest and barcodes_source.'''
        self.ssbc_dest, self.ssbc_source = barcodes_dest.match(barcodes_source)
        if len(self.ssbc_dest) < 2:
            raise ValueError("Not enough barcodes to translate")

    def inverse(self):
        '''Returns a time machine that translates in the opposite direction.'''
        rev = TimeMachine()
        rev.ssbc_source = self.ssbc_dest
        rev.ssbc_dest = self.ssbc_source
        return rev

    def translatedata(self, t0_source, data_source):
        """
        Translate the data from the source to the destination time zone.

        """
        N = len(data_source)
        t1_source = t0_source + N
        # Figure out edges of interval in destination time zone
        t01d = self.translatetimes(np.array([t0_source, t1_source]))
        t0_dest = t01d[0]
        t1_dest = t01d[1]
        ttd = np.arange(t0_dest, t1_dest)
        # Figure out timepoints in source time zone corresponding to interval in dest.
        tts = self.inverse().translatetimes(ttd)
        # Interpolate the data
        data_dest = np.interp(tts, np.arange(t0_source, t1_source), data_source)
        return data_dest, t0_dest

    def translatetimes(self, ss_source):
        """
        Translate the events from the source to the destination time zone.
        Returns sample stamps in destination time zone.
        """
        ss_dest = np.interp(ss_source, self.ssbc_source, self.ssbc_dest)

        # Extrapolate times before the first barcode and after the last
        isbefore = np.nonzero(ss_source < self.ssbc_source[0])[0]
        if len(isbefore):
            print(f"Caution: Extrapolating {len(isbefore)} event(s) before start of bar codes")
            a_before, b_before = np.polyfit(self.ssbc_source[:2], self.ssbc_dest[:2], 1)
            ss_dest[isbefore] = a_before * ss_source[isbefore] + b_before

        isafter = np.nonzero(ss_source > self.ssbc_source[-1])[0]
        if len(isafter):
            print(f"Caution: Extrapolating {len(isafter)} event(s) after end of bar codes")
            a_after, b_after = np.polyfit(self.ssbc_source[-2:], self.ssbc_dest[-2:], 1)
            ss_dest[isafter] = a_after * ss_source[isafter] + b_after

        return ss_dest
