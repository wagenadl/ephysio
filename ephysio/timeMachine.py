class BarCodes:
    """
    Find different KINDS of barcodes from the events of  the Nidaq stream and the Imec stream.
    """
    pass

    @staticmethod
    def cntl(events):
        bc = BarCodes()
        bc.trivial = False
        bc.codes = {} # fill in the blanks
        pass

    @staticmethod
    def oephys(events):
        pass

    @staticmethod
    def single(events):
        pass

    # self.codes = { time -> code }
    # self.trivial = False for CNTL/OEPHYS and True for Kofiko; If Trivial, need to match time intervals, not just codes

    def match(self, other):
        """
        Return ss_self, ss_other: Match the barcodes from two different streams.
        """
        pass


bc1 = BarCodes.cntl(events1)

class TimeMachine:
    """
    A class to manage the time alignment of the Nidaq and Imec streams.
    """
    def __init__(self, barcodes_dest, barcodes_source):
        pass

    def translatedata(self, t0_dest, data_source):
        """
        Translate the data from the source to the destination time.
        Return dest_data
        """
        pass

    def translateevents(self, evts_source):
        """
        Translate the events from the source to the destination time.
        Return dest_evts
        """
        pass