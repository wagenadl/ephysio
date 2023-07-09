#!/usr/bin/python3

try:
    from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog
    from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygon
    from PyQt5.QtCore import Qt, QRect, QPoint
except ModuleNotFoundError:
    print('''
============================================================
To use vizio, you must first install PyQt5, e.g., by running

    pip install PyQt5
============================================================''')
    raise

import numpy as np


def lastlessthan(xx, y):
    j = None
    for i in range(len(xx)):
        if xx[i]<y:
            j = i
        else:
            return j

        
def firstgreaterthan(xx, y):
    for i in range(len(xx)):
        if xx[i]>y:
            return i
    return None


def sensiblestep(mx):
    '''dx = SENSIBLESTEP(mx) returns a sensible step size not much smaller
    than MX:

      1<=MX<2  -> DX=1
      2<=MX<5  -> DX=2
      5<=MX<10 -> DX=5
    etc.'''
    
    lg = np.log10(mx)
    mag = np.floor(lg)
    sub = 10**(lg-mag)
    if sub>5:
        sub = 5
    elif sub>2:
        sub = 2
    else:
        sub = 1
    return sub * 10**mag


class _EPhysView(QWidget):
    def __init__(self, parent=None):
        self.mem = None
        # DJP edits
        self.showTraces=True
        self.fs_Hz = None
        self.chlist = None
        self.tscale_s = .1
        self.csep_digi = None
        self.cscale_chans = 0
        self.t0_s = 0
        self.c0 = 0
        self.margin_left = 60
        self.margin_top = 20
        self.margin_bottom = 20
        self.ticklen = 3
        self.tstim_s = np.array([])
        self.stimlabels = []
        self.spikes = [] # List of (c, tt) pairs
        super().__init__(parent)
        self.setWindowTitle('Vizio')

    def setData(self, mem, fs_Hz, chlist=None):
        '''SETDATA - Specify data to display
        SETDATA(mem, fs_Hz) specifies the data (shaped TxC) to display and
        the sample rate for the data.
        Optional argument CHLIST must be a list or dict with channel numbers.
        Each value in CHLIST must be a dict with channel info as contained
        in OpenEphys's metadata. The only key required here is CHANNEL_NAME.
        '''
        self.mem = mem
        self.fs_Hz = fs_Hz
        self.chlist = chlist
        self.c0 = 0
        self.cscale_chans = min(mem.shape[1], 10)
        T = min(self.mem.shape[0], 65536)
        self.csep_digi = np.median(np.std(self.mem[:T,:], 0)) * 10
        self.update()

    def setSpikes(self, spkmap):
        '''SETSPIKES - Add spike markers to the display
        SETSPIKES(spkmap), where SPKMAP is a list of (c, tt) pairs
        containing electrode channels (C) and associated vectors of 
        spike times (TT), adds graphical marks for those
        spikes.'''
        self.spikes = spkmap
        self.update()

    def setStimuli(self, tt_s, labels=None):
        '''SETSTIMULI - Add stimulus markers to the display
        SETSTIMULI(tt) where TT is a vector of times in seconds, adds
        stimulus markers to the display.
        SETSTIMULI(tt, labels) where LABELS is a list of labels that
        has the same length as TT also provides labels for the vis_stimuli.'''
        
        self.tstim_s = np.array(tt_s)
        self.stimlabels = labels
        self.update()

    def wheelEvent(self, evt):
        if self.mem is None:
            return
        delta = evt.pixelDelta()
        dx = -delta.x()
        dy = -delta.y()
        C = self.mem.shape[1]
        T = self.mem.shape[0] / self.fs_Hz
        self.c0 = int(max(min(self.c0 + dy/2, C - self.cscale_chans), 0))
        self.t0_s = max(min(self.t0_s + dx/10 * self.tscale_s,
                            T - self.tscale_s), 0)
                            
        self.update()

    def keyPressEvent(self, evt):
        k = evt.key()
        m = evt.modifiers()
        if m==Qt.ShiftModifier:
            scl = 5
        elif m==Qt.ControlModifier:
            scl = 25
        elif m==Qt.AltModifier:
            scl = .2
        else:
            scl = 1
        if k==Qt.Key_Plus or k==Qt.Key_Equal:
            self.csep_digi /= np.sqrt(2)
            self.setWindowTitle(f'Vertical scale: {self.csep_digi}')
        elif k==Qt.Key_Minus:
            self.csep_digi *= np.sqrt(2)
            self.setWindowTitle(f'Vertical scale: {self.csep_digi}')
        elif k==Qt.Key_Period:
            self.t0_s = self.t0_s + self.tscale_s/4
            self.tscale_s /= 2
            self.t0_s = max(0, self.t0_s - self.tscale_s/4)
        elif k==Qt.Key_Comma:
            self.t0_s = self.t0_s + self.tscale_s/4
            self.tscale_s *= 2
            self.t0_s = max(0, self.t0_s - self.tscale_s/4)
        elif k==Qt.Key_BracketRight:
            self.c0 = self.c0 + self.cscale_chans//2
            self.cscale_chans = max(10, int(self.cscale_chans/2))
            self.c0 = max(0, self.c0 - self.cscale_chans//2)
        elif k==Qt.Key_BracketLeft:
            self.c0 = self.c0 + self.cscale_chans//2
            self.cscale_chans = min(self.mem.shape[1], 100,
                                    int(self.cscale_chans*2))
            self.c0 = max(0, self.c0 - self.cscale_chans//2)
        elif k==Qt.Key_PageUp or k==Qt.Key_Up:
            self.c0 = max(0, int(self.c0 - scl*self.cscale_chans))
        elif k==Qt.Key_PageDown or k==Qt.Key_Down:
            self.c0 = max(0, min(int(self.c0 + scl*self.cscale_chans),
                                 self.mem.shape[1]-self.cscale_chans))
        elif k==Qt.Key_Left:
            self.t0_s = max(0, self.t0_s - scl*self.tscale_s)
        elif k==Qt.Key_Right:
            self.t0_s = max(0, min(self.t0_s + scl*self.tscale_s,
                                   self.mem.shape[0]/self.fs_Hz
                                   - self.tscale_s))
        elif k==Qt.Key_T:
            t,ok = QInputDialog.getDouble(None, 'Go to time:',
                                       '(seconds)',
                                       self.t0_s + self.tscale_s/2,
                                       self.tscale_s/2,
                                       self.mem.shape[0]/self.fs_Hz
                                       - self.tscale_s/2,
                                       3)
            if ok:
                self.t0_s = max(0, min(t - self.tscale_s/2,
                                       self.mem.shape[0]/self.fs_Hz
                                       - self.tscale_s))
        elif k==Qt.Key_P:
            t0 = self.t0_s
            i = lastlessthan(self.tstim_s, t0)
            if i is not None:
                t1 = self.tstim_s[i]
                self.t0_s = max(0, t1 - self.tscale_s/4)
                lbl = f'Stimulus #{i} at {t1:.3f}'
                if self.stimlabels is not None:
                    lbl += f' “{self.stimlabels[i]}”'
                self.setWindowTitle(lbl)
        elif k==Qt.Key_N:
            t0 = self.t0_s + self.tscale_s/2
            i = firstgreaterthan(self.tstim_s, t0)
            if i is not None:
                t1 = self.tstim_s[i]
                self.t0_s = max(0, t1 - self.tscale_s/4)
                self.setWindowTitle(f'Stimulus #{i} at {t1:.3f}')
        elif k==Qt.Key_S:
            i,ok = QInputDialog.getInt(None, 'Go to stimulus:',
                                          '(#)',
                                          0,
                                          0,
                                          len(self.tstim_s))
            if ok:
                t1 = self.tstim_s[i]
                self.t0_s =  max(0, t1 - self.tscale_s/4)
                self.setWindowTitle(f'Stimulus #{i} at {t1:.3f}')
        # DJP edits
        elif k==Qt.Key_E:
            self.showTraces = not self.showTraces        
        else:
            return
        self.update()

    def paintEvent(self, evt):
        ptr = QPainter(self)
        try:
            self._doPaint(ptr)
        except Exception as e:
            traceback.print_exc()
            print('Exception:', e)
        finally:
            del ptr

    def _doPaint(self, ptr):
        ptr.fillRect(QRect(0,0,self.width(),self.height()),
                     QColor(0,0,0))
        w = self.width() - self.margin_left
        h = self.height() - self.margin_top - self.margin_bottom
        if w<5 or h<5:
            return

        def t2x(t): # Single value
            return int(self.margin_left
                       + (t-self.t0_s)*w/self.tscale_s + .5)
        def tt2x(t): # Vector
            return (self.margin_left
                    + (t-self.t0_s)*w/self.tscale_s + .5).astype(int)
        def cv2y(c, v=0): # Single value
            return int(self.margin_top + h*(c+.5-self.c0)/self.cscale_chans
                       + h*v/self.cscale_chans/self.csep_digi + .5)

        self._drawTimeAxis(ptr, t2x, sensiblestep(self.tscale_s/(1 + w/200)), h)
        self._drawChannelNames(ptr, cv2y)
        self._drawStimuli(ptr, tt2x, self.margin_top, h)
        self._drawSpikes(ptr, tt2x, cv2y)
        # DJP edit
        if self.mem is not None and self.showTraces:
            self._drawEphysTraces(ptr, t2x, cv2y, w , h)

    def _drawTimeAxis(self, ptr, t2x, ttick, h):
        t0 = np.ceil(self.t0_s/ttick) * ttick
        t1 = np.floor((self.t0_s+self.tscale_s)//ttick) * ttick

        # Draw text for time along top and bottom
        ptr.setPen(QColor(255, 255, 255))
        for t in np.arange(t0, t1, ttick):
            x = t2x(t)
            ptr.drawText(QRect(x-100, 0,
                               200, self.margin_top),
                         Qt.AlignCenter,
                         f'{t:.3f}')
            ptr.drawText(QRect(x-100, self.margin_top+h,
                               200, self.margin_bottom),
                         Qt.AlignCenter,
                         f'{t:.3f}')

        # Draw tick marks for time
        ptr.setPen(QColor(64, 64, 64))
        for t in np.arange(t0, t1, ttick):
            x = t2x(t)
            ptr.drawLine(QPoint(x, self.margin_top),
                         QPoint(x, self.margin_top+h))

    def _drawChannelNames(self, ptr, cv2y):
        # Draw channel names
        ptr.setPen(QColor(255, 255, 255))
        for c in np.arange(self.c0, self.c0+self.cscale_chans, dtype=int):
            y = cv2y(c)
            if self.chlist is None:
                cname = f'{c}'
            else:
                cname = self.chlist[c]['channel_name']
            ptr.drawText(QRect(0, y-50,
                               self.margin_left - 5, 100),
                         Qt.AlignRight + Qt.AlignVCenter,
                         cname)

    def _drawEphysTraces(self, ptr, t2x, cv2y, w, h):
        # Draw electrophysiology traces
        L = self.mem.shape[0]
        K = int(self.tscale_s * self.fs_Hz)
        k0 = min(int(self.t0_s * self.fs_Hz), L)
        k1 = min(k0 + K, L)
        C = self.mem.shape[1]
        c1 = min(self.c0 + self.cscale_chans, C)
        x0 = t2x(self.t0_s)
        L = k1 - k0 # length of data segment we may use
        fac = L//w # data reduction factor
        if fac<4:
            fac = 1 # just draw individual points
        N = L//fac # number of points to draw
        L1 = N*fac # length of data segment to actually use
        xx = x0 + np.arange(N) * fac * w/K
        yscl = h/self.cscale_chans/self.csep_digi
        sizeof_int = 4
        if fac>1:
            poly = QPolygon(2*N)
            dat = poly.data()
            dat.setsize(2*N * 2 * sizeof_int)
            aa = np.ndarray(shape=(2*N, 2), dtype=np.int32, buffer=dat)
            aa[:N, 0] = xx
            aa[N:, 0] = xx[-1::-1]
        else:
            poly = QPolygon(N)
            dat = poly.data()
            dat.setsize(N * 2 * sizeof_int)
            aa = np.ndarray(shape=(N, 2), dtype=np.int32, buffer=dat)
            aa[:,0] = xx
        for c in np.arange(self.c0, c1, dtype=int):
            clr = QColor(int(150+100*np.cos(c*2.127+.1238)),
                         int(150+100*np.cos(c*4.5789+4.123809)),
                         int(150+100*np.cos(c*8.123+23.32412)))
            ptr.setBrush(clr)
            ptr.setPen(clr)
            dt1 = self.mem[k0:k0+L1, c]
            if fac>1:
                yy0 = dt1.reshape(N, fac).min(-1)
                yy1 = dt1.reshape(N, fac).max(-1)
                aa[:N,1] = cv2y(c) - yy1 * yscl
                aa[N:,1] = cv2y(c) - yy0[-1::-1] * yscl
                ptr.drawPolygon(poly)
            else:
                yy = dt1
                aa[:,1] = cv2y(c) - yy * yscl
                ptr.drawPolyline(poly)

    def _drawStimuli(self, ptr, tt2x, y0, h):
        t0 = self.t0_s
        t1 = self.t0_s+self.tscale_s
        tt = self.tstim_s
        xx = tt2x(tt[np.logical_and(tt>=t0, tt<t1)])
        if len(xx):
            ptr.setPen(QPen(QColor(255,0,0), 2))
            for x in xx:
                ptr.drawLine(QPoint(x,y0), QPoint(x,y0+h))
                
    def _drawSpikes(self, ptr, tt2x, cv2y):
        t0 = self.t0_s
        t1 = self.t0_s+self.tscale_s
        R = 6
        c1 = .5
        for cc, tt in self.spikes:
            c1 += 1
            if type(cc)==list or type(cc)==np.ndarray:
                pass
            else:
                cc = [cc]
            cc = np.array(cc)
            if not np.any(np.logical_and(cc>=self.c0,
                                         cc<self.c0 + self.cscale_chans)):
                continue
            xx = tt2x(tt[np.logical_and(tt>=t0, tt<t1)])
            if len(xx):
                clr = QColor(int(150+100*np.cos(c1*2.127+.1238)),
                             int(150+100*np.cos(c1*4.5789+4.123809)),
                             int(150+100*np.cos(c1*8.123+23.32412)))
                ptr.setBrush(clr)
                ptr.setPen(Qt.NoPen)
                r = R
                for c in cc:
                    y = cv2y(c-.25)
                    for x in xx:
                        # DJP edit
                        if self.showTraces:
                            ptr.drawEllipse(QRect(x-r, y-r, 2*r, 2*r))
                        else:
                            ptr.drawRect(x, y-r/2, 1, r)
                    r = R//2

            
class Viz:
    app = None
    
    def __init__(self):
        if QApplication.instance() is None:
            Viz.app = QApplication(["vizio"])
        self.win = _EPhysView()
        self.win.show()

    def __del__(self):
        try:
            self.win.close()
            del self.win
        except:
            pass

    def setData(self, mem, fs_Hz, chlist=None):
        '''SETDATA - Specify data to display
        SETDATA(mem, fs_Hz) specifies the data (shaped TxC) to display and
        the sample rate for the data.
        Optional argument CHLIST must be a list or dict with channel numbers.
        Each value in CHLIST must be a dict with channel info as contained
        in OpenEphys's metadata. The only key required here is CHANNEL_NAME.
        '''
        self.win.setData(mem, fs_Hz, chlist)

    def setStimuli(self, tt_s, labels=None):
        '''SETSTIMULI - Add stimulus markers to the display
        SETSTIMULI(tt) where TT is a vector of times in seconds, adds
        stimulus markers to the display.
        SETSTIMULI(tt, labels) where LABELS is a list of labels that
        has the same length as TT also provides labels for the markers.'''
        self.win.setStimuli(tt_s, labels)

    def setSpikes(self, spkmap):
        '''SETSPIKES - Add spike markers to the display
        SETSPIKES(spkmap), where SPKMAP is a list of (c, tt) pairs
        containing electrode channels (C) and associated vectors of
        spike times (TT, in seconds), adds graphical marks for those
        spikes.'''
        self.win.setSpikes(spkmap)

        
def viz(dat, fs_Hz, chlist=None, stims=None, spikes=None, self_contained=True):
    '''VIZ - Functional form of Vizio display
    VIZ(dat, fs_Hz) displays the data (TxC) for interactive exploration.
    fs_Hz must specify the sampling rate (in Hertz).

    Optional argument CHLIST specifies channel list as a dict with
    channel info as contained in OpenEphys's metadata. The only key
    required here is CHANNEL_NAME. If you use ephysio's openEphysIO.Loader,
    you can use the CHANNELLIST method from that class.

    Optional argument STIMS specifies a list of times (in seconds) where 
    stimulus markers should be plotted.

    Optional argument SPIKES specifies spike streams. SPIKES must be a
    list of (c, tt) pairs, where C is an electrode number and TT are
    spike times (in seconds) on that electrode. Each element in the
    SPIKES list corresponds to one unit from spike sorting. Note that
    it is perfectly allowed to have more than one unit (element in the
    SPIKES list) that plots on the same electrode.

    Optional argument SELF_CONTAINED can be set to False to cause VIZ to 
    return a Vizio structure and return immediately rather than to wait
    for the user to close the window.

    Key bindings:
    
      Left/right arrows: navigate through time
      Up/down arrows: navigate through channels
      Comma/period: zoom in/out in time
      Plus/minus: zoom in/out in voltage
      Open/close bracket: increase/decrease number of channels shown
      n/p: jump to next/previous stimulus marker
      t: open dialog to jump to specific time
      s: open dialog to jump to specific stimulus 
    '''
    wdg = Viz()
    wdg.setData(dat, fs_Hz, chlist)
    if stims is not None:
        wdg.setStimuli(stims)
    if spikes is not None:
        wdg.setSpikes(spikes)
    if self_contained:
        app = Viz.app
        if app is None:
            app = QApplication(["viz"])
        app.exec()
        return None
    else:
        return wdg
