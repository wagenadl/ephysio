#!/usr/bin/python3

try:
    from dash import Dash, dcc, html, Input, Output, State, callback, ctx
    import numpy as np
    import plotly.express as px
except ModuleNotFoundError:
    print('''
======================================================================
To use vizly, you must first install Dash and plotly, e.g., by running

    pip install dash
======================================================================''')
    raise


class Vizly(Dash):
    def __init__(self, dat, fhz, tstim_s=[]):
        super().__init__("Vizly")
        self.dat = dat
        self.fhz = fhz
        self.istim = 0
        self.tstim_s = tstim_s
        self.t0_s = 0
        self.c0 = 0
        self.dt_s = 1
        self.dc = 20
        self.vscl = 500
        lay = [
            html.Span("• Voltage: "),
            html.Button('⤓', id='b-ybot'),
            html.Button('↓', id='b-down'),
            html.Button('↑', id='b-up'),
            html.Button('⤒', id='b-ytop'),
            html.Span(" "),
            html.Button('+', id='b-yzoomin'),
            html.Button('–', id='b-yzoomout'),
            html.Span(" • Channels: "),
            html.Button('+', id='b-plus'),
            html.Button('–', id='b-minus'),
            html.Div(),
            html.Span("• Time: "),
            html.Button('⇤', id='b-xhome'),
            html.Button('←', id='b-left'),
            html.Button('→', id='b-right'),
            html.Button('⇥', id='b-xend'),
            html.Span(" "),
            html.Button('+', id='b-xzoomin'),
            html.Button('–', id='b-xzoomout'),
            html.Span(" "),
            dcc.Input(id='go-time', placeholder="Time (s)", type="text", debounce=True),
        ]
        if len(tstim_s):
            lay += [
                html.Div(),
                html.Span("• Stimulus: "),
                html.Button('⇤', id='b-shome'),
                html.Button('←', id='b-sleft'),
                html.Button('→', id='b-sright'),
                html.Button('⇥', id='b-send'),
                html.Span(" "),
                dcc.Input(id='go-stim', placeholder="Stim #", type="number", debounce=True),
            ]
        lay += [
            dcc.Graph(id='graph-content')
        ]
        self.layout = html.Div(lay)
        inputs = [Input('go-time', 'value'),
                  ]
        if len(tstim_s):
            inputs += [
                Input('go-stim', 'value'),
                Input('b-sleft', 'n_clicks'),
                Input('b-sright', 'n_clicks'),
                Input('b-shome', 'n_clicks'),
                Input('b-send', 'n_clicks'),
            ]
        inputs += [Input('b-left', 'n_clicks'),
                  Input('b-right', 'n_clicks'),
                  Input('b-up', 'n_clicks'),
                  Input('b-down', 'n_clicks'),
                  Input('b-yzoomin', 'n_clicks'),
                  Input('b-yzoomout', 'n_clicks'),
                  Input('b-xzoomin', 'n_clicks'),
                  Input('b-xzoomout', 'n_clicks'),
                  Input('b-ytop', 'n_clicks'),
                  Input('b-ybot', 'n_clicks'),
                  Input('b-xhome', 'n_clicks'),
                  Input('b-xend', 'n_clicks'),
                  Input('b-plus', 'n_clicks'),
                  Input('b-minus', 'n_clicks'),
               ]
        outputs = [Output('graph-content', 'figure'),
                  Output('go-time', 'value')]
        if len(tstim_s):
            outputs += [Output('go-stim', 'value')]
        @callback(*outputs, *inputs)
        def update_graph(tgo, isgo, *args):
            clk = ctx.triggered_id
            if clk=="b-left":
                self.t0_s = max(self.t0_s - self.dt_s, 0)
            elif clk=="b-right":
                self.t0_s = min(self.t0_s + self.dt_s, dat.shape[0]/self.fhz - self.dt_s)                
            elif clk=="b-up":
                self.c0 = min(self.c0 + self.dc, dat.shape[1] - self.dc)
            elif clk=="b-down":
                self.c0 = max(self.c0 - self.dc, 0)
            elif clk=="b-yzoomin":
                self.vscl = max(self.vscl/1.4, 50)
            elif clk=="b-yzoomout":
                self.vscl = min(self.vscl*1.4, 10000)
            elif clk=="b-xzoomin":
                tc = self.t0_s + self.dt_s / 4
                self.dt_s = max(self.dt_s/2, .010)
                self.t0_s = tc - self.dt_s / 4
            elif clk=="b-xzoomout":
                tc = self.t0_s + self.dt_s / 4
                self.dt_s = min(self.dt_s*2, 2.0)
                self.t0_s = tc - self.dt_s / 4
            elif clk=="b-ytop":
                self.c0 = dat.shape[1] - self.dc
            elif clk=="b-ybot":
                self.c0 = 0
            elif clk=="b-xhome":
                self.t0_s = 0
            elif clk=="b-xend":
                self.t0_s = dat.shape[0]/self.fhz - self.dt_s
            elif clk=="b-plus":
                self.dc = min(self.dc*2, 40)
            elif clk=="b-minus":
                self.dc = max(self.dc//2, 5)
            elif clk=="go-time":
                try:
                    t_s = float(tgo)
                    self.t0_s = max(min(t_s, dat.shape[0]/self.fhz - self.dt_s), 0)
                except:
                    pass # simply ignore illegal input
            elif clk=="b-shome":
                self.istim = 0
                self.t0_s = self.tstim_s[self.istim] - self.dt_s/4
            elif clk=="b-send":
                self.istim = len(self.tstim_s) - 1
                self.t0_s = self.tstim_s[self.istim] - self.dt_s/4
            elif clk=="b-sleft":
                self.istim = max(self.istim - 1, 0)
                self.t0_s = self.tstim_s[self.istim] - self.dt_s/4
            elif clk=="b-sright":
                self.istim = min(self.istim + 1, len(self.tstim_s) - 1)
                self.t0_s = self.tstim_s[self.istim] - self.dt_s/4
            elif clk=="go-stim":
                try: 
                    istim = int(isgo)
                    self.istim = max(min(istim, len(self.tstim_s) - 1), 0)
                    self.t0_s = self.tstim_s[self.istim] - self.dt_s/4
                except:
                    pass # simply ignore illegal input
            i0 = int(self.t0_s*self.fhz)
            di = int(self.dt_s*self.fhz)
            ddi = max(1, di//2000)
            xx = np.arange(0, di, ddi)/self.fhz + self.t0_s
            yyy = []
            for c in range(self.c0, self.c0+self.dc):
                yy = self.dat[i0:i0+di:ddi, c] / self.vscl
                yyy.append(yy - yy.mean() + c)
            fig = px.line(x=xx, y=yyy, labels={"x":"Time", "value": "Channels"})
            for d in fig.data:
                d['showlegend'] = False
            if len(self.tstim_s):
                return fig, f"{self.t0_s:.3f}", self.istim
            else:
                return fig, f"{self.t0_s:.3f}"

        
