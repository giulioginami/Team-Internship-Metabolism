import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint

# ============================================================
#  EDES DDE model — mirrors edes_dde.jl (Rozendaal / JuliaCon)
#  The integral term uses a delayed glucose value:
#      dGint = Gpl(t) - Gpl(t - t_int)    where t_int = 30 min
#  We approximate the DDE by carrying the history ourselves.
# ============================================================

def simulate_edes_dde(k4, k5, k6, k7, k8, beta=1.0):
    """
    Simulate the EDES DDE model. Fixed parameters taken from edes_dde.jl.
    Free parameters for the interactive demo: k4, k5, k6, k8.
    beta is fixed to 1.0 for all cases.
    """
    beta = 1.0
    k1    = 0.0105
    k2    = 0.28
    k3    = 6.07e-3
    # k7 is now a free parameter passed in
    k9    = 3.83e-2
    k10   = 2.84e-1
    tau_i = 31.0
    tau_d = 3.0
    Gren  = 9.0       # renal glucose threshold
    EGPb  = 0.043     # basal hepatic glucose productionä
    Km    = 13.2
    f     = 0.005551  # mg/dL -> mmol/L
    Vg    = 17.0/70.0
    c1    = 0.1
    t_int = 30.0      # DDE delay (min)
    sigma = 1.4
    bw    = 70.0
    Gb    = 5.0       # basal plasma glucose (mmol/L)
    Ib    = 10.0      # basal plasma insulin (mU/L)
    Dmeal = 75.0e3    # 75 g glucose oral load (mg)

    dt   = 0.5        # step size (min)
    tmax = 240.0
    N    = int(tmax / dt) + 1
    t_vals  = np.linspace(0, tmax, N)

    # State arrays: [Ggut, Gpl, Gint, Ipl, Irem]
    Y = np.zeros((N, 5))
    # Initial conditions (from Julia: u0 = [0, Gb, Gb, Ib, 0])
    Y[0] = [0.0, Gb, Gb, Ib, 0.0]

    # Pre-history: before t=0 all glucose = Gb
    def Ghist(t_query):
        """Return Gpl at (t - t_int), using pre-history = Gb for t<0."""
        if t_query < 0:
            return Gb
        idx = t_query / dt
        lo  = int(idx)
        lo  = min(lo, N-1)
        hi  = min(lo+1, N-1)
        frac = idx - lo
        return Y[lo, 1] * (1 - frac) + Y[hi, 1] * frac

    # Euler integration (fine enough for visualisation)
    for i in range(N - 1):
        t   = t_vals[i]
        st  = Y[i]
        Ggut, Gpl, Gint, Ipl, Irem = st

        # delayed glucose for the DDE integral term
        Gpl_delayed = Ghist(t - t_int)

        t_s  = max(t, 1e-8)   # avoid 0^negative for sigma<1
        # gut glucose
        G_meal = sigma * k1**sigma * t_s**(sigma-1) * np.exp(-(k1*t_s)**sigma) * Dmeal
        dGgut  = G_meal - k2 * Ggut

        # plasma glucose
        gliv  = EGPb - k3*(Gpl - Gb) - k4*beta*Irem
        ggut  = k2 * (f/(Vg*bw)) * Ggut
        u_ii  = EGPb * ((Km + Gb)/Gb) * (Gpl/(Km + Gpl))
        u_id  = k5 * beta * Irem * (Gpl/(Km + Gpl))
        u_ren = c1/(Vg*bw)*(Gpl - Gren) if Gpl > Gren else 0.0
        dGpl  = gliv + ggut - u_ii - u_id - u_ren

        # DDE integral (delta glucose over delay window)
        dGint = Gpl - Gpl_delayed

        # plasma insulin   — from Julia line 93:
        # i_pnc = beta^(-1) * (k6*(Gpl-Gb) + (k7/tau_i)*(Gint+Gb) + k8*tau_d*dGpl)
        i_pnc = (1/beta) * (k6*(Gpl - Gb) + (k7/tau_i)*(Gint + Gb) + k8*tau_d*dGpl)
        i_pnc = max(i_pnc, 0.0)
        i_liv = k7 * Gb * Ipl / (beta * tau_i * Ib)
        i_int = k9 * (Ipl - Ib)
        dIpl  = i_pnc - i_liv - i_int
        dIrem = i_int - k10 * Irem

        Y[i+1] = Y[i] + dt * np.array([dGgut, dGpl, dGint, dIpl, dIrem])
        # clamp to physically sensible values
        Y[i+1, 1] = max(Y[i+1, 1], 0.5)   # Gpl >= 0.5
        Y[i+1, 3] = max(Y[i+1, 3], 0.0)   # Ipl >= 0

    return t_vals, Y[:, 1], Y[:, 3]  # time, Gpl, Ipl


# ----------------------------------------------------------------
# Patient presets (beta fixed at 1.0 for all cases)
# ----------------------------------------------------------------
presets = {
    'Healthy': {
        'k4': 2.35e-4, 'k5': 0.0424, 'k6': 2.2975, 'k7': 1.15, 'k8': 7.27, 'beta': 1.0
    },
    'Impaired': {
        'k4': 2.35e-4 * 0.5, 'k5': 0.0424 * 0.4, 'k6': 2.2975 * 1.3, 'k7': 1.15, 'k8': 7.27 * 0.5, 'beta': 1.0
    },
    'T2D': {
        'k4': 2.35e-4 * 0.15, 'k5': 0.0424 * 0.1, 'k6': 2.2975 * 0.4, 'k7': 1.15 * 0.5, 'k8': 7.27 * 0.1, 'beta': 1.0
    }
}

# ----------------------------------------------------------------
# Dash layout
# ----------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "EDES Interactive — DDE Model"

SLIDER_STYLE = {"marginBottom": "12px"}

def make_slider(slider_id, label, min_v, max_v, step, value, marks, disabled=False):
    return html.Div([
        html.Label(label, style={"fontWeight": "600", "fontSize": "0.85rem"}),
        dcc.Slider(id=slider_id, min=min_v, max=max_v, step=step,
                   value=value, marks=marks, disabled=disabled,
                   tooltip={"placement": "bottom", "always_visible": True}),
    ], style=SLIDER_STYLE)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3(
            "EDES Model — Interactive DDE Simulation",
            className="text-center mt-3 mb-1",
            style={"fontWeight": "700", "color": "#2c3e50"}
        ))
    ]),
    dbc.Row([
        dbc.Col(html.P(
            "Tune the PID controller and insulin-sensitivity parameters below. "
            "The model uses the DDE formulation from edes_dde.jl (t_int = 30 min delay). "
            "Graphs update in real-time.",
            className="text-center text-muted mb-3", style={"fontSize": "0.85rem"}
        ))
    ]),

    dbc.Row([
        # ---- LEFT PANEL - controls ----
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔬 Patient Presets", style={"fontWeight": "700"}),
                dbc.CardBody([
                    dbc.ButtonGroup([
                        dbc.Button("✅ Healthy",          id="btn-healthy",  color="success", outline=True, className="me-1"),
                        dbc.Button("⚠️ Impaired (IGT)",  id="btn-impaired", color="warning", outline=True, className="me-1"),
                        dbc.Button("🔴 Type 2 Diabetes", id="btn-t2d",      color="danger",  outline=True),
                    ], className="d-flex flex-wrap gap-1")
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("🧠 Pancreatic Beta-Cell (PID Controller)", style={"fontWeight": "700"}),
                dbc.CardBody([
                    make_slider("k6-slider", "k6 — Proportional Gain (P): current glucose deviation",
                                0, 7, 0.05, presets['Healthy']['k6'],
                                {0: "0", 2.3: "Healthy", 7: "7"}),
                    make_slider("k7-slider", "k7 — Integral Gain (I): steady-state glucose correction",
                                0, 4, 0.05, presets['Healthy']['k7'],
                                {0: "0", 1.15: "Healthy", 4: "4"}),
                    make_slider("k8-slider", "k8 — Derivative Gain (D): first-phase / rate of rise",
                                0, 15, 0.1, presets['Healthy']['k8'],
                                {0: "0", 7.27: "Healthy", 15: "15"}),
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("💉 Insulin Resistance", style={"fontWeight": "700"}),
                dbc.CardBody([
                    make_slider("beta-slider", "β (beta) — Overall beta-cell / insulin action scaling",
                                1.0, 1.0, None, presets['Healthy']['beta'],
                                {1.0: "Fixed 1.0"}, disabled=True),
                    make_slider("k5-slider", "k5 — Peripheral insulin sensitivity (muscle/fat)",
                                0, 0.08, 0.001, presets['Healthy']['k5'],
                                {0: "0", 0.0424: "Healthy", 0.08: "0.08"}),
                    make_slider("k4-slider", "k4 — Hepatic insulin sensitivity (liver)",
                                0, 0.001, 0.00002, presets['Healthy']['k4'],
                                {0: "0", 2.35e-4: "Healthy", 0.001: "0.001"}),
                ])
            ], className="mb-3"),
        ], md=4),

        # ---- RIGHT PANEL - plots ----
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='glucose-graph', style={"height": "280px"}),
                    dcc.Graph(id='insulin-graph', style={"height": "280px"}),
                ])
            ])
        ], md=8)
    ])
], fluid=True)


# ---- Presets callback ----
@app.callback(
    [Output('k6-slider', 'value'), Output('k7-slider', 'value'), Output('k8-slider', 'value'),
     Output('k5-slider', 'value'), Output('k4-slider', 'value'),
     Output('beta-slider', 'value')],
    [Input('btn-healthy', 'n_clicks'), Input('btn-impaired', 'n_clicks'), Input('btn-t2d', 'n_clicks')],
    prevent_initial_call=True
)
def set_preset(h, i, t):
    ctx = dash.callback_context
    btn = ctx.triggered[0]['prop_id'].split('.')[0]
    p = presets['Healthy'] if btn == 'btn-healthy' else (presets['Impaired'] if btn == 'btn-impaired' else presets['T2D'])
    return p['k6'], p['k7'], p['k8'], p['k5'], p['k4'], 1.0


# ---- Simulation + plot callback ----
@app.callback(
    [Output('glucose-graph', 'figure'), Output('insulin-graph', 'figure')],
    [Input('k4-slider', 'value'), Input('k5-slider', 'value'),
     Input('k6-slider', 'value'), Input('k7-slider', 'value'), Input('k8-slider', 'value'),
     Input('beta-slider', 'value')]
)
def update_graphs(k4, k5, k6, k7, k8, beta):
    t, G, I = simulate_edes_dde(k4, k5, k6, k7, k8, 1.0)

    # --- glucose figure ---
    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(
        x=t, y=G, mode='lines',
        line=dict(color='#2980b9', width=3), name='Plasma Glucose'))
    fig_g.add_hline(y=5.0, line_dash="dash", line_color="#95a5a6",
                    annotation_text="Basal (5.0 mmol/L)", annotation_position="top right")
    fig_g.add_hline(y=7.8, line_dash="dot", line_color="#e74c3c",
                    annotation_text="IGT threshold (7.8)", annotation_position="bottom right")
    fig_g.update_layout(
        title="Plasma Glucose", xaxis_title="Time (min)", yaxis_title="Glucose (mmol/L)",
        template="plotly_white", margin=dict(l=40, r=20, t=40, b=30),
        yaxis_range=[0, max(15, float(max(G)) + 1)])

    # --- insulin figure ---
    fig_i = go.Figure()
    fig_i.add_trace(go.Scatter(
        x=t, y=I, mode='lines',
        line=dict(color='#e67e22', width=3), name='Plasma Insulin'))
    fig_i.add_hline(y=10.0, line_dash="dash", line_color="#95a5a6",
                    annotation_text="Basal (10.0 mU/L)", annotation_position="top right")
    fig_i.update_layout(
        title="Plasma Insulin", xaxis_title="Time (min)", yaxis_title="Insulin (mU/L)",
        template="plotly_white", margin=dict(l=40, r=20, t=40, b=30),
        yaxis_range=[0, max(150, float(max(I)) + 10)])

    return fig_g, fig_i


if __name__ == '__main__':
    app.run(debug=False, port=8050)
