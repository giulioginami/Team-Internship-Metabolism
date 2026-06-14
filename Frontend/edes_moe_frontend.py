"""
edes_moe_frontend.py — Interactive frontend for the MoE-EDES diabetes pipeline
==============================================================================

An interactive Dash layer over this project's Mixture-of-Experts EDES model.
It reproduces the exact EDES ODE used in MATLAB (EDES_PID/EDES_ODE.m), the three
population-specific PID experts (NGT / IGT / T2DM), and the trained neural gating
network — all in pure Python so the model can be explored without MATLAB.

What you can do:
  * Pick one of the 118 real Japanese OGTT patients (or "Manual").
  * The gating network runs a forward pass on the patient's sparse OGTT
    [G(0,30,60,90,120) | I(0,30,60,90,120)] and produces expert weights w.
  * Tune the two MoE-optimised parameters k1 (gastric emptying) and k5
    (insulin-dependent glucose uptake); k6 and k8 stay expert-specific.
  * Inspect the three expert EDES simulations, the MoE weighted prediction,
    the observed sparse data, and the population reference shadow for the
    patient's ADA category.

Model / parameter conventions match the project README:
  Experts (k5, k6, k8) from PID_optimization.m:
      NGT  k5=0.092  k6=0.079  k8=7.394
      IGT  k5=0.006  k6=0.089  k8=4.724
      T2DM k5=0.014  k6=0.000  k8=5.755
  Gating input  [10] = [G_sparse | I_sparse] standardised with X_mean / X_std.
  Fixed EDES params taken from EDES_Parameters.m / EDES_Initial.m.

Run:
    pip install dash dash-bootstrap-components plotly numpy scipy
    python Frontend/edes_moe_frontend.py
    # open http://127.0.0.1:8050
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import scipy.io as sio
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# EDES model — faithful port of EDES_PID/EDES_ODE.m
# ─────────────────────────────────────────────────────────────────────────────
# Fixed parameters (EDES_Parameters.m) and model constants (EDES_Initial.m).
# Rozendaal et al. 2018 baseline; only k1, k5, k6, k8 vary in the MoE pipeline.
FIXED = {
    "k2": 0.28,        # glucose appearance from gut [1/min]
    "k3": 6.07e-3,     # hepatic suppression by plasma glucose change [1/min]
    "k4": 2.35e-4,     # hepatic suppression by remote insulin [1/min]
    "k7": 1.15,        # integral gain on insulin production (beta cell) [1/min]
    "k9": 3.83e-2,     # plasma -> remote insulin outflow [1/min]
    "k10": 2.84e-1,    # remote insulin degradation [1/min]
    "sigma": 1.4,      # meal appearance shape factor [-]
    "KM": 13.2,        # Michaelis-Menten coefficient for glucose uptake [mmol/L]
    "f_G": 0.005551,   # glucose conversion mg/L -> mmol/L
    "f_I": 1.0,        # insulin conversion uIU/mL -> (kept 1 as in EDES_Initial.m)
    "V_G": 17.0 / 70.0,  # glucose distribution volume [L/kg]
    "tau_i": 31.0,     # integration time constant [min]
    "tau_d": 3.0,      # differential time constant [min]
    "G_th_PL": 9.0,    # renal extraction threshold [mmol/L]
    "c1": 0.1,         # renal extraction rate constant [L/min]
    "t_iw": 30.0,      # moving integral window for G_int [min]
    "G_liv_b": 0.043,  # basal hepatic glucose release
    "D_meal_G": 75.0e3,  # 75 g oral glucose load expressed in mg
}

SPARSE_T = np.array([0.0, 30.0, 60.0, 90.0, 120.0])

# Optimised expert PID parameters (README §2 / PID_optimization.m).
EXPERTS = {
    "NGT":  {"k5": 0.092, "k6": 0.079, "k8": 7.394, "color": "#2e9e2e"},
    "IGT":  {"k5": 0.006, "k6": 0.089, "k8": 4.724, "color": "#ee9b14"},
    "T2DM": {"k5": 0.014, "k6": 0.000, "k8": 5.755, "color": "#cc2222"},
}
EXPERT_ORDER = ["NGT", "IGT", "T2DM"]


def simulate_edes(k1, k5, k6, k8, G_b, I_PL_b, BW, dt=0.1, tmax=240.0):
    """Explicit-Euler integration of the EDES 5-state ODE (EDES_ODE.m).

    States: [M_G_gut, G_PL, G_int, I_PL, I_d1]. Returns dense trajectories plus
    the per-step P/I/D contributions to pancreatic insulin production.
    """
    F = FIXED
    n = int(tmax / dt) + 1
    t = np.linspace(0.0, tmax, n)

    y = np.zeros((n, 5), dtype=float)
    y[0] = np.array([0.0, G_b, 0.0, I_PL_b, 0.0])  # EDES_Initial.m

    p_term = np.zeros(n)
    i_term = np.zeros(n)
    d_term = np.zeros(n)

    def g_pl_at(t_query):
        """Plasma glucose history for the moving-window integral (G_int)."""
        if t_query <= 0.0:
            return G_b
        idx = t_query / dt
        lo = min(max(int(np.floor(idx)), 0), n - 1)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return y[lo, 1] * (1.0 - frac) + y[hi, 1] * frac

    for i in range(n - 1):
        ti = t[i]
        M_G_gut, G_PL, G_int, I_PL, I_d1 = y[i]
        t_safe = max(ti, 1e-8)

        # Appearance of glucose from the meal
        G_meal = (F["sigma"] * (k1 ** F["sigma"]) * t_safe ** (F["sigma"] - 1.0)
                  * np.exp(-((k1 * t_safe) ** F["sigma"])) * F["D_meal_G"])
        dM = G_meal - F["k2"] * M_G_gut

        # Plasma glucose fluxes
        G_liv = F["G_liv_b"] - F["k4"] * F["f_I"] * I_d1 - F["k3"] * (G_PL - G_b)
        G_gut = F["k2"] * (F["f_G"] / (F["V_G"] * BW)) * M_G_gut
        U_ii = F["G_liv_b"] * ((F["KM"] + G_b) / G_b) * (G_PL / (F["KM"] + G_PL))
        U_id = k5 * F["f_I"] * I_d1 * (G_PL / (F["KM"] + G_PL))
        U_ren = (F["c1"] / (F["V_G"] * BW) * (G_PL - F["G_th_PL"])) * (G_PL > F["G_th_PL"])
        dG = G_liv + G_gut - U_ii - U_id - U_ren

        # Moving-window integral of (G_PL - G_b)
        G_PL_lb = g_pl_at(ti - F["t_iw"])
        dGint = (G_PL - G_b) - (G_PL_lb - G_b)

        # Pancreatic insulin production (PID controller on glucose)
        p_contrib = k6 * (G_PL - G_b)
        i_contrib = (F["k7"] / F["tau_i"]) * G_int
        d_contrib = k8 * F["tau_d"] * dG
        I_pnc = (1.0 / F["f_I"]) * (p_contrib + i_contrib
                                    + (F["k7"] / F["tau_i"]) * G_b + d_contrib)
        I_liv = F["k7"] * (G_b / (F["f_I"] * F["tau_i"] * I_PL_b)) * I_PL
        i_rem = F["k9"] * (I_PL - I_PL_b)
        dI = I_pnc - I_liv - i_rem
        dId = F["k9"] * (I_PL - I_PL_b) - F["k10"] * I_d1

        p_term[i], i_term[i], d_term[i] = p_contrib, i_contrib, d_contrib

        y[i + 1] = y[i] + dt * np.array([dM, dG, dGint, dI, dId])
        # Light clamping keeps interactive extremes physiologic.
        y[i + 1, 1] = max(y[i + 1, 1], 0.3)
        y[i + 1, 3] = max(y[i + 1, 3], 0.0)

    p_term[-1], i_term[-1], d_term[-1] = p_term[-2], i_term[-2], d_term[-2]
    return {"t": t, "gpl": y[:, 1], "ipl": y[:, 3],
            "pid_p": p_term, "pid_i": i_term, "pid_d": d_term}


def simulate_moe(k1, k5, weights, G_b, I_PL_b, BW):
    """Run all three experts (shared k1, k5; expert k6, k8) and blend by `weights`."""
    expert_sims = {}
    t = None
    g_mix = i_mix = p_mix = i_term_mix = d_mix = None
    for name in EXPERT_ORDER:
        e = EXPERTS[name]
        sim = simulate_edes(k1, k5, e["k6"], e["k8"], G_b, I_PL_b, BW)
        expert_sims[name] = sim
        w = weights[name]
        if t is None:
            t = sim["t"]
            g_mix = np.zeros_like(t)
            i_mix = np.zeros_like(t)
            p_mix = np.zeros_like(t)
            i_term_mix = np.zeros_like(t)
            d_mix = np.zeros_like(t)
        g_mix += w * sim["gpl"]
        i_mix += w * sim["ipl"]
        p_mix += w * sim["pid_p"]
        i_term_mix += w * sim["pid_i"]
        d_mix += w * sim["pid_d"]
    moe = {"t": t, "gpl": g_mix, "ipl": i_mix,
           "pid_p": p_mix, "pid_i": i_term_mix, "pid_d": d_mix}
    return expert_sims, moe


# ─────────────────────────────────────────────────────────────────────────────
# Data + gating-network loading
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
JAPAN_PATH = ROOT / "EDES_MoE" / "Datasets" / "Real Dataset" / "japan_population_labelled.mat"
GATING_PATH = ROOT / "EDES_MoE" / "Gating Network" / "gating_weights.mat"


def load_gating():
    """Load exported gating-network weights for the pure-Python forward pass."""
    if not GATING_PATH.exists():
        return None
    gw = sio.loadmat(GATING_PATH)
    return {
        "W1": gw["W1"], "b1": gw["b1"].ravel(),
        "W2": gw["W2"], "b2": gw["b2"].ravel(),
        "W3": gw["W3"], "b3": gw["b3"].ravel(),
        "X_mean": gw["X_mean"].ravel(), "X_std": gw["X_std"].ravel(),
    }


def gating_forward(gnet, g_sparse, i_sparse):
    """Forward pass: [G|I] -> softmax expert weights (matches Fit_MoE.m §2)."""
    x = np.concatenate([np.asarray(g_sparse, float), np.asarray(i_sparse, float)])
    xn = (x - gnet["X_mean"]) / gnet["X_std"]
    h1 = np.maximum(0.0, gnet["W1"] @ xn + gnet["b1"])
    h2 = np.maximum(0.0, gnet["W2"] @ h1 + gnet["b2"])
    z = gnet["W3"] @ h2 + gnet["b3"]
    e = np.exp(z - z.max())
    w = e / e.sum()
    return {name: float(w[k]) for k, name in enumerate(EXPERT_ORDER)}


def load_japan():
    """Load the 118-patient Japan OGTT dataset and per-category reference stats."""
    if not JAPAN_PATH.exists():
        return None
    jp = sio.loadmat(JAPAN_PATH, simplify_cells=True)["japan_population"]
    g = np.asarray(jp["glucose_noisy"], float)   # [118 x 5]
    ins = np.asarray(jp["insulin_noisy"], float)  # [118 x 5]
    bw = np.asarray(jp["BW"], float).ravel()
    t = np.asarray(jp["time"], float).ravel()

    masks = {
        "NGT": np.asarray(jp["is_NGT"], int).ravel().astype(bool),
        "IGT": np.asarray(jp["is_IGT"], int).ravel().astype(bool),
        "T2DM": np.asarray(jp["is_T2DM"], int).ravel().astype(bool),
    }
    label = np.array(["NGT"] * g.shape[0], dtype=object)
    label[masks["IGT"]] = "IGT"
    label[masks["T2DM"]] = "T2DM"

    def summarize(mask):
        if mask.sum() == 0:
            return None
        return {
            "n": int(mask.sum()),
            "g_q25": np.percentile(g[mask], 25, axis=0),
            "g_med": np.percentile(g[mask], 50, axis=0),
            "g_q75": np.percentile(g[mask], 75, axis=0),
            "i_q25": np.percentile(ins[mask], 25, axis=0),
            "i_med": np.percentile(ins[mask], 50, axis=0),
            "i_q75": np.percentile(ins[mask], 75, axis=0),
        }

    groups = {name: summarize(m) for name, m in masks.items()}
    return {"g": g, "ins": ins, "bw": bw, "t": t, "label": label, "groups": groups}


GNET = load_gating()
JAPAN = load_japan()

# Default fasting / BW used in "Manual" mode (population-ish baseline).
DEFAULT_GB = 5.0
DEFAULT_IB = 10.0
DEFAULT_BW = 70.0
DEFAULT_K1 = 0.028   # population median initial k1 (Fit_MoE.m)
DEFAULT_K5 = 0.05


def patient_options():
    if JAPAN is None:
        return [{"label": "Manual (no dataset found)", "value": "manual"}]
    opts = [{"label": "Manual (custom fasting values)", "value": "manual"}]
    for idx, lab in enumerate(JAPAN["label"]):
        opts.append({"label": f"Patient {idx + 1:3d}  ·  {lab}", "value": str(idx)})
    return opts


# ─────────────────────────────────────────────────────────────────────────────
# Dash app
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "MoE-EDES Frontend"


def make_slider(slider_id, label, min_v, max_v, step, value, marks):
    return html.Div(
        [
            html.Label(label, style={"fontWeight": "600", "fontSize": "0.86rem"}),
            dcc.Slider(id=slider_id, min=min_v, max=max_v, step=step, value=value,
                       marks=marks, tooltip={"placement": "bottom", "always_visible": True}),
        ],
        style={"marginBottom": "16px"},
    )


_data_note = (
    "Live: Japan dataset + gating network loaded."
    if (JAPAN is not None and GNET is not None)
    else "Dataset/gating files not found — running in manual mode only."
)

app.layout = dbc.Container(
    [
        dcc.Store(id="patient-store"),
        dbc.Row(dbc.Col(html.H3(
            "MoE-EDES Frontend: Gating Network + Expert Mixture",
            className="text-center mt-3 mb-1",
            style={"fontWeight": "700", "color": "#1f2d3d"}))),
        dbc.Row(dbc.Col(html.P(
            "Interactive layer over the Mixture-of-Experts EDES pipeline. Select a real "
            "patient, watch the gating network weight the NGT/IGT/T2DM experts, then tune "
            "the personalised parameters k1 and k5.",
            className="text-center text-muted mb-1"))),
        dbc.Row(dbc.Col(html.P(_data_note, className="text-center",
                               style={"fontSize": "0.8rem", "color": "#6c757d"}))),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Patient (Japan OGTT dataset)", style={"fontWeight": "700"}),
                                dbc.CardBody(
                                    [
                                        dcc.Dropdown(id="patient-dropdown",
                                                     options=patient_options(),
                                                     value="0" if JAPAN is not None else "manual",
                                                     clearable=False),
                                        html.Div(id="patient-info",
                                                 style={"marginTop": "10px", "fontSize": "0.84rem"}),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Gating Network Weights", style={"fontWeight": "700"}),
                                dbc.CardBody(dcc.Graph(id="weights-graph",
                                                       style={"height": "190px"},
                                                       config={"displayModeBar": False})),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Personalised Parameters (k1, k5)", style={"fontWeight": "700"}),
                                dbc.CardBody(
                                    [
                                        make_slider("k1-slider", "k1 (gastric emptying rate)",
                                                    0.0, 0.05, 0.0005, DEFAULT_K1,
                                                    {0: "0", 0.028: "med", 0.05: "0.05"}),
                                        make_slider("k5-slider", "k5 (insulin-dependent glucose uptake)",
                                                    0.0, 0.17, 0.001, DEFAULT_K5,
                                                    {0: "0", 0.092: "NGT", 0.17: "0.17"}),
                                        dbc.Button("Reset k1, k5 to gating estimate",
                                                   id="reset-btn", size="sm",
                                                   color="secondary", outline=True),
                                        dcc.Checklist(
                                            id="shadow-toggle",
                                            options=[{"label": " Show population reference shadow (IQR + median)",
                                                      "value": "on"}],
                                            value=["on"], style={"marginTop": "12px", "fontSize": "0.85rem"}),
                                        html.Div(
                                            "k6 and k8 stay expert-specific and are mixed by the gating "
                                            "weights; all other EDES parameters are fixed to the Rozendaal "
                                            "2018 baseline.",
                                            style={"marginTop": "10px", "fontSize": "0.8rem", "color": "#6c757d"}),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(dbc.CardBody([
                        dcc.Graph(id="glucose-graph", style={"height": "270px"}),
                        dcc.Graph(id="insulin-graph", style={"height": "270px"}),
                        dcc.Graph(id="pid-graph", style={"height": "260px"}),
                    ])),
                    md=8,
                ),
            ]
        ),
    ],
    fluid=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    [Output("patient-store", "data"), Output("patient-info", "children")],
    Input("patient-dropdown", "value"),
)
def select_patient(value):
    """Resolve the dropdown selection into fasting values, BW, sparse obs and gating weights."""
    if value == "manual" or JAPAN is None:
        weights = {"NGT": 1 / 3, "IGT": 1 / 3, "T2DM": 1 / 3}
        data = {"mode": "manual", "G_b": DEFAULT_GB, "I_PL_b": DEFAULT_IB, "BW": DEFAULT_BW,
                "g_obs": None, "i_obs": None, "label": None, "weights": weights}
        info = html.Span("Manual mode — basal G=5.0 mmol/L, I=10 mU/L, BW=70 kg, "
                         "equal expert weights.", style={"color": "#6c757d"})
        return data, info

    idx = int(value)
    g_obs = JAPAN["g"][idx].tolist()
    i_obs = JAPAN["ins"][idx].tolist()
    bw = float(JAPAN["bw"][idx])
    label = str(JAPAN["label"][idx])

    if GNET is not None:
        weights = gating_forward(GNET, g_obs, i_obs)
    else:
        weights = {"NGT": 1 / 3, "IGT": 1 / 3, "T2DM": 1 / 3}

    data = {"mode": "patient", "G_b": g_obs[0], "I_PL_b": i_obs[0], "BW": bw,
            "g_obs": g_obs, "i_obs": i_obs, "label": label, "weights": weights}

    color = EXPERTS[label]["color"]
    top = max(weights, key=weights.get)
    info = html.Div([
        html.Span(f"True ADA label: ", style={"color": "#555"}),
        html.Span(label, style={"fontWeight": "700", "color": color}),
        html.Br(),
        html.Span(f"Fasting G {g_obs[0]:.1f} mmol/L · I {i_obs[0]:.0f} mU/L · BW {bw:.0f} kg",
                  style={"color": "#555"}),
        html.Br(),
        html.Span(f"Gating favours: ", style={"color": "#555"}),
        html.Span(top, style={"fontWeight": "700", "color": EXPERTS[top]["color"]}),
    ])
    return data, info


@app.callback(
    [Output("k1-slider", "value"), Output("k5-slider", "value")],
    [Input("patient-store", "data"), Input("reset-btn", "n_clicks")],
)
def set_default_params(store, _n):
    """Seed k1 (median) and k5 (gating-weighted expert k5) — Fit_MoE.m initial guess."""
    if not store:
        return DEFAULT_K1, DEFAULT_K5
    w = store["weights"]
    k5_init = sum(w[name] * EXPERTS[name]["k5"] for name in EXPERT_ORDER)
    return DEFAULT_K1, round(k5_init, 3)


@app.callback(
    Output("weights-graph", "figure"),
    Input("patient-store", "data"),
)
def render_weights(store):
    weights = store["weights"] if store else {n: 1 / 3 for n in EXPERT_ORDER}
    vals = [weights[n] for n in EXPERT_ORDER]
    colors = [EXPERTS[n]["color"] for n in EXPERT_ORDER]
    fig = go.Figure(go.Bar(x=EXPERT_ORDER, y=vals, marker_color=colors,
                           text=[f"{v:.2f}" for v in vals], textposition="outside"))
    fig.update_layout(template="plotly_white", yaxis={"range": [0, 1], "title": "weight"},
                      margin={"l": 40, "r": 10, "t": 10, "b": 30}, showlegend=False)
    return fig


@app.callback(
    [Output("glucose-graph", "figure"), Output("insulin-graph", "figure"),
     Output("pid-graph", "figure")],
    [Input("k1-slider", "value"), Input("k5-slider", "value"),
     Input("shadow-toggle", "value"), Input("patient-store", "data")],
)
def update_graphs(k1, k5, shadow_toggle, store):
    if not store:
        store = {"G_b": DEFAULT_GB, "I_PL_b": DEFAULT_IB, "BW": DEFAULT_BW,
                 "g_obs": None, "i_obs": None, "label": None,
                 "weights": {n: 1 / 3 for n in EXPERT_ORDER}}

    weights = store["weights"]
    expert_sims, moe = simulate_moe(k1, k5, weights, store["G_b"], store["I_PL_b"], store["BW"])
    t = moe["t"]

    show_shadow = "on" in (shadow_toggle or []) and JAPAN is not None
    label = store.get("label")
    ref = JAPAN["groups"].get(label) if (show_shadow and label) else None

    # ---- Glucose ----
    fig_g = go.Figure()
    if ref is not None:
        _add_shadow(fig_g, JAPAN["t"], ref["g_q25"], ref["g_q75"], ref["g_med"],
                    "rgba(33,102,172,0.15)", "rgba(33,102,172,0.7)", f"{label} reference", ref["n"])
    for name in EXPERT_ORDER:
        fig_g.add_trace(go.Scatter(x=t, y=expert_sims[name]["gpl"], mode="lines",
                                   line={"width": 1.6, "color": EXPERTS[name]["color"], "dash": "dot"},
                                   opacity=0.55, name=f"{name} expert"))
    fig_g.add_trace(go.Scatter(x=t, y=moe["gpl"], mode="lines",
                               line={"width": 3.2, "color": "#111"}, name="MoE prediction"))
    if store.get("g_obs"):
        fig_g.add_trace(go.Scatter(x=SPARSE_T, y=store["g_obs"], mode="markers",
                                   marker={"size": 9, "color": "#111", "symbol": "circle-open",
                                           "line": {"width": 2}}, name="Observed"))
    fig_g.add_hline(y=7.8, line_dash="dot", line_color="#d73027", annotation_text="IGT 2h threshold")
    fig_g.add_vline(x=120.0, line_dash="dash", line_color="#999")
    _layout(fig_g, "Glucose Response", "Glucose (mmol/L)")

    # ---- Insulin ----
    fig_i = go.Figure()
    if ref is not None:
        _add_shadow(fig_i, JAPAN["t"], ref["i_q25"], ref["i_q75"], ref["i_med"],
                    "rgba(230,130,20,0.15)", "rgba(230,130,20,0.7)", f"{label} reference", ref["n"])
    for name in EXPERT_ORDER:
        fig_i.add_trace(go.Scatter(x=t, y=expert_sims[name]["ipl"], mode="lines",
                                   line={"width": 1.6, "color": EXPERTS[name]["color"], "dash": "dot"},
                                   opacity=0.55, name=f"{name} expert"))
    fig_i.add_trace(go.Scatter(x=t, y=moe["ipl"], mode="lines",
                               line={"width": 3.2, "color": "#111"}, name="MoE prediction"))
    if store.get("i_obs"):
        fig_i.add_trace(go.Scatter(x=SPARSE_T, y=store["i_obs"], mode="markers",
                                   marker={"size": 9, "color": "#111", "symbol": "circle-open",
                                           "line": {"width": 2}}, name="Observed"))
    fig_i.add_vline(x=120.0, line_dash="dash", line_color="#999")
    _layout(fig_i, "Insulin Response", "Insulin (mU/L)")

    # ---- PID contributions of the MoE prediction ----
    fig_pid = go.Figure()
    fig_pid.add_trace(go.Scatter(x=t, y=moe["pid_p"], mode="lines",
                                 line={"width": 2.4, "color": "#1b9e77"}, name="P (k6 · ΔG)"))
    fig_pid.add_trace(go.Scatter(x=t, y=moe["pid_i"], mode="lines",
                                 line={"width": 2.4, "color": "#7570b3"}, name="I (k7/τi · G_int)"))
    fig_pid.add_trace(go.Scatter(x=t, y=moe["pid_d"], mode="lines",
                                 line={"width": 2.4, "color": "#d95f02"}, name="D (k8·τd · dG)"))
    fig_pid.add_vline(x=120.0, line_dash="dash", line_color="#999")
    _layout(fig_pid, "PID Contribution to Pancreatic Insulin (MoE-weighted)", "Contribution (mU/L/min)")
    fig_pid.update_layout(legend={"orientation": "h", "y": 1.16, "x": 0.0})
    return fig_g, fig_i, fig_pid


def _add_shadow(fig, x, q25, q75, med, fill, line, name, n):
    fig.add_trace(go.Scatter(x=x, y=q75, mode="lines", line={"width": 0},
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=q25, mode="lines", line={"width": 0}, fill="tonexty",
                             fillcolor=fill, name=f"{name} IQR (n={n})", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=med, mode="lines",
                             line={"width": 2, "color": line, "dash": "dash"},
                             name=f"{name} median"))


def _layout(fig, title, ytitle):
    fig.update_layout(title=title, xaxis_title="Time (min)", yaxis_title=ytitle,
                      template="plotly_white", margin={"l": 45, "r": 20, "t": 40, "b": 30})


if __name__ == "__main__":
    app.run(debug=False, port=8050)
