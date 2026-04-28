import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from pathlib import Path


# EDES DDE approximation matching juliacon-2024/1_implementation/edes_dde.jl.
# Uses explicit stepping with delayed-history interpolation for interactive speed.
def simulate_edes_dde(params):
    k1 = params["k1"]
    k2 = params["k2"]
    k3 = params["k3"]
    k4 = params["k4"]
    k5 = params["k5"]
    k6 = params["k6"]
    k7 = params["k7"]
    k8 = params["k8"]
    k9 = params["k9"]
    k10 = params["k10"]
    tau_i = params["tau_i"]
    tau_d = params["tau_d"]
    beta = params["beta"]
    gren = params["gren"]
    egpb = params["egpb"]
    km = params["km"]
    f = params["f"]
    vg = params["vg"]
    c1 = params["c1"]
    t_int = params["t_int"]
    sigma = params["sigma"]
    dmeal = params["dmeal"]
    bw = params["bw"]
    gb = params["gb"]
    ib = params["ib"]

    dt = params["dt"]
    tmax = params["tmax"]
    n = int(tmax / dt) + 1
    t_vals = np.linspace(0.0, tmax, n)

    # [Ggut, Gpl, Gint, Ipl, Irem]
    y = np.zeros((n, 5), dtype=float)
    y[0] = np.array([0.0, gb, gb, ib, 0.0], dtype=float)

    p_term = np.zeros(n, dtype=float)
    i_term = np.zeros(n, dtype=float)
    d_term = np.zeros(n, dtype=float)

    def g_hist(t_query):
        if t_query <= 0.0:
            return gb
        idx = t_query / dt
        lo = int(np.floor(idx))
        lo = min(max(lo, 0), n - 1)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return y[lo, 1] * (1.0 - frac) + y[hi, 1] * frac

    for i in range(n - 1):
        t = t_vals[i]
        ggut, gpl, gint, ipl, irem = y[i]
        ghist = g_hist(t - t_int)

        t_safe = max(t, 1e-8)
        d_ggut = sigma * (k1 ** sigma) * (t_safe ** (sigma - 1.0)) * np.exp(-((k1 * t_safe) ** sigma)) * dmeal - k2 * ggut

        gliv = egpb - k3 * (gpl - gb) - k4 * beta * irem
        ggut_flux = k2 * (f / (vg * bw)) * ggut
        u_ii = egpb * ((km + gb) / gb) * (gpl / (km + gpl))
        u_id = k5 * beta * irem * (gpl / (km + gpl))
        u_ren = c1 / (vg * bw) * (gpl - gren) * (gpl > gren)

        d_gpl = gliv + ggut_flux - u_ii - u_id - u_ren
        d_gint = gpl - ghist

        p_now = k6 * (gpl - gb)
        i_now = (k7 / tau_i) * (gint + gb)
        d_now = k8 * tau_d * d_gpl

        p_term[i] = p_now / beta
        i_term[i] = i_now / beta
        d_term[i] = d_now / beta

        i_pnc = (1.0 / beta) * (p_now + i_now + d_now)
        i_liv = k7 * gb * ipl / (beta * tau_i * ib)
        i_int = k9 * (ipl - ib)
        d_ipl = i_pnc - i_liv - i_int
        d_irem = i_int - k10 * irem

        y[i + 1] = y[i] + dt * np.array([d_ggut, d_gpl, d_gint, d_ipl, d_irem], dtype=float)

        # Keep states in physiologic ranges for robust interactive behavior.
        y[i + 1, 1] = max(y[i + 1, 1], 0.5)
        y[i + 1, 3] = max(y[i + 1, 3], 0.0)

    p_term[-1] = p_term[-2]
    i_term[-1] = i_term[-2]
    d_term[-1] = d_term[-2]

    return {
        "t": t_vals,
        "gpl": y[:, 1],
        "ipl": y[:, 3],
        "pid_p": p_term,
        "pid_i": i_term,
        "pid_d": d_term,
        "gb": gb,
        "ib": ib,
    }


def make_slider(slider_id, label, min_v, max_v, step, value, marks):
    return html.Div(
        [
            html.Label(label, style={"fontWeight": "600", "fontSize": "0.88rem"}),
            dcc.Slider(
                id=slider_id,
                min=min_v,
                max=max_v,
                step=step,
                value=value,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        style={"marginBottom": "14px"},
    )


def _parse_semicolon_floats(tokens):
    vals = []
    for tok in tokens:
        tok = tok.strip()
        if tok == "":
            continue
        vals.append(float(tok))
    return np.array(vals, dtype=float)


def load_reference_shadows():
    """Load references and split into healthy/impaired/t2d by 120-min glucose."""
    base = Path(__file__).resolve().parent / "juliacon-2024" / "2_parameter_estimation" / "predict"
    tp_path = base / "timepoints.csv"
    g_path = base / "glucose.csv"
    i_path = base / "insulin.csv"

    if not (tp_path.exists() and g_path.exists() and i_path.exists()):
        return None

    tp_rows = [line.strip() for line in tp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    g_times = _parse_semicolon_floats(tp_rows[0].split(";")[1:])
    i_times = _parse_semicolon_floats(tp_rows[1].split(";")[1:])

    g_rows = []
    for line in g_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            g_rows.append(_parse_semicolon_floats(line.split(";")))
    i_rows = []
    for line in i_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            i_rows.append(_parse_semicolon_floats(line.split(";")))

    g_data = np.vstack(g_rows)
    i_data = np.vstack(i_rows)

    def summarize_group(mask):
        g_sel = g_data[mask]
        i_sel = i_data[mask]
        if g_sel.shape[0] == 0 or i_sel.shape[0] == 0:
            return None
        return {
            "n": int(g_sel.shape[0]),
            "g_q25": np.percentile(g_sel, 25, axis=0),
            "g_med": np.percentile(g_sel, 50, axis=0),
            "g_q75": np.percentile(g_sel, 75, axis=0),
            "i_q25": np.percentile(i_sel, 25, axis=0),
            "i_med": np.percentile(i_sel, 50, axis=0),
            "i_q75": np.percentile(i_sel, 75, axis=0),
        }

    # OGTT 2h (120-min) groups by glucose concentration (mmol/L).
    g120 = g_data[:, 4]
    threshold_masks = {
        "healthy": g120 < 7.8,
        "impaired": (g120 >= 7.8) & (g120 < 11.1),
        "t2d": g120 >= 11.1,
    }
    threshold_counts = {name: int(np.sum(mask)) for name, mask in threshold_masks.items()}

    # If thresholds produce sparse groups (common in mixed but non-diabetic datasets),
    # use balanced tertiles so all 3 reference shadows are visible and comparable.
    use_tertiles = min(threshold_counts.values()) < 3
    if use_tertiles:
        order = np.argsort(g120)
        buckets = np.array_split(order, 3)
        masks = {
            "healthy": np.isin(np.arange(len(g120)), buckets[0]),
            "impaired": np.isin(np.arange(len(g120)), buckets[1]),
            "t2d": np.isin(np.arange(len(g120)), buckets[2]),
        }
        grouping_method = "tertiles"
    else:
        masks = threshold_masks
        grouping_method = "thresholds"

    groups = {name: summarize_group(mask) for name, mask in masks.items()}

    # Fallback to global summary if any group is empty.
    global_summary = {
        "n": int(g_data.shape[0]),
        "g_q25": np.percentile(g_data, 25, axis=0),
        "g_med": np.percentile(g_data, 50, axis=0),
        "g_q75": np.percentile(g_data, 75, axis=0),
        "i_q25": np.percentile(i_data, 25, axis=0),
        "i_med": np.percentile(i_data, 50, axis=0),
        "i_q75": np.percentile(i_data, 75, axis=0),
    }
    for name in groups:
        if groups[name] is None:
            groups[name] = global_summary

    return {
        "g_t": g_times,
        "i_t": i_times,
        "groups": groups,
        "grouping_method": grouping_method,
    }


BASE = {
    "k1": 0.0105,
    "k2": 0.28,
    "k3": 6.07e-3,
    "k4": 2.35e-4,
    "k5": 0.0424,
    "k6": 2.2975,
    "k7": 1.15,
    "k8": 7.27,
    "k9": 3.83e-2,
    "k10": 2.84e-1,
    "tau_i": 31.0,
    "tau_d": 3.0,
    "beta": 1.0,
    "gren": 9.0,
    "egpb": 0.043,
    "km": 13.2,
    "f": 0.005551,
    "vg": 17.0 / 70.0,
    "c1": 0.1,
    "t_int": 30.0,
    "sigma": 1.4,
    "dmeal": 75.0e3,
    "bw": 70.0,
    "gb": 5.0,
    "ib": 10.0,
    "dt": 0.2,
    "tmax": 240.0,
}

PRESETS = {
    "healthy": {"k1": 0.0105, "k5": 0.0424, "k6": 2.2975},
    "impaired": {"k1": 0.0095, "k5": 0.0424 * 0.4, "k6": 2.2975 * 1.3},
    "t2d": {"k1": 0.0085, "k5": 0.0424 * 0.1, "k6": 2.2975 * 0.4},
}


REFERENCE = load_reference_shadows()


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "EDES PID Frontend"

app.layout = dbc.Container(
    [
        dcc.Store(id="active-group", data="healthy"),
        html.Div(id="active-group-badge", style={"textAlign": "center", "marginTop": "8px", "marginBottom": "4px"}),
        dbc.Row(
            dbc.Col(
                html.H3(
                    "EDES DDE Frontend: Parameter + PID Tuning",
                    className="text-center mt-3 mb-2",
                    style={"fontWeight": "700", "color": "#1f2d3d"},
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                html.P(
                    "Interactive layer for the Julia EDES model. Tune PID gains and physiologic parameters, then inspect glucose, insulin, and PID term contributions.",
                    className="text-center text-muted mb-3",
                )
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Patient Presets", style={"fontWeight": "700"}),
                                dbc.CardBody(
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button("Healthy", id="btn-healthy", color="success", outline=True, className="me-2"),
                                            dbc.Button("Impaired", id="btn-impaired", color="warning", outline=True, className="me-2"),
                                            dbc.Button("T2D", id="btn-t2d", color="danger", outline=True),
                                        ]
                                    )
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("PID Gains", style={"fontWeight": "700"}),
                                dbc.CardBody(
                                    [
                                        make_slider("k6-slider", "k6 (P gain)", 0.0, 12.0, 0.05, BASE["k6"], {0: "0", 2.3: "base", 12: "12"}),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Tunable Parameters (Only k1, k5, k6)", style={"fontWeight": "700"}),
                                dbc.CardBody(
                                    [
                                        make_slider("k1-slider", "k1 (meal absorption timescale)", 0.002, 0.04, 0.0005, BASE["k1"], {0.002: "0.002", 0.0105: "base", 0.04: "0.04"}),
                                        make_slider("k5-slider", "k5 (peripheral insulin sensitivity)", 0.0, 0.12, 0.001, BASE["k5"], {0: "0", 0.0424: "base", 0.12: "0.12"}),
                                        dcc.Checklist(
                                            id="shadow-toggle",
                                            options=[{"label": " Show reference shadow (population IQR + median)", "value": "on"}],
                                            value=["on"],
                                            style={"marginTop": "8px"},
                                        ),
                                        html.Div(
                                            "k2, k3, k4, k7, k8, k9, k10, tau_i, tau_d, beta and all physiology terms are fixed to Julia baseline.",
                                            style={"marginTop": "8px", "fontSize": "0.82rem", "color": "#6c757d"},
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dcc.Graph(id="glucose-graph", style={"height": "280px"}),
                                    dcc.Graph(id="insulin-graph", style={"height": "280px"}),
                                    dcc.Graph(id="pid-graph", style={"height": "280px"}),
                                ]
                            )
                        )
                    ],
                    md=8,
                ),
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    [
        Output("k1-slider", "value"),
        Output("k5-slider", "value"),
        Output("k6-slider", "value"),
        Output("active-group", "data"),
    ],
    [
        Input("btn-healthy", "n_clicks"),
        Input("btn-impaired", "n_clicks"),
        Input("btn-t2d", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def set_preset(_h, _i, _t):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "btn-impaired":
        group = "impaired"
    elif trigger == "btn-t2d":
        group = "t2d"
    else:
        group = "healthy"

    p = PRESETS[group]

    return p["k1"], p["k5"], p["k6"], group


@app.callback(
    Output("active-group-badge", "children"),
    Input("active-group", "data"),
)
def render_active_group_badge(active_group):
    labels = {
        "healthy": "Healthy",
        "impaired": "Impaired",
        "t2d": "T2D",
    }
    colors = {
        "healthy": "#198754",
        "impaired": "#b8860b",
        "t2d": "#b02a37",
    }
    g = active_group if active_group in labels else "healthy"
    method = "thresholds"
    if REFERENCE is not None:
        method = REFERENCE.get("grouping_method", "thresholds")
    method_label = "clinical thresholds" if method == "thresholds" else "data tertiles"
    return html.Span(
        f"Reference group: {labels[g]} ({method_label})",
        style={
            "display": "inline-block",
            "padding": "4px 10px",
            "borderRadius": "999px",
            "fontSize": "0.82rem",
            "fontWeight": "700",
            "backgroundColor": "#f8f9fa",
            "color": colors[g],
            "border": f"1px solid {colors[g]}",
        },
    )


@app.callback(
    [Output("glucose-graph", "figure"), Output("insulin-graph", "figure"), Output("pid-graph", "figure")],
    [
        Input("k1-slider", "value"),
        Input("k5-slider", "value"),
        Input("k6-slider", "value"),
        Input("shadow-toggle", "value"),
        Input("active-group", "data"),
    ],
)
def update_graphs(k1, k5, k6, shadow_toggle, active_group):
    p = BASE.copy()
    p.update({"k1": k1, "k5": k5, "k6": k6})

    result = simulate_edes_dde(p)
    t = result["t"]
    g = result["gpl"]
    i = result["ipl"]

    fig_g = go.Figure()
    show_shadow = REFERENCE is not None and "on" in shadow_toggle
    if show_shadow:
        ref = REFERENCE["groups"].get(active_group, REFERENCE["groups"]["healthy"])
        fig_g.add_trace(go.Scatter(
            x=REFERENCE["g_t"], y=ref["g_q75"], mode="lines",
            line={"width": 0}, showlegend=False, hoverinfo="skip", name="Glucose IQR"))
        fig_g.add_trace(go.Scatter(
            x=REFERENCE["g_t"], y=ref["g_q25"], mode="lines",
            line={"width": 0}, fill="tonexty", fillcolor="rgba(33, 102, 172, 0.18)",
            name=f"Glucose reference IQR ({active_group}, n={ref['n']})", hoverinfo="skip"))
        fig_g.add_trace(go.Scatter(
            x=REFERENCE["g_t"], y=ref["g_med"], mode="lines",
            line={"width": 2, "color": "rgba(33, 102, 172, 0.75)", "dash": "dash"}, name=f"Glucose reference median ({active_group})"))
    fig_g.add_trace(go.Scatter(x=t, y=g, mode="lines", line={"width": 3, "color": "#2166ac"}, name="Plasma glucose"))
    fig_g.add_hline(y=result["gb"], line_dash="dash", line_color="#888", annotation_text="Basal glucose")
    fig_g.add_hline(y=7.8, line_dash="dot", line_color="#d73027", annotation_text="IGT threshold")
    fig_g.add_vline(x=120.0, line_dash="dash", line_color="#444", annotation_text="120 min", annotation_position="top")
    fig_g.update_layout(
        title="Glucose Response",
        xaxis_title="Time (min)",
        yaxis_title="Glucose (mmol/L)",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 40, "b": 30},
    )

    fig_i = go.Figure()
    if show_shadow:
        ref = REFERENCE["groups"].get(active_group, REFERENCE["groups"]["healthy"])
        fig_i.add_trace(go.Scatter(
            x=REFERENCE["i_t"], y=ref["i_q75"], mode="lines",
            line={"width": 0}, showlegend=False, hoverinfo="skip", name="Insulin IQR"))
        fig_i.add_trace(go.Scatter(
            x=REFERENCE["i_t"], y=ref["i_q25"], mode="lines",
            line={"width": 0}, fill="tonexty", fillcolor="rgba(230, 130, 20, 0.18)",
            name=f"Insulin reference IQR ({active_group}, n={ref['n']})", hoverinfo="skip"))
        fig_i.add_trace(go.Scatter(
            x=REFERENCE["i_t"], y=ref["i_med"], mode="lines",
            line={"width": 2, "color": "rgba(230, 130, 20, 0.75)", "dash": "dash"}, name=f"Insulin reference median ({active_group})"))
    fig_i.add_trace(go.Scatter(x=t, y=i, mode="lines", line={"width": 3, "color": "#e08214"}, name="Plasma insulin"))
    fig_i.add_hline(y=result["ib"], line_dash="dash", line_color="#888", annotation_text="Basal insulin")
    fig_i.add_vline(x=120.0, line_dash="dash", line_color="#444", annotation_text="120 min", annotation_position="top")
    fig_i.update_layout(
        title="Insulin Response",
        xaxis_title="Time (min)",
        yaxis_title="Insulin (mU/L)",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 40, "b": 30},
    )

    fig_pid = go.Figure()
    fig_pid.add_trace(go.Scatter(x=t, y=result["pid_p"], mode="lines", line={"width": 2.5, "color": "#1b9e77"}, name="P term"))
    fig_pid.add_trace(go.Scatter(x=t, y=result["pid_i"], mode="lines", line={"width": 2.5, "color": "#7570b3"}, name="I term"))
    fig_pid.add_trace(go.Scatter(x=t, y=result["pid_d"], mode="lines", line={"width": 2.5, "color": "#d95f02"}, name="D term"))
    fig_pid.add_vline(x=120.0, line_dash="dash", line_color="#444", annotation_text="120 min", annotation_position="top")
    fig_pid.update_layout(
        title="PID Contribution to Pancreatic Output",
        xaxis_title="Time (min)",
        yaxis_title="Contribution (model units)",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 40, "b": 30},
        legend={"orientation": "h", "y": 1.12, "x": 0.0},
    )

    return fig_g, fig_i, fig_pid


if __name__ == "__main__":
    app.run(debug=False, port=8050)
