import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Hand Gesture Viewer", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: white; }
    .block-container { padding-top: 1rem; }
    div[data-testid="stMetricValue"] { color: #a8dadc; font-size: 1.1rem; }
    div[data-testid="stMetricLabel"] { color: #7a8a9a; }
    .pred-box {
        background: #1e2532; border-radius: 8px; padding: 10px 18px;
        text-align: center; margin-bottom: 10px;
    }
    .pred-label { color: #7a8a9a; font-size: 0.8rem; }
    .pred-value { color: #a8dadc; font-size: 1.4rem; font-weight: bold; }
    label, .stTextInput label { color: #7a8a9a !important; }

    /* Tighten sidebar spacing */
    section[data-testid="stSidebar"] .block-container { padding-top: 0.5rem; }
    section[data-testid="stSidebar"] h2 { margin-bottom: 0.3rem; margin-top: 0; }
    section[data-testid="stSidebar"] .stSlider { margin-top: 0; padding-top: 0; }
    section[data-testid="stSidebar"] .stTextInput { margin-bottom: 0; padding-bottom: 0; }
    section[data-testid="stSidebar"] .stButton { margin-top: 0; margin-bottom: 0; }
    section[data-testid="stSidebar"] hr { margin-top: 0.4rem; margin-bottom: 0.4rem; }
    section[data-testid="stSidebar"] p { margin-bottom: 0.2rem; }
    div[data-testid="stVerticalBlock"] { gap: 0rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
BG_COLOR = "#0d1117"
PANEL_COLOR = "#0d1117"

FINGER_PREFIXES = {
    "Thumb":  "TH",
    "Pinky":  "F1",
    "Ring":   "F2",
    "Middle": "F3",
    "Index":  "F4",
}
FINGER_JOINTS = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]
FINGER_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#f4a261", "#457b9d"]
FINGER_NAMES  = list(FINGER_PREFIXES.keys())

VIEWS = {
    "Front": (1, 2),
    "Side":  (0, 2),
    "Top":   (1, 0),
}
AXIS_LABELS = {
    "Front": ("Y", "Z"),
    "Side":  ("X", "Z"),
    "Top":   ("Z", "X"),
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def get_coord_cols():
    cols = ["PALM_POSITION_X", "PALM_POSITION_Y", "PALM_POSITION_Z"]
    for prefix in FINGER_PREFIXES.values():
        for suffix in FINGER_JOINTS:
            cols += [f"{prefix}_{suffix}_X", f"{prefix}_{suffix}_Y", f"{prefix}_{suffix}_Z"]
    return cols

def row_to_points(row):
    cols = get_coord_cols()
    coords = row[cols].to_numpy(dtype=float)
    return coords.reshape(-1, 3)

def center_hand(points):
    return points - points[0]

# ---------------------------------------------------------------------------
# Plotly 2D projection builder
# ---------------------------------------------------------------------------
def build_hand_traces(pts, axes, view_name, invert_x=False):
    """Build plotly traces for one 2D projection of the hand."""
    i, j = axes
    traces = []
    base = 1

    # Palm point
    traces.append(go.Scatter(
        x=[pts[0, i]], y=[pts[0, j]],
        mode="markers",
        marker=dict(color="tomato", size=10, line=dict(color="white", width=1)),
        name="Palm", showlegend=False, hoverinfo="skip"
    ))

    for f in range(5):
        color = FINGER_COLORS[f]
        name  = FINGER_NAMES[f]
        finger_pts = pts[base: base + 4]

        if f == 0:  # Thumb — skip KNU1_B, connect palm to KNU1_A
            visible = finger_pts[1:]
            # Dashed palm connector
            traces.append(go.Scatter(
                x=[pts[0, i], visible[0, i]],
                y=[pts[0, j], visible[0, j]],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                opacity=0.55, showlegend=False, hoverinfo="skip"
            ))
            # Solid finger chain
            traces.append(go.Scatter(
                x=visible[:, i], y=visible[:, j],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(color=color, size=6, line=dict(color="white", width=0.8)),
                name=name, showlegend=(view_name == "Front"),
                hovertemplate=f"{name}<extra></extra>"
            ))
        else:
            # Dashed palm connector
            traces.append(go.Scatter(
                x=[pts[0, i], finger_pts[0, i]],
                y=[pts[0, j], finger_pts[0, j]],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                opacity=0.55, showlegend=False, hoverinfo="skip"
            ))
            # Solid finger chain
            traces.append(go.Scatter(
                x=finger_pts[:, i], y=finger_pts[:, j],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(color=color, size=6, line=dict(color="white", width=0.8)),
                name=name, showlegend=(view_name == "Front"),
                hovertemplate=f"{name}<extra></extra>"
            ))
        base += 4

    return traces

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    preds = pd.read_csv("outputs/predictions_open_set.csv")
    data  = pd.read_csv("data/normalised_hand_data_DATA18REMOVED.csv")
    df = pd.merge(preds, data, on=["video_id", "frame_id"])
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "idx" not in st.session_state:
    st.session_state.idx = 0

# ---------------------------------------------------------------------------
# Sidebar — navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Navigation")

c1, c2 = st.sidebar.columns(2)
if c1.button("◀ Prev", use_container_width=True):
    st.session_state.idx = max(0, st.session_state.idx - 1)
if c2.button("Next ▶", use_container_width=True):
    st.session_state.idx = min(len(df) - 1, st.session_state.idx + 1)

st.session_state.idx = st.sidebar.slider(
    "Row index", 0, len(df) - 1, st.session_state.idx
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Search**")
s_vid = st.sidebar.text_input("Video ID (number only, e.g. 4)")
s_frm = st.sidebar.text_input("Frame ID (number only, e.g. 948)")

if st.sidebar.button("Search", use_container_width=True):
    v = f"data_{s_vid}" if s_vid.isdigit() else s_vid
    f = f"{s_frm}_joints" if s_frm.isdigit() else s_frm
    mask = pd.Series([True] * len(df))
    if s_vid: mask &= df["video_id"] == v
    if s_frm: mask &= df["frame_id"] == f
    res = df[mask]
    if not res.empty:
        st.session_state.idx = int(res.index[0])
    else:
        st.sidebar.error("No match found.")

# ---------------------------------------------------------------------------
# Current row
# ---------------------------------------------------------------------------
row   = df.iloc[st.session_state.idx]
pts   = center_hand(row_to_points(row))
pred  = row.get("ensemble_open_set", "N/A")
vid   = row["video_id"]
frm   = row["frame_id"]

# ---------------------------------------------------------------------------
# Header info
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="pred-box">
    <div class="pred-label">Predicted Gesture</div>
    <div class="pred-value">{pred}</div>
</div>
""", unsafe_allow_html=True)

info_cols = st.columns(4)
info_cols[0].metric("Video", vid.replace("data_", ""))
info_cols[1].metric("Frame", frm.replace("_joints", ""))
info_cols[2].metric("Row", f"{st.session_state.idx + 1} / {len(df)}")

# Confidence columns if present
conf_cols = [c for c in df.columns if c.endswith("_confidence_pct")]
if conf_cols and pred != "UNKNOWN":
    avg_conf = row[conf_cols].mean()
    info_cols[3].metric("Avg Confidence", f"{avg_conf:.1f}%")

# Per-model confidence badges
if conf_cols and pred != "UNKNOWN":
    model_labels = {
        "logistic_regression_confidence_pct": "Logistic Reg.",
        "mlp_confidence_pct": "MLP",
        "random_forest_confidence_pct": "Random Forest",
    }
    badge_cols = st.columns(len(conf_cols))
    for col_el, col_name in zip(badge_cols, conf_cols):
        label = model_labels.get(col_name, col_name)
        val   = row[col_name]
        col_el.metric(label, f"{val:.1f}%" if not np.isnan(val) else "—")

st.markdown("---")

# ---------------------------------------------------------------------------
# Three 2D projection panels
# ---------------------------------------------------------------------------
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=list(VIEWS.keys()),
    horizontal_spacing=0.05
)

for col_idx, (view_name, axes) in enumerate(VIEWS.items(), start=1):
    invert = (view_name == "Top")
    traces = build_hand_traces(pts, axes, view_name, invert_x=invert)
    for trace in traces:
        fig.add_trace(trace, row=1, col=col_idx)

    xl, yl = AXIS_LABELS[view_name]
    fig.update_xaxes(
        title_text=xl,
        scaleanchor=f"y{'' if col_idx == 1 else col_idx}",
        scaleratio=1,
        showgrid=False, zeroline=False,
        color="#7a8a9a",
        autorange="reversed" if invert else True,
        row=1, col=col_idx
    )
    fig.update_yaxes(
        title_text=yl,
        showgrid=False, zeroline=False,
        color="#7a8a9a",
        row=1, col=col_idx
    )

fig.update_layout(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=PANEL_COLOR,
    font=dict(color="white"),
    height=480,
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=-0.25,
        xanchor="center", x=0.5,
        font=dict(color="white"),
        bgcolor="rgba(0,0,0,0)"
    ),
    uirevision="constant",
)

# Style subplot title colors
for annotation in fig.layout.annotations:
    annotation.font.color = "white"
    annotation.font.size  = 13

st.plotly_chart(fig, use_container_width=True)