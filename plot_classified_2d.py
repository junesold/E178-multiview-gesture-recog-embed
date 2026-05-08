import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#set up the page configuration
st.set_page_config(page_title="Hand Gesture Viewer", layout="wide")

#set the stylization (in markdown) for the header of the figure
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
</style>
""", unsafe_allow_html=True)
# ^^ decides all the stylization

#Joint, finger, axis, and color definitions
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

#establish coordinate columns
def get_coord_cols():
    cols = ["PALM_POSITION_X", "PALM_POSITION_Y", "PALM_POSITION_Z"]
    for prefix in FINGER_PREFIXES.values():
        for suffix in FINGER_JOINTS:
            cols += [f"{prefix}_{suffix}_X", f"{prefix}_{suffix}_Y", f"{prefix}_{suffix}_Z"]
    return cols

#process row data into array for processing ease
def row_to_points(row):
    cols = get_coord_cols()
    coords = row[cols].to_numpy(dtype=float)
    return coords.reshape(-1, 3)

def center_hand(points):
    return points - points[0]

#Use plotly for the 2d graphical representation
def build_hand_traces(pts, axes, view_name, invert_x=False):
    i, j = axes
    traces = []
    base = 1

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

        if f == 0:
            visible = finger_pts[1:]
            traces.append(go.Scatter(
                x=[pts[0, i], visible[0, i]],
                y=[pts[0, j], visible[0, j]],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                opacity=0.55, showlegend=False, hoverinfo="skip"
            ))
            traces.append(go.Scatter(
                x=visible[:, i], y=visible[:, j],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(color=color, size=6, line=dict(color="white", width=0.8)),
                name=name, showlegend=(view_name == "Front"),
                hovertemplate=f"{name}<extra></extra>"
            ))
        else:
            traces.append(go.Scatter(
                x=[pts[0, i], finger_pts[0, i]],
                y=[pts[0, j], finger_pts[0, j]],
                mode="lines",
                line=dict(color=color, width=2, dash="solid"),
                opacity=0.55, showlegend=False, hoverinfo="skip"
            ))
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

# load the data
@st.cache_data
def load_data():
    preds = pd.read_csv("outputs/predictions_open_set.csv")
    data  = pd.read_csv("data/cleaned_normalised_data_NOdata18.csv")

    # normalize and ensure clean data (.strip to take care of possible floating-point errors)
    for col in ["video_id", "frame_id"]:
        preds[col] = preds[col].astype(str).str.strip()
        data[col]  = data[col].astype(str).str.strip()

    df = pd.merge(preds, data, on=["video_id", "frame_id"], how="inner")
    df = df.reset_index(drop=True)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()


if "idx" not in st.session_state:
    st.session_state.idx = 0

# build the sidebar
st.sidebar.markdown("## Navigation")
st.sidebar.markdown("**Filter by Gesture**")

#make the dropdown selection (using original names to avoid confusion)
all_gestures = sorted(df["ensemble_open_set"].unique().tolist())

selected = st.sidebar.multiselect(
    "Show only:",
    options=all_gestures,
    format_func=lambda g: g.replace("_", " ").title(),
    default=[]
)

#use the selection to create data set
if selected:
    df_filtered = df[df["ensemble_open_set"].isin(selected)].reset_index(drop=True)
else:
    df_filtered = df.reset_index(drop=True)

if len(df_filtered) == 0:
    st.sidebar.error("No rows match the selected filter.")
    st.stop()

st.sidebar.markdown("---")

max_idx = max(0, len(df_filtered) - 1)
st.session_state.idx = min(st.session_state.idx, max_idx)

#build the previous/next buttons
c1, c2 = st.sidebar.columns(2)
if c1.button("◀ Prev", use_container_width=True):
    st.session_state.idx = max(0, st.session_state.idx - 1)
if c2.button("Next ▶", use_container_width=True):
    st.session_state.idx = min(max_idx, st.session_state.idx + 1)

#make the slider (size adjusted to the selection)
st.session_state.idx = st.sidebar.slider(
    "Row index",
    min_value=0,
    max_value=max_idx,
    value=st.session_state.idx
)

#create the search function
st.sidebar.markdown("---")
st.sidebar.markdown("**Search**")
s_vid = st.sidebar.text_input("Video ID (number only, e.g. 4)")
s_frm = st.sidebar.text_input("Frame ID (number only, e.g. 948)")

if st.sidebar.button("Search", use_container_width=True):
    v = f"data_{s_vid}" if s_vid.isdigit() else s_vid
    f = f"{s_frm}_joints" if s_frm.isdigit() else s_frm
    mask = pd.Series([True] * len(df_filtered))
    if s_vid: mask &= df_filtered["video_id"] == v
    if s_frm: mask &= df_filtered["frame_id"] == f
    res = df_filtered[mask]
    if not res.empty:
        st.session_state.idx = int(res.index[0])
    else:
        st.sidebar.error("No match found.")

# define current state
row  = df_filtered.iloc[st.session_state.idx]
pts  = center_hand(row_to_points(row))
pred = row.get("ensemble_open_set", "N/A")
vid  = row["video_id"]
frm  = row["frame_id"]

# more header stuff
st.markdown(f"""
<div class="pred-box">
    <div class="pred-label">Predicted Gesture</div>
    <div class="pred-value">{pred.replace("_", " ") if pred != "N/A" else pred}</div>
</div>
""", unsafe_allow_html=True)

info_cols = st.columns(4)
info_cols[0].metric("Video", vid.replace("data_", ""))
info_cols[1].metric("Frame", frm.replace("_joints", ""))
info_cols[2].metric("Row", f"{st.session_state.idx + 1} / {len(df_filtered)}")

conf_cols = [c for c in df.columns if c.endswith("_confidence_pct")]
if conf_cols and pred != "UNKNOWN":
    avg_conf = row[conf_cols].mean()
    info_cols[3].metric("Avg Confidence", f"{avg_conf:.1f}%")

#establish the different models, respective scores, and conclusion
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

# make the 2D plots
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
        showline=True, linecolor="#444e5e", linewidth=1,
        color="#7a8a9a",
        autorange="reversed" if invert else True,
        row=1, col=col_idx
    )
    fig.update_yaxes(
        title_text=yl,
        showgrid=False, zeroline=False,
        showline=True, linecolor="#444e5e", linewidth=1,
        color="#7a8a9a",
        row=1, col=col_idx
    )

fig.update_layout(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=PANEL_COLOR,
    font=dict(color="white"),
    height=480,
    dragmode=False,
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

for annotation in fig.layout.annotations:
    annotation.font.color = "white"
    annotation.font.size  = 13

#call the plot!
st.plotly_chart(fig, use_container_width=True, config={"modeBarButtonsToRemove": ["zoom", "select2d", "lasso2d"]})