import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="Hand 3D Visualizer", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: white; }
    .block-container { 
        padding-top: 0rem !important; 
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 0.5rem; }
    div[data-testid="stVerticalBlock"] { gap: 0rem; }
    header { visibility: hidden; height: 0; }
    footer { visibility: hidden; height: 0; }
</style>
""", unsafe_allow_html=True)

FINGER_PREFIXES = {
    "Thumb": "TH",
    "Pinky": "F1",
    "Ring": "F2",
    "Middle": "F3",
    "Index": "F4",
}
FINGER_JOINTS = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]

FINGER_COLORS = {
    "Thumb": "#e63946",
    "Index": "#457b9d",
    "Middle": "#e9c46a",
    "Ring": "#f4a261",
    "Pinky": "#2a9d8f",
}
PALM_COLOR = "#a8dadc"
BG_COLOR = "#0d1117"

# --- LOGIC ---
def build_skeleton():
    bones = []
    for finger in FINGER_PREFIXES:
        color = FINGER_COLORS[finger]
        chain = [f"{finger}_{s}" for s in FINGER_JOINTS]
        knu1b, knu1a, knu2a, knu3a = chain
        if finger == "Thumb":
            bones.append(("Palm", knu1a, color, True))
            bones.append((knu1a, knu2a, color, False))
            bones.append((knu2a, knu3a, color, False))
        else:
            bones.append(("Palm", knu1b, color, False))
            bones.append((knu1b, knu1a, color, False))
            bones.append((knu1a, knu2a, color, False))
            bones.append((knu2a, knu3a, color, False))
    return bones

BONES = build_skeleton()
HIDDEN_JOINTS = {f"{f}_KNU1_B" for f in FINGER_PREFIXES if f == "Thumb"}

def extract_joints(row):
    joints = {"Palm": (row["PALM_POSITION_X"], row["PALM_POSITION_Y"], row["PALM_POSITION_Z"])}
    for finger, prefix in FINGER_PREFIXES.items():
        for suffix in FINGER_JOINTS:
            joints[f"{finger}_{suffix}"] = (
                row[f"{prefix}_{suffix}_X"],
                row[f"{prefix}_{suffix}_Y"],
                row[f"{prefix}_{suffix}_Z"]
            )
    return joints

def get_equal_axis_ranges(joints):
    xs = [v[0] for v in joints.values()]
    ys = [v[1] for v in joints.values()]
    zs = [v[2] for v in joints.values()]
    max_range = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
    ) / 2.0
    mid_x = (max(xs) + min(xs)) / 2
    mid_y = (max(ys) + min(ys)) / 2
    mid_z = (max(zs) + min(zs)) / 2
    return (
        [mid_x - max_range, mid_x + max_range],
        [mid_y - max_range, mid_y + max_range],
        [mid_z - max_range, mid_z + max_range],
    )

# --- DATA ---
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_normalised_data_NOdata18.csv")

try:
    df = load_data()
except Exception:
    st.error("CSV not found in /data folder.")
    st.stop()

# --- STATE ---
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# Sidebar Navigation
st.sidebar.title("Navigation")
col1, col2 = st.sidebar.columns(2)
if col1.button("◀ Prev"):
    st.session_state.idx = max(0, st.session_state.idx - 1)
if col2.button("Next ▶"):
    st.session_state.idx = min(len(df) - 1, st.session_state.idx + 1)

st.session_state.idx = st.sidebar.slider("Index", 1, len(df), st.session_state.idx+1)-1

# Search
st.sidebar.markdown("---")
s_vid = st.sidebar.text_input("Video ID")
s_frm = st.sidebar.text_input("Frame ID")
if st.sidebar.button("Search"):
    v = f"data_{s_vid}" if s_vid.isdigit() else s_vid
    f = f"{s_frm}_joints" if s_frm.isdigit() else s_frm
    m = pd.Series([True]*len(df))
    if s_vid: m &= (df["video_id"] == v)
    if s_frm: m &= (df["frame_id"] == f)
    res = df[m]
    if not res.empty: st.session_state.idx = int(res.index[0])

# --- RENDER ---
row = df.iloc[st.session_state.idx]
joints = extract_joints(row)
x_range, y_range, z_range = get_equal_axis_ranges(joints)
fig = go.Figure()

# Skeleton Lines
for a, b, color, dotted in BONES:
    fig.add_trace(go.Scatter3d(
        x=[joints[a][0], joints[b][0]],
        y=[joints[a][1], joints[b][1]],
        z=[joints[a][2], joints[b][2]],
        mode='lines',
        line=dict(color=color, width=7, dash='dash' if dotted else 'solid'),
        showlegend=False, hoverinfo='none'
    ))

# Joint Markers
for name, pos in joints.items():
    if name in HIDDEN_JOINTS: continue
    color = PALM_COLOR if name == "Palm" else FINGER_COLORS.get(name.split('_')[0])
    is_tip = "KNU3_A" in name

    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text' if is_tip else 'markers',
        marker=dict(color=color, size=8 if name == "Palm" else 5, line=dict(color='white', width=1)),
        text=name.split('_')[0] if is_tip else None,
        textposition="top center",
        name=name, showlegend=False
    ))

fig.update_layout(
    title=f"Row: {st.session_state.idx + 1} | {row['video_id']} | {row['frame_id']}",
    template="plotly_dark",
    height=720,
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor=BG_COLOR,
    scene=dict(
        aspectmode='cube',
        xaxis=dict(title='X', gridcolor='#333', range=x_range),
        yaxis=dict(title='Y', gridcolor='#333', range=y_range),
        zaxis=dict(title='Z', gridcolor='#333', range=z_range),
    ),
    uirevision='constant'
)

st.plotly_chart(fig, use_container_width=True)