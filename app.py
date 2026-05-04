import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="Hand 3D Visualizer", layout="wide")

# Joint and Color Definitions
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
            bones.append(("Palm", knu1b, color, True))
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

# --- DATA & BOUNDS CALCULATION ---
@st.cache_data
def load_and_bound_data():
    df = pd.read_csv("data/normalised_hand_data_DATA18REMOVED.csv")
    
    # Identify X, Y, Z columns to find global min/max
    x_cols = [c for c in df.columns if c.endswith('_X')]
    y_cols = [c for c in df.columns if c.endswith('_Y')]
    z_cols = [c for c in df.columns if c.endswith('_Z')]
    
    # Calculate global range for fixed axis scaling
    bounds = {
        'x': (df[x_cols].values.min(), df[x_cols].values.max()),
        'y': (df[y_cols].values.min(), df[y_cols].values.max()),
        'z': (df[z_cols].values.min(), df[z_cols].values.max())
    }
    return df, bounds

try:
    df, bounds = load_and_bound_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- STATE MANAGEMENT ---
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# Sidebar Navigation
st.sidebar.title("Navigation")
col1, col2 = st.sidebar.columns(2)
if col1.button("◀ Prev"):
    st.session_state.idx = max(0, st.session_state.idx - 1)
if col2.button("Next ▶"):
    st.session_state.idx = min(len(df) - 1, st.session_state.idx + 1)

st.session_state.idx = st.sidebar.slider("Select Row", 0, len(df)-1, st.session_state.idx)

# Search
st.sidebar.markdown("---")
s_vid = st.sidebar.text_input("Search Video ID")
s_frm = st.sidebar.text_input("Search Frame ID")
if st.sidebar.button("Go"):
    v = f"data_{s_vid}" if s_vid.isdigit() else s_vid
    f = f"{s_frm}_joints" if s_frm.isdigit() else s_frm
    m = pd.Series([True]*len(df))
    if s_vid: m &= (df["video_id"] == v)
    if s_frm: m &= (df["frame_id"] == f)
    res = df[m]
    if not res.empty:
        st.session_state.idx = int(res.index[0])
    else:
        st.sidebar.error("No match found.")

# --- RENDERING ---
row = df.iloc[st.session_state.idx]
joints = extract_joints(row)
fig = go.Figure()

# Skeleton Bones (Lines)
for a, b, color, dotted in BONES:
    fig.add_trace(go.Scatter3d(
        x=[joints[a][0], joints[b][0]],
        y=[joints[a][1], joints[b][1]],
        z=[joints[a][2], joints[b][2]],
        mode='lines',
        line=dict(color=color, width=7, dash='dash' if dotted else 'solid'),
        showlegend=False, hoverinfo='none'
    ))

# Joint Markers (Points)
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

# FIXED AXIS SCALING
# Calculate ratio to keep physical proportions correct
x_range = bounds['x'][1] - bounds['x'][0]
y_range = bounds['y'][1] - bounds['y'][0]
z_range = bounds['z'][1] - bounds['z'][0]

fig.update_layout(
    title=f"Video: {row['video_id']} | Frame: {row['frame_id']} | Row: {st.session_state.idx}",
    template="plotly_dark",
    height=800,
    paper_bgcolor=BG_COLOR,
    scene=dict(
        aspectmode='manual',
        aspectratio=dict(x=1, y=y_range/x_range, z=z_range/x_range),
        xaxis=dict(title='X', range=bounds['x'], gridcolor='#333', zeroline=False),
        yaxis=dict(title='Y', range=bounds['y'], gridcolor='#333', zeroline=False),
        zaxis=dict(title='Z', range=bounds['z'], gridcolor='#333', zeroline=False),
    ),
    uirevision='constant' # Keeps your custom rotation when clicking Next/Prev
)

st.plotly_chart(fig, use_container_width=True)