import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="Hand 3D Visualizer", layout="wide")

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
    try:
        joints = {"Palm": (row["PALM_POSITION_X"], row["PALM_POSITION_Y"], row["PALM_POSITION_Z"])}
        for finger, prefix in FINGER_PREFIXES.items():
            for suffix in FINGER_JOINTS:
                joints[f"{finger}_{suffix}"] = (
                    row[f"{prefix}_{suffix}_X"],
                    row[f"{prefix}_{suffix}_Y"],
                    row[f"{prefix}_{suffix}_Z"]
                )
        return joints
    except KeyError as e:
        st.error(f"Missing column in CSV: {e}")
        return None

# --- DATA & BOUNDS CALCULATION ---
@st.cache_data
def load_and_bound_data():
    df = pd.read_csv("data/normalised_hand_data_DATA18REMOVED.csv")
    
    # Identify X, Y, Z columns
    x_cols = [c for c in df.columns if c.endswith('_X')]
    y_cols = [c for c in df.columns if c.endswith('_Y')]
    z_cols = [c for c in df.columns if c.endswith('_Z')]
    
    # Use nanmin/nanmax to ignore any NaN values in the data
    # This prevents the "np.float64(nan)" error
    bounds = {
        'x': (float(np.nanmin(df[x_cols])), float(np.nanmax(df[x_cols]))),
        'y': (float(np.nanmin(df[y_cols])), float(np.nanmax(df[y_cols]))),
        'z': (float(np.nanmin(df[z_cols])), float(np.nanmax(df[z_cols])))
    }
    return df, bounds

try:
    df, bounds = load_and_bound_data()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# --- STATE MANAGEMENT ---
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# Sidebar
st.sidebar.title("Navigation")
col1, col2 = st.sidebar.columns(2)
if col1.button("◀ Prev"):
    st.session_state.idx = max(0, st.session_state.idx - 1)
if col2.button("Next ▶"):
    st.session_state.idx = min(len(df) - 1, st.session_state.idx + 1)

st.session_state.idx = st.sidebar.slider("Select Row", 0, len(df)-1, st.session_state.idx)

# Search Logic
st.sidebar.markdown("---")
s_vid = st.sidebar.text_input("Search Video ID")
s_frm = st.sidebar.text_input("Search Frame ID")
if st.sidebar.button("Go"):
    v = f"data_{s_vid}" if s_vid.isdigit() else s_vid
    f = f"{s_frm}_joints" if s_frm.isdigit() else s_frm
    m = pd.Series([True]*len(df))
    if s_vid: m &= (df["video_id"].astype(str) == v)
    if s_frm: m &= (df["frame_id"].astype(str) == f)
    res = df[m]
    if not res.empty:
        st.session_state.idx = int(res.index[0])
    else:
        st.sidebar.error("No match found.")

# --- RENDERING ---
row = df.iloc[st.session_state.idx]
joints = extract_joints(row)

if joints:
    fig = go.Figure()

    # Skeleton Bones
    for a, b, color, dotted in BONES:
        # Check for NaNs in specific joints to avoid drawing broken lines
        if any(np.isnan(joints[a])) or any(np.isnan(joints[b])): continue
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
        if name in HIDDEN_JOINTS or any(np.isnan(pos)): continue
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
    x_range = bounds['x'][1] - bounds['x'][0]
    y_range = bounds['y'][1] - bounds['y'][0]
    z_range = bounds['z'][1] - bounds['z'][0]
    
    # Avoid division by zero if x_range is 0
    divisor = x_range if x_range != 0 else 1.0

    fig.update_layout(
        title=f"Video: {row['video_id']} | Frame: {row['frame_id']} | Row: {st.session_state.idx}",
        template="plotly_dark",
        height=800,
        paper_bgcolor=BG_COLOR,
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=y_range/divisor, z=z_range/divisor),
            xaxis=dict(title='X', range=bounds['x'], gridcolor='#333'),
            yaxis=dict(title='Y', range=bounds['y'], gridcolor='#333'),
            zaxis=dict(title='Z', range=bounds['z'], gridcolor='#333'),
        ),
        uirevision='constant'
    )

    st.plotly_chart(fig, width='stretch')