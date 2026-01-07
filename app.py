import streamlit as st
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from shapely.geometry import Polygon
from collections import defaultdict

st.set_page_config(page_title="MoReVis Extension: Gallery View", layout="wide")

@st.cache_data
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    t_col = 'timestep' if 'timestep' in df.columns else 'frame'
    id_col = 'object'
    
    frames = {}
    for t in sorted(df[t_col].unique()):
        frame_df = df[df[t_col] == t]
        frame_objects = {}
        for _, row in frame_df.iterrows():
            try:
                pts = ast.literal_eval(row['points'])
                if len(pts) >= 3:
                    poly = Polygon(pts)
                    frame_objects[str(row[id_col])] = {
                        'poly': poly,
                        'y': row['ycenter'],
                        'area': row['area'],
                        'id': str(row[id_col])
                    }
            except:
                continue
        frames[t] = frame_objects
    return frames

class MoReVisExtension:
    def __init__(self, iou_threshold=0.4):
        self.iou_threshold = iou_threshold

    def get_iou(self, p1, p2):
        if not p1.intersects(p2): return 0
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        return inter / union if union > 0 else 0

    def analyze(self, frames):
        events = {'splits': [], 'merges': [], 'collisions': []}
        timesteps_list = sorted(frames.keys())
        for i in range(len(timesteps_list)):
            t = timesteps_list[i]
            objs = frames[t]
            ids = list(objs.keys())
            for idx1 in range(len(ids)):
                for idx2 in range(idx1 + 1, len(ids)):
                    if objs[ids[idx1]]['poly'].intersects(objs[ids[idx2]]['poly']):
                        events['collisions'].append({'t': t, 'objs': (ids[idx1], ids[idx2])})
            if i < len(timesteps_list) - 1:
                t_next = timesteps_list[i+1]
                mapping = defaultdict(list)
                rev_mapping = defaultdict(list)
                for p_id, p_data in objs.items():
                    for n_id, n_data in frames[t_next].items():
                        if p_data['poly'].intersects(n_data['poly']):
                            if self.get_iou(p_data['poly'], n_data['poly']) > self.iou_threshold:
                                mapping[p_id].append(n_id)
                                rev_mapping[n_id].append(p_id)
                for p_id, children in mapping.items():
                    if len(children) > 1: events['splits'].append({'t': t, 'p': p_id, 'c': children})
                for n_id, parents in rev_mapping.items():
                    if len(parents) > 1: events['merges'].append({'t': t+1, 'c': n_id, 'p': parents})
        return events

# --- SIDEBAR ---
st.sidebar.header("Data Selection")
dataset_choice = st.sidebar.selectbox("Choose Dataset", 
    ["data/processed/wildtrack.csv", "data/processed/hurdat.csv", "data/processed/motivating.csv"])
iou_val = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45)

st.sidebar.header("Gallery Settings")
chunk_size = st.sidebar.slider("Frames per Segment", 20, 100, 50)
v_scale = st.sidebar.slider("Vertical Spreading (Higher = Thinner)", 1.0, 10.0, 4.0)
show_collisions = st.sidebar.checkbox("Show Collisions (Red X)", value=True)

# --- LOAD & ANALYZE ---
full_data = load_dataset(dataset_choice)
all_timesteps = sorted(full_data.keys())
analyzer = MoReVisExtension(iou_threshold=iou_val)
events = analyzer.analyze(full_data)

# --- DISPLAY ---
st.title("MoReVis Extension: Sequential Gallery Summary")
st.markdown("""
<div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <strong>Navigation Legend:</strong> 
    <span style="color: lime;">▲</span> Split | 
    <span style="color: cyan;">▼</span> Merge | 
    <span style="color: red;">X</span> Collision
</div>
""", unsafe_allow_html=True)

# Calculate global Y limits so all gallery plots align vertically
all_y = [obj['y'] for t in full_data for obj in full_data[t].values()]
y_min, y_max = min(all_y) - 100, max(all_y) + 100

# GALLERY LOOP
for i in range(0, len(all_timesteps), chunk_size):
    current_chunk = all_timesteps[i : i + chunk_size]
    if len(current_chunk) < 2: continue
    
    fig, ax = plt.subplots(figsize=(15, 3.5))
    colors = plt.cm.tab20.colors
    
    # Plot Ribbons for this chunk
    for idx in range(len(current_chunk)-1):
        t, t_next = current_chunk[idx], current_chunk[idx+1]
        for obj_id, d0 in full_data[t].items():
            y0 = d0['y']
            h0 = np.sqrt(d0['area']) / v_scale  # Apply thinning
            
            for n_id, d1 in full_data[t_next].items():
                if d0['poly'].intersects(d1['poly']):
                    y1 = d1['y']
                    h1 = np.sqrt(d1['area']) / v_scale
                    
                    verts = [(t, y0-h0/2), (t+0.5, y0-h0/2), (t_next-0.5, y1-h1/2), (t_next, y1-h1/2),
                             (t_next, y1+h1/2), (t_next-0.5, y1+h1/2), (t+0.5, y0+h0/2), (t, y0+h0/2), (t, y0-h0/2)]
                    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
                    
                    # Add thin white edge to separate overlapping ribbons
                    ax.add_patch(patches.PathPatch(Path(verts, codes), 
                                 facecolor=colors[int(obj_id)%20], alpha=0.7, lw=0.5, edgecolor='white'))

    # Plot Events for this chunk
    chunk_splits = [s for s in events['splits'] if s['t'] in current_chunk]
    for s in chunk_splits:
        ax.scatter(s['t'], full_data[s['t']][s['p']]['y'], color='lime', marker='^', s=70, edgecolors='black', zorder=25)
    
    chunk_merges = [m for m in events['merges'] if m['t'] in current_chunk]
    for m in chunk_merges:
        ax.scatter(m['t'], full_data[m['t']][m['c']]['y'], color='cyan', marker='v', s=70, edgecolors='black', zorder=25)

    if show_collisions:
        chunk_cols = [c for c in events['collisions'] if c['t'] in current_chunk]
        for c in chunk_cols:
            ax.scatter(c['t'], full_data[c['t']][c['objs'][0]]['y'], color='red', marker='x', s=50, lw=1.5, zorder=30)

    # Style
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Frames {current_chunk[0]} - {current_chunk[-1]}", fontsize=9, loc='left', color='grey')
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    
    st.pyplot(fig)
    plt.close(fig)