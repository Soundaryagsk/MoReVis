import streamlit as st
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from shapely.geometry import Polygon
from collections import defaultdict

st.set_page_config(page_title="MoReVis Extension: Project Final", layout="wide")

@st.cache_data
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    t_col = next((c for c in ['timestep', 'frame', 'time'] if c in df.columns), None)
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
                        'poly': poly, 'y': row['ycenter'],
                        'area': row['area'], 'id': str(row[id_col])
                    }
            except: continue
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
        ts_list = sorted(frames.keys())
        for i in range(len(ts_list)):
            t = ts_list[i]
            objs = frames[t]
            ids = list(objs.keys())
            # Spatial Collisions
            for idx1 in range(len(ids)):
                for idx2 in range(idx1 + 1, len(ids)):
                    if objs[ids[idx1]]['poly'].intersects(objs[ids[idx2]]['poly']):
                        events['collisions'].append({'t': t, 'objs': (ids[idx1], ids[idx2])})
            # Topological Splits/Merges
            if i < len(ts_list) - 1:
                t_next = ts_list[i+1]
                mapping, rev_mapping = defaultdict(list), defaultdict(list)
                for p_id, p_data in objs.items():
                    for n_id, n_data in frames[t_next].items():
                        if p_data['poly'].intersects(n_data['poly']):
                            if self.get_iou(p_data['poly'], n_data['poly']) > self.iou_threshold:
                                mapping[p_id].append(n_id); rev_mapping[n_id].append(p_id)
                for p_id, children in mapping.items():
                    if len(children) > 1: events['splits'].append({'t': t, 'p': p_id, 'c': children})
                for n_id, parents in rev_mapping.items():
                    if len(parents) > 1: events['merges'].append({'t': t+1, 'c': n_id, 'p': parents})
        return events

st.sidebar.header("1. Data Selection")
dataset_choice = st.sidebar.selectbox("Dataset", ["data/wildtrack.csv", "data/hurdat.csv", "data/motivating.csv"])

st.sidebar.header("2. Semantic Filters")
show_splits = st.sidebar.checkbox("Show Splits (Green ▲)", value=True)
show_merges = st.sidebar.checkbox("Show Merges (Blue ▼)", value=True)
show_collisions = st.sidebar.checkbox("Show Collisions (Red X)", value=True)

st.sidebar.header("3. Visual Tuning")
iou_val = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.45)
chunk_size = st.sidebar.slider("Frames per Segment", 20, 100, 50)
v_scale = st.sidebar.slider("Ribbon Thinning", 1.0, 15.0, 8.0)

full_data = load_dataset(dataset_choice)
all_timesteps = sorted(full_data.keys())
analyzer = MoReVisExtension(iou_threshold=iou_val)
events = analyzer.analyze(full_data)

st.title("MoReVis Extension: Topological Event Summary")
st.info("Interactive Toggle enabled in Sidebar. Adjust the 'Semantic Filters' to focus on specific event types.")

# Calculate global Y limits for alignment across segments
all_y = [obj['y'] for t in full_data for obj in full_data[t].values()]
y_min, y_max = min(all_y) - 200, max(all_y) + 200


for i in range(0, len(all_timesteps), chunk_size):
    current_chunk = all_timesteps[i : i + chunk_size]
    if len(current_chunk) < 2: continue
    
    fig, ax = plt.subplots(figsize=(15, 4))
    colors = plt.cm.tab20.colors
    
    # Draw Ribbons using Cubic Bezier Paths
    for idx in range(len(current_chunk)-1):
        t, t_next = current_chunk[idx], current_chunk[idx+1]
        for obj_id, d0 in full_data[t].items():
            y0, h0 = d0['y'], np.sqrt(d0['area']) / v_scale
            for n_id, d1 in full_data[t_next].items():
                if d0['poly'].intersects(d1['poly']):
                    y1, h1 = d1['y'], np.sqrt(d1['area']) / v_scale
                    
                    # C1 Continuous Cubic Bezier Ribbon
                    verts = [(t, y0-h0/2), (t+0.5, y0-h0/2), (t_next-0.5, y1-h1/2), (t_next, y1-h1/2),
                             (t_next, y1+h1/2), (t_next-0.5, y1+h1/2), (t+0.5, y0+h0/2), (t, y0+h0/2), (t, y0-h0/2)]
                    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
                    ax.add_patch(patches.PathPatch(Path(verts, codes), facecolor=colors[int(obj_id)%20], alpha=0.6, lw=0.3, edgecolor='white'))

    # To draw events
    if show_splits:
        for s in [s for s in events['splits'] if s['t'] in current_chunk]:
            y = full_data[s['t']][s['p']]['y']
            # marker (line + triangle)
            ax.plot([s['t'], s['t']], [y, y+40], color='black', lw=0.5, alpha=0.6)
            ax.scatter(s['t'], y+40, color='lime', marker='^', s=100, edgecolors='black', zorder=30)

    if show_merges:
        for m in [m for m in events['merges'] if m['t'] in current_chunk]:
            y = full_data[m['t']][m['c']]['y']
            # marker (line + triangle)
            ax.plot([m['t'], m['t']], [y, y-40], color='black', lw=0.5, alpha=0.6)
            ax.scatter(s['t'], y-40, color='cyan', marker='v', s=100, edgecolors='black', zorder=30)

    if show_collisions:
        for c in [c for c in events['collisions'] if c['t'] in current_chunk]:
            # Draw collision at the y-center of the first object involved
            ax.scatter(c['t'], full_data[c['t']][c['objs'][0]]['y'], color='red', marker='x', s=60, lw=2, zorder=35)

    # Styling and Axis Alignment
    ax.set_ylim(y_min, y_max)
    ax.set_facecolor('#ffffff')
    ax.set_title(f"Frames {current_chunk[0]} - {current_chunk[-1]}", loc='left', color='grey', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)
    plt.close(fig)