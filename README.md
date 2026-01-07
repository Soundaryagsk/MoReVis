# MoReVis Extension - Spatiotemporal Region Summaries

**Course:** Visualization (TU Wien)  
**Task:** Extension of "MoReVis: Moving Region Visualization" (Valdrighi et al. 2023)

## Extension Features
Beyond the original 1D projection proposed by Valdrighi, this implementation adds:
1. **Topological Event Detection:** Explicit identification and visualization of region 'Splits' and 'Merges' using IOU-based identity tracking.
2. **Ambiguity Resolution:** Red 'X' markers identify 2D spatial collisions, helping users distinguish between true intersections and overlapping 1D projections.
3. **Gallery View & Vertical Scaling:** A scannable chronological summary that prevents ribbon occlusion through adjustable thinning factors.

## Installation
1. Install requirements: `pip install streamlit pandas matplotlib shapely`
2. Run the application: `streamlit run app.py`

## Data
Place datasets in `data/`. Supported: Wildtrack (pedestrians), HURDAT (hurricanes).
