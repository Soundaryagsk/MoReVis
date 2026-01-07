# MoReVis Extension: Topological Event Summaries

**Live Demo:** https://morevis-valdrighi.streamlit.app/

## Project Overview
This project extends the "MoReVis" framework (Valdrighi et al. 2023). Our extension introduces:
* **Event Detection:** Automated identification of Splits and Merges using IoU identity tracking.
* **Ambiguity Resolution:** Red 'X' markers for 2D spatial collisions to clarify 1D projection overlaps.
* **Interactive Summary:** A segmented "Gallery View" with adjustable thinning to handle dense datasets like Wildtrack.

## How to Run Locally
1. Ensure Python 3.9+ is installed.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Folder Contents
* `app.py`: Interactive Streamlit application.
* `MoReVis_Documentation.html`: Scientific documentation and algorithm explanation.
* `data/`: Processed datasets (Wildtrack, HURDAT, Motivating).
