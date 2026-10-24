"""
config.py
Configuration file for the LEMON dataset pipeline.
Contains paths, hyperparameters, and target regions.
"""
from pathlib import Path

# =============================================================================
# FILE PATHS AND DIRECTORIES
# =============================================================================
# Folder containing .set and .fdt files
my_folder = r"../LEMON/Preprocessed/Closed_Eyes_Recordings"

# Directory to save generated graphs and HTML networks
graph_dir = Path(r"../LEMON/Processed/Graphs_And_Outputs")

# MNE fsaverage head model directories
fs_dir = Path(r"../MNE_Data/MNE-fsaverage-data/fsaverage")
subjects_dir = fs_dir.parent

# =============================================================================
# PIPELINE PARAMETERS
# =============================================================================
# Segmentation parameters
TARGET_EVENT_LABEL = '4'  # Event marked as 4 is Eyes Closed (in MNE it is 2)
DURATION_SEC = 58         # Duration to analyze (58 seconds safety margin)
WINDOW_SIZE = 2.0         # 2-second windows

# Consensus analysis threshold
#MIN_SUBJECTS_RATIO = 0.35

# =============================================================================
# REGIONS OF INTEREST (ROI)
# =============================================================================
# DMN (Default Mode Network) target regions defined from the Desikan-Killiany atlas
dmn_target_regions = [
    'rostralanteriorcingulate', 'medialorbitofrontal',
    'isthmuscingulate', 'posteriorcingulate', 'precuneus', 
    'parahippocampal', 'lateralorbitofrontal'
]
