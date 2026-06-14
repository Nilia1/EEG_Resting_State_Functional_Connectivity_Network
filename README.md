# EEG_Resting_State_Functional_Connectivity_Network
# Resting-State EEG Connectivity & DMN Analysis Pipeline

This repository contains a complete Python pipeline for processing resting-state EEG data from the **LEMON Dataset**. The pipeline calculates source-space connectivity using Alpha-band Lagged Linear Coherence, isolates the Default Mode Network (DMN), and performs group-level statistical validation and consensus analyses.

## Features

* **Automated Epoching:** Recursively scans directories to extract and segment "Eyes Closed" resting-state data from EEGLAB (`.set`) files.
* **Source Estimation:** Utilizes MNE-Python to build an `fsaverage` head model and applies **eLORETA** inverse modeling to map sensor data to 68 Desikan-Killiany cortical regions.
* **Connectivity Calculation:** Computes Complex Coherency and transforms it into **Lagged Linear Coherence** in the Alpha band (8-12 Hz).
* **DMN Isolation:** Dynamically slices the full 68x68 brain network to isolate a 14x14 network of Default Mode Network (DMN) nodes.
* **Statistical Validation:** Performs paired t-tests to compare DMN connectivity strength against the rest of the brain.
* **Advanced Visualization:** Generates group-level heatmaps, circular graphs, statistical violin plots, and interactive 3D Plotly brain networks.

## Project Structure

* `config.py`: Contains hardcoded file paths, dataset configurations, hyperparameters, and specific target regions (DMN).
* `data_loader.py`: Handles recursive file searching, event ID mapping, and MNE Epoch creation.
* `main.py`: The primary execution script that ties together data loading, source space calculations, connectivity modeling, and statistical outputs.

## Requirements

Ensure you have Python installed along with the following primary libraries:
* `mne`
* `mne-connectivity`
* `numpy`
* `scipy`
* `pandas`
* `matplotlib`
* `seaborn`
* `plotly`

## Usage

1. **Configure Paths:** Open `config.py` and update the local paths to point to your LEMON dataset and `mne_data/MNE-fsaverage-data/fsaverage` directory.
2. **Run the Pipeline:** Execute the main script from your terminal or IDE:
   ```bash
   python main.py

## References
Vecchio et al. "Human Brain Networks in Physiological Aging: A Graph Theoretical Analysis of Cortical Connectivity from EEG Data", Journal of Alzheimers Disease, 2014: 1239-1249

Kabbara, Aya, et al. "The dynamic functional core network of the human brain at rest." Scientific reports 7.1 (2017): 2936.

Olbrich et al. “Altered EEG lagged coherence during rest in obsessive–compulsive disorder”, Clinical Neurophysiology vol. 124, no. 12(2013): 2421-2430

Gramfort et al. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7(267):1–13, 2013.

Oostenveld et al. “FieldTrip: Open Source Software for Advanced Analysis of MEG, EEG, and Invasive Electrophysiological Data”, Computational Intelligence and Neuroscience, (2011):156869.

Desikan, Rahul S et al. “An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest.” NeuroImage vol. 31,3 (2006): 968-80

RD Pascual-Marqui. “Instantaneous and lagged measurements of linear and nonlinear dependence between groups of multivariate time series: frequency decomposition” arXiv:0711.1455 [stat.ME], 2007

Babayan et al. “A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults” Sci Data 6, 180308 (2019)

