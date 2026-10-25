# EEG_Resting_State_Functional_Connectivity_Network
# Resting-State EEG Connectivity, Graph Theory & DMN Analysis Pipeline

This repository provides an end-to-end Python pipeline for analyzing resting-state EEG data from the **LEMON Dataset**. The framework estimates cortical source-space dynamics, calculates Alpha-band Lagged Linear Coherence, computes complex network graph metrics (via `bctpy`), isolates Default Mode Network (DMN) topology, and models relationships with behavioral traits (e.g., BIS/BAS scores) using robust machine learning regression.

## Features

* **Automated Data Ingestion & Epoching:** Recursively locates and segments "Eyes Closed" resting-state intervals from EEGLAB (`.set`) files.
* **Cortical Source Estimation:** Employs MNE-Python with an `fsaverage` BEM model and **eLORETA** inverse modeling across 68 Desikan-Killiany cortical labels.
* **Lagged Linear Coherence:** Computes Alpha-band ($8-12\text{ Hz}$) connectivity robust to volume conduction artifact.
* **Subnetwork Isolation:** Isolates a $14 \times 14$ subnetwork corresponding to key Default Mode Network (DMN) nodes.
* **Network & Graph Theory Analysis:** Computes network-level properties using the Brain Connectivity Toolbox (`bctpy`):
  * **Weighted Global Efficiency ($GE$):** Evaluates parallel information transfer within the DMN versus the rest of the cortex.
  * **Clustering Coefficient ($CC$):** Quantifies local network segregation.
  * **Eigenvector Centrality ($EC$):** Evaluates high-influence nodal hubs.
  * **Participation Coefficient ($PC$):** Assesses inter-modular integration across hemispheres and networks.
* **Feature Export for Machine Learning:** Consolidates network-level metrics and subject-level ratios into `Combined_Graph_Metrics.csv`.
* **Behavioral & Trait Analysis:**
  * **Spearman Correlation Profiling (`correlation_matrix.py`):** Evaluates non-linear dependencies between behavioral traits and network metrics such as `GE_Ratio`.
  * **Robust Outlier-Resistant Regression (`linear_regression.py`):** Applies RANSAC regression with Ridge regularization and Robust Scaling to model behavioral predictors of network integration under cross-validation.

## Repository Structure

```text
├── config.py                 # Central configuration for paths, parameters, and DMN labels
├── data_loader.py            # Recursive file scanning, trigger alignment, and epoch generation
├── main.py                   # Source modeling, connectivity, graph theory, and metric export
├── correlation_matrix.py     # Spearman correlation matrix and heatmap visualization
├── linear_regression.py      # RANSAC-Ridge regression pipeline with outlier detection
├── visualization.py          # Histograms, connectomes, and consensus plotting routines
└── README.md
```

## Requirements & Dependencies

Install required dependencies via `pip` or `conda`:

```bash
pip install mne mne-connectivity bctpy numpy scipy pandas matplotlib seaborn plotly scikit-learn openpyxl
```

Key libraries:
* **MNE-Python & mne-connectivity**: Source estimation, BEM solving, and spectral connectivity.
* **bctpy**: Brain Connectivity Toolbox for weighted network measures.
* **scikit-learn**: Robust scaling, imputation, Ridge estimation, and RANSAC regression.
* **Plotly & Seaborn**: Interactive 3D connectomes, publication-ready heatmaps, and violin distributions.

## Execution Workflow

### 1. Preprocessing, Source Localization & Network Metrics
Configure your local directories in `config.py` (pointing to the raw LEMON files and `fsaverage` head model), then run:

```bash
python main.py
```

* **Caching:** Computes and caches full $68 \times 68$ (`all_subjects_matrices.npy`) and sliced $14 \times 14$ DMN matrices (`DMN_subjects_matrices.npy`) for rapid subsequent runs.
* **Statistical Output:** Runs paired t-tests comparing DMN connectivity and Global Efficiency against the rest of the brain.
* **Data Export:** Exports `Combined_Graph_Metrics.csv` containing subject-level metrics (`DMN_GE`, `Rest_GE`, `GE_Ratio`, `Mean_CC_DMN`, `Mean_EC_DMN`, `Mean_PC_DMN`).

### 2. Behavioral Correlation Analysis
To identify associations between behavioral questionnaires from the **LEMON Dataset** and network topology:

```bash
python correlation_matrix.py
```
* Generates a ranked Spearman correlation report against `GE_Ratio` and outputs `correlation_heatmap.png`.

### 3. Predictive Modeling (RANSAC Regression)
To predict DMN topological efficiency from behavioral indices while rejecting high-leverage outliers:

```bash
python linear_regression.py
```
* Fits a cross-validated RANSAC regressor (using a Ridge estimator) over median-imputed, robust-scaled features.
* Outputs model performance (MAE, $R^2$), inlier/outlier counts, feature importance coefficients, and generates `regression_ransac_results.png`.

## Visual Outputs

* `Global_Efficiency_DMN_vs_Rest.png`: Paired distribution violin plots for network integration.
* `Group_Mean_DMN_3d_network.html`: Interactive 3D WebGL network displaying mean connection weights and variance.
* `correlation_heatmap.png`: Matrix showing correlations between brain topology and behavioral features.
* `regression_ransac_results.png`: Regression fits highlighting consensus inliers vs. excluded outliers alongside feature impact weights.

## References

Please cite the following articles and resources if you use or adapt this pipeline:

  Vecchio, F., Miraglia, F., Marra, C., Quaranta, D., Vita, M. G., Bramanti, P., & Rossini, P. M. (2014). Human brain networks in physiological aging: A graph theoretical analysis of cortical connectivity from EEG data. *Journal of Alzheimer's Disease*, 41(4), 1239–1249. https://doi.org/10.3233/JAD-140090
* **Desikan-Killiany Atlas to DMN Mapping:**
  Kabbara, A., El Falou, W., Khalil, M., Wendling, F., & Hassan, M. (2017). The dynamic functional core network of the human brain at rest. *Scientific Reports*, 7(1), 2936. https://doi.org/10.1038/s41598-017-03420-6
* **LEMON Dataset:**
  Babayan, A., Erbey, M., Kumral, D., Reinelt, J., Reiter, A., Röbbig, J., Schaare, H. L., Uhlig, M., ... Gaebler, M., & Villringer, A. (2019). A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults. *Scientific Data*, 6, 180308. https://doi.org/10.1038/sdata.2018.308
* **MNE-Python:**
  Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. S. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267. https://doi.org/10.3389/fnins.2013.00267
* **Brain Connectivity Toolbox (BCT):**
  Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: Uses and interpretations. *NeuroImage*, 52(3), 1059–1069. https://doi.org/10.1016/j.neuroimage.2009.10.003

## References

Please cite the following articles and resources if you use or adapt this pipeline:

* **Pipeline & Graph Theoretical Connectivity Framework:**
Vecchio et al. "Human Brain Networks in Physiological Aging: A Graph Theoretical Analysis of Cortical Connectivity from EEG Data", Journal of Alzheimers Disease, 2014: 1239-1249
* **Desikan-Killiany Atlas to DMN Mapping:**
Kabbara, Aya, et al. "The dynamic functional core network of the human brain at rest." Scientific reports 7.1 (2017): 2936.
* **Lagged Coherence:**
RD Pascual-Marqui. “Instantaneous and lagged measurements of linear and nonlinear dependence between groups of multivariate time series: frequency decomposition” arXiv:0711.1455 [stat.ME], 2007
* **MNE-Python:**
Gramfort et al. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7(267):1–13, 2013.
* **fsaverage:**
Oostenveld et al. “FieldTrip: Open Source Software for Advanced Analysis of MEG, EEG, and Invasive Electrophysiological Data”, Computational Intelligence and Neuroscience, (2011):156869.
* **Desikan-Killiany Atlas:**
Desikan, Rahul S et al. “An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest.” NeuroImage vol. 31,3 (2006): 968-80
* **LEMON Dataset:**
Babayan et al. “A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults” Sci Data 6, 180308 (2019)
* **Brain Connectivity Toolbox (BCT):**
  Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: Uses and interpretations. *NeuroImage*, 52(3), 1059–1069. https://doi.org/10.1016/j.neuroimage.2009.10.003

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this software with attribution.

