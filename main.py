"""
main.py
Main execution script for the LEMON dataset pipeline.
Handles source space creation, inverse modeling, lagged linear coherence, and statistics.
"""
import os
import numpy as np
import pandas as pd
import mne
from scipy import stats
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator
from mne_connectivity import spectral_connectivity_epochs
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle

# Import our custom modules
import config
from data_loader import load_lemon_dataset

matplotlib.use('Agg')
mne.viz.set_3d_backend("pyvistaqt")

# =============================================================================
# 1. LOAD DATASET
# =============================================================================
dataset = load_lemon_dataset(config.my_folder)
# Format: (n_epochs, n_channels, n_times)
# n_epochs - the number of EC events in recording
# n_channels - the number of the relevant electrodes (out of 62)
# n_times - the length of the EC recording calculated by Fs*window_length + 1 = 250*2 + 1 = 501
# =============================================================================
# 2. SETUP HEAD MODEL & SOURCE SPACE
# =============================================================================
#spacing  Sources per hemisphere  Source spacing / mm  Surface area per source / mm2
#'oct6'         4098                    4.9                     24

# 5120-5120-5120 are the ammount of points in each layer of the fsaverage BEM model
# Make source space and save it for future use (uncomment if you want to regenerate it, but it is already saved in the repo)
# src = mne.setup_source_space(
#     subject='fsaverage',
#     spacing='oct6',
#     subjects_dir=subjects_dir,  
#     add_dist='patch',          
#     verbose=True
# )
# write_source_spaces(fs_dir / 'bem' / 'fsaverage-oct-6-src.fif', src, overwrite=True)

# Read back the source space and BEM solution from fsaverage
src = mne.read_source_spaces(config.fs_dir / 'bem' / 'fsaverage-oct-6-src.fif')    
existing_bem_sol = os.path.join(config.fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# Read and set the EEG electrode locations (MNI space), which are already in fsaverage
montage = mne.channels.make_standard_montage("standard_1005")

# =============================================================================
# 3. LOAD AND FILTER LABELS (DMN ISOLATION)
# =============================================================================
print("Loading Desikan-Killiany atlas and isolating DMN nodes...")

# Load the full atlas and isolate the 68 clean cortical regions
temp_full_labels = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=config.subjects_dir)
full_labels_no_unknown = [lbl for lbl in temp_full_labels if 'unknown' not in lbl.name]
original_68_names = [lbl.name for lbl in full_labels_no_unknown]

# Filter the list to isolate ONLY the DMN nodes into your 'labels' variable
labels = []
for label in temp_full_labels:
    base_name = label.name.replace('-lh', '').replace('-rh', '') 
    if base_name in config.dmn_target_regions:
        labels.append(label)

print(f"Extracted {len(labels)} DMN nodes for downstream analysis.")

# --- TRACKING PATHS AND STATE FLAGS ---
matrices_save_path = config.graph_dir / 'all_subjects_matrices.npy'
dmn_matrices_save_path = config.graph_dir / 'DMN_subjects_matrices.npy'
was_calculated = False  # Our safety flag!

# =============================================================================
# 4. LOAD AND SLICE MATRICES (WITH OVERWRITE PROTECTION)
# =============================================================================
# OPTION A: If the 14x14 DMN matrices already exist, load them instantly!
if os.path.exists(dmn_matrices_save_path):
    print(f"Found saved DMN matrices in '{dmn_matrices_save_path.name}'. Loading directly...")
    all_subjects_matrices = list(np.load(dmn_matrices_save_path))
    all_subjects_matrices_68 = list(np.load(matrices_save_path))

# OPTION B: If only the full 68x68 matrix exists, load it and slice it safely in memory
elif os.path.exists(matrices_save_path):
    print(f"Found saved full-brain 68x68 matrices in '{matrices_save_path.name}'. Slicing...")
    all_subjects_matrices_68 = list(np.load(matrices_save_path))
    
    dmn_names = [lbl.name for lbl in labels]
    dmn_indices = [original_68_names.index(name) for name in dmn_names]
    
    all_subjects_matrices = []
    for mat_68 in all_subjects_matrices_68:
        mat_sliced = mat_68[np.ix_(dmn_indices, dmn_indices)]
        all_subjects_matrices.append(mat_sliced)
        
    print(f"Successfully sliced {len(all_subjects_matrices)} matrices to 14x14 in active RAM.")

# OPTION C: If nothing exists, calculate the full 68x68 matrices first
else:
    print("No saved matrices found. Starting heavy calculations for FULL 68x68 brain networks...")
    all_subjects_matrices_68 = []
    was_calculated = True # Flag flipped! We are performing a fresh calculation

    for i in range(len(dataset)):
        dataset[i].set_montage(montage, on_missing='warn')
        dataset[i].set_eeg_reference(projection=True) 
        
        noise_cov = mne.compute_covariance(dataset[i], tmax=None, method=["shrunk", "empirical"], rank=None, verbose=True)
        
        # Make forward solution
        fwd = mne.make_forward_solution(
            dataset[i].info,
            trans="fsaverage",
            src=src,
            bem=existing_bem_sol,
            meg=False,
            eeg=True,
            mindist=5.0,
            n_jobs=None,
            verbose=True,
        )

        # Make the inverse operator for this subject using the forward solution and noise covariance
        inverse_operator = make_inverse_operator(
            dataset[i].info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True
        )

        method = "eLORETA"  
        snr = 1.0               # typical SNR value for EEG which is not evoked is 1      
        lambda2 = 1.0 / snr**2  # regularization parameter
        method_params = dict(force_equal=None, max_iter=100) 
        
        
        # apply_inverse_epochs gives us the Time domain for the epochs
        # stc.time is in miliseconds
        # stc.data is in Amperes
        stc = apply_inverse_epochs(
            dataset[i],
            inverse_operator,
            lambda2,
            method=method,
            pick_ori="normal",
            method_params=method_params,
            verbose=True,
        )

        # We MUST use the Time-Domain epochs for connectivity!

        #Extract the Time-Series data for all epochs
        # VERY IMPORTANT: mode must be 'mean_flip' for time-domain data to avoid canceling out!
        label_ts = mne.extract_label_time_course(
            stcs=stc,
            labels=original_68_names,
            src=inverse_operator['src'],
            mode='mean_flip',
            return_generator=False
        )
        
        # Calculate Complex Coherency ('cohy') 
        # 'cohy' returns complex numbers (Real + Imaginary), unlike 'coh' which just gives magnitude
        print("Calculating Complex Coherency...")
        con = spectral_connectivity_epochs(
            label_ts,                       
            method='cohy',                  
            sfreq=dataset[i].info['sfreq'], 
            fmin=8.0,             # Alpha band          
            fmax=12.0,                      
            faverage=True,                  
            mode='multitaper',              
            n_jobs=1
        )

        # Get the dense complex matrix (Shape: 68 x 68 x 1 frequency band)
        cohy_matrix = con.get_data(output='dense')[:, :, 0]

        # Apply the Pascual-Marqui formula for Lagged Linear Coherence
        print("Applying Lagged Linear Coherence transformation...")
        real_c = np.real(cohy_matrix)
        imag_c = np.imag(cohy_matrix)

        # Formula: Im(C)^2 / (1 - Re(C)^2)
        # (We use np.errstate to safely ignore the division-by-zero warning on the diagonal)
        with np.errstate(divide='ignore', invalid='ignore'):
            lagged_coh_sq = (imag_c ** 2) / (1.0 - (real_c ** 2))

        # Take the square root to get the linear value (rather than squared)
        lagged_coh = np.sqrt(lagged_coh_sq)
        # The diagonal (a region's connectivity to itself) will be NaN or Inf. Set it to 0.
        np.fill_diagonal(lagged_coh, 0)
        # Clean up any other potential NaNs just to be safe
        lagged_coh = np.nan_to_num(lagged_coh)
        # Since the lagged coherence matrix is not symmetric, we can symmetrize it by taking the maximum value between (i,j) and (j,i) for each pair of regions. This way, we ensure that if there is a strong connection in either direction, it will be reflected in the final matrix.
        lagged_coh_symmetric = np.maximum(lagged_coh, lagged_coh.T)

        all_subjects_matrices_68.append(lagged_coh_symmetric)
        print(f"Subject {i+1} full 68x68 Lagged Coherence matrix computed! Shape: {lagged_coh_symmetric.shape}")
    
    np.save(matrices_save_path, all_subjects_matrices_68)
    print(f"Saved brand new full-brain matrices to '{matrices_save_path.name}'.")

    # Now, slice them down to 14x14 in memory for the rest of this execution run
    print("Slicing newly calculated matrices down to DMN nodes...")
    dmn_names = [lbl.name for lbl in labels]
    dmn_indices = [original_68_names.index(name) for name in dmn_names]
    
    all_subjects_matrices = []
    for mat_68 in all_subjects_matrices_68:
        mat_sliced = mat_68[np.ix_(dmn_indices, dmn_indices)]
        all_subjects_matrices.append(mat_sliced)

# =============================================================================
# 5. SAFE EXPORT BLOCK
# =============================================================================
# Save the 14 ROI DMN matrices separately if they haven't been cached yet
if not os.path.exists(dmn_matrices_save_path) and len(all_subjects_matrices) > 0:
    np.save(dmn_matrices_save_path, all_subjects_matrices)
    print(f"Saved isolated 14x14 DMN matrices separately to '{dmn_matrices_save_path.name}'!")

# Only save the main 68x68 matrix if it was newly calculated (via our flag)
if was_calculated and len(dataset) > 0:
    print("Full-brain 68x68 matrix file was safely preserved and updated.")


# =============================================================================
# 6. GROUP LEVEL ANALYSIS (MEAN & STD ACROSS ALL SUBJECTS)
# =============================================================================
if len(all_subjects_matrices) > 0:
    print("\n" + "="*50)
    print("STARTING GROUP-LEVEL STATISTICAL ANALYSIS")
    print("="*50)
    
    # Stack all matrices and calculate Mean and Standard Deviation
    # Shape goes from a list of (14x14) to a 3D array of (N_Subjects, 14, 14)
    stacked_matrices = np.array(all_subjects_matrices)
    mean_matrix = np.mean(stacked_matrices, axis=0)
    std_matrix = np.std(stacked_matrices, axis=0)
    
    label_names = [label.name for label in labels]
    num_nodes = len(label_names)

    # =========================================================================
    # EXTRACT, SORT, AND EXPORT THE STRONGEST CONNECTIONS
    # =========================================================================
    connections = []
    # Loop through the upper triangle of the matrix to get unique edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            connections.append({
                'node1': label_names[i],
                'node2': label_names[j],
                'mean': mean_matrix[i, j],
                'std': std_matrix[i, j]
            })
    # Sort the list of dictionaries by the 'mean' value in descending order
    connections.sort(key=lambda x: x['mean'], reverse=True)
    
    # Print to console and save to a text file
    list_save_path = config.graph_dir / 'Group_Mean_DMN_Connections_Ranked.txt'
    with open(list_save_path, 'w') as f:
        header = f"{'Rank':<5} | {'Node 1':<25} | {'Node 2':<25} | {'Mean Connectivity':<18} | {'Standard Deviation'}\n"
        header += "-" * 95 + "\n"
        f.write(header)
        print(f"\nTop 15 Strongest Connections (Full list saved to {list_save_path.name}):")
        
        for rank, edge in enumerate(connections):
            line = f"{rank+1:<5} | {edge['node1']:<25} | {edge['node2']:<25} | {edge['mean']:<18.5f} | ± {edge['std']:.5f}\n"
            f.write(line)
            if rank < 15: # Only print the top 15 to console so it doesn't flood the screen
                print(line.strip())
                
    print(f"\nSuccessfully saved full ranked list to: {list_save_path.name}")

    # =========================================================================
    # GRAPH 1: Heatmap of the Mean Matrix
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        mean_matrix,
        xticklabels=label_names,
        yticklabels=label_names,
        cmap='viridis',
        square=True,
        cbar_kws={'label': 'Mean Lagged Linear Coherence'}
    )
    plt.title("Group Mean: DMN Alpha Band Lagged Coherence", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    heatmap_save_path = config.graph_dir / 'Group_Mean_DMN_heatmap.png'
    fig.savefig(heatmap_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"1. Saved Group Heatmap to: {heatmap_save_path.name}")

    # =========================================================================
    # GRAPH 2: 3D Plotly Network (With Mean and Std in Hover Text)
    # =========================================================================
    node_coords = np.array([label.pos.mean(axis=0) * 1000 for label in labels])
    traces = []
    colormap = plt.cm.YlOrRd
    max_weight = np.max(mean_matrix)
    min_weight = np.min(mean_matrix[mean_matrix > 0]) 

    nodes_trace = go.Scatter3d(
        x=node_coords[:, 0], y=node_coords[:, 1], z=node_coords[:, 2],
        mode='markers+text',
        marker=dict(size=8, color='skyblue', line=dict(color='darkblue', width=2)),
        text=label_names,       
        hoverinfo='text',
        name='DMN Regions'
    )
    traces.append(nodes_trace)
   
    # Edge Traces (Plotting ALL 91 connections - 14*(14-1)/2 = 91)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            m_weight = mean_matrix[i, j]
            s_weight = std_matrix[i, j]
            
            if m_weight > 0: 
                normalized_weight = (m_weight - min_weight) / (max_weight - min_weight + 1e-9)
                color_rgb = colormap(normalized_weight)[:3]
                color_hex = colors.rgb2hex(color_rgb)
                
                x0, y0, z0 = node_coords[i]
                x1, y1, z1 = node_coords[j]
                
                traces.append(go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1],
                    mode='lines',
                    line=dict(color=color_hex, width=3),
                    hoverinfo='text',
                    text=f"{label_names[i]} ↔ {label_names[j]}<br>Mean: {m_weight:.4f}<br>Std: ±{s_weight:.4f}",
                    showlegend=False
                ))

    fig3d = go.Figure(data=traces)
    fig3d.update_layout(
        title="Group Mean: 3D DMN Connectivity",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    plotly_save_path = config.graph_dir / 'Group_Mean_DMN_3d_network.html'
    fig3d.write_html(plotly_save_path)
    print(f"2. Saved Group 3D Network to: {plotly_save_path.name}")

    # =========================================================================
    # GRAPH 3: Circular Graph
    # =========================================================================
    lh_labels = [name for name in label_names if name.endswith('-lh')]
    rh_labels = [name for name in label_names if name.endswith('-rh')]

    lh_labels.sort(key=lambda name: labels[label_names.index(name)].pos[:, 1].mean())
    rh_labels.sort(key=lambda name: labels[label_names.index(name)].pos[:, 1].mean())
    node_order = lh_labels[::-1] + rh_labels

    node_angles = circular_layout(
        label_names, 
        node_order, 
        start_pos=90,  
        group_boundaries=[0, len(label_names) // 2] 
    )

    node_colors = ['indianred' if name.endswith('-lh') else 'royalblue' for name in label_names]

    fig_circ, ax_circ = plot_connectivity_circle(
        mean_matrix,  
        label_names,              
        n_lines=91,              
        node_angles=node_angles,    
        node_colors=node_colors,  
        fontsize_names=10,         
        title='Group Mean: DMN Alpha Lagged Coherence',
        show=False
    )
    
    circle_save_path = config.graph_dir / 'Group_Mean_DMN_circular_graph.png'
    fig_circ.savefig(circle_save_path, dpi=300, facecolor='black', bbox_inches='tight')
    plt.close(fig_circ)
    print(f"3. Saved Group Circular Graph to: {circle_save_path.name}")


# =============================================================================
# 7. STATISTICAL VALIDATION: DMN DOMINANCE VS. REST OF BRAIN
# =============================================================================
if len(all_subjects_matrices_68) > 0:
    print("\n" + "="*50)
    print("STATISTICAL TEST: DMN vs. REST OF BRAIN")
    print("="*50)

    num_nodes_total = 68
    mean_dmn_per_subj = []
    mean_rest_per_subj = []     #including DMN ROI
    mean_pure_rest_per_subj = []#Not including DMN ROI

    filtered_matrices_68 = []
    filtered_matrices_14 = []
    dominant_dmn_indices = [] 
    
    filtered_mean_dmn = []
    filtered_mean_pure_rest = []
    
    full_labels_for_stats = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=config.subjects_dir)
    original_68_names = [lbl.name for lbl in full_labels_for_stats if 'unknown' not in lbl.name]
    dmn_names = [lbl.name for lbl in labels]
    dmn_indices = [original_68_names.index(name) for name in dmn_names]
    rest_indices_stats = [i for i in range(num_nodes_total) if i not in dmn_indices]

    for idx, mat_68 in enumerate(all_subjects_matrices_68):
        # Create a blank 68x68 boolean mask
        dmn_mask = np.zeros((num_nodes_total, num_nodes_total), dtype=bool)
        # Flag only the intersections of the DMN nodes as True
        dmn_mask[np.ix_(dmn_indices, dmn_indices)] = True

        # Mask for Pure "Rest of Brain" (54x54)
        # Get all indices that are NOT in the DMN
        rest_indices = [i for i in range(num_nodes_total) if i not in dmn_indices]
        pure_rest_mask = np.zeros((num_nodes_total, num_nodes_total), dtype=bool)
        pure_rest_mask[np.ix_(rest_indices, rest_indices)] = True
        
        # Create an upper-triangle mask to avoid duplicate edges and the 0-diagonal
        upper_tri = np.triu(np.ones((num_nodes_total, num_nodes_total), dtype=bool), k=1)
        
        
        # Create the final masks: 
        # DMN edges = True in DMN mask AND True in upper triangle
        # Rest of Brain edges = False in DMN mask AND True in upper triangle
        # Pure rest of the Brain edges = True in Pure rest of the brain mask AND True in upper triangle
        valid_dmn_mask = dmn_mask & upper_tri
        valid_rest_mask = (~dmn_mask) & upper_tri
        valid_pure_rest_mask = pure_rest_mask & upper_tri

        # Extract the actual connectivity values using the masks
        dmn_edge_values = mat_68[valid_dmn_mask]
        rest_edge_values = mat_68[valid_rest_mask]
        pure_rest_edge_values = mat_68[valid_pure_rest_mask]
        
        subj_mean_dmn = np.mean(dmn_edge_values)
        subj_mean_rest = np.mean(rest_edge_values)
        subj_mean_pure_rest = np.mean(pure_rest_edge_values)
        
        # Condition: Keep subject only if DMN connectivity is stronger than the rest of the brain without DMN ROI
        if subj_mean_dmn > subj_mean_pure_rest:
            filtered_matrices_68.append(mat_68)
            filtered_matrices_14.append(all_subjects_matrices[idx])
            dominant_dmn_indices.append(idx) 

            filtered_mean_dmn.append(subj_mean_dmn)
            filtered_mean_pure_rest.append(subj_mean_pure_rest)

        # Calculate the average connectivity for this specific subject
        mean_dmn_per_subj.append(np.mean(dmn_edge_values))
        mean_rest_per_subj.append(np.mean(rest_edge_values))
        mean_pure_rest_per_subj.append(np.mean(pure_rest_edge_values))
        
    print(f"Filtered results: Kept {len(filtered_matrices_68)} out of {len(all_subjects_matrices_68)} subjects.")

    if len(filtered_mean_dmn) > 1:
        t_stat_filtered, p_val_filtered = stats.ttest_rel(filtered_mean_dmn, filtered_mean_pure_rest)
        # Put rest of the brain with DMN ROI
        print("\n--- STATISTICAL VALIDATION ON FILTERED SUB-GROUP ---")
        print(f"Filtered DMN Mean:       {np.mean(filtered_mean_dmn):.5f} ± {np.std(filtered_mean_dmn):.5f}")
        print(f"Filtered Pure Rest Mean: {np.mean(filtered_mean_pure_rest):.5f} ± {np.std(filtered_mean_pure_rest):.5f}")
        print(f"Paired t-test results:   t = {t_stat_filtered:.3f}, p = {p_val_filtered:.3e}")
        
        if p_val_filtered < 0.05:
            print("-> As expected: The DMN is significantly stronger in this sub-group.")
        else:
            print("-> Note: Difference is not statistically significant.")
    else:
        print("\nNot enough dominant subjects to perform a t-test.")
    print(f"\nIndices of subjects with dominant DMN (0-based):")
    print(dominant_dmn_indices)

    indices_save_path = config.graph_dir / 'Dominant_DMN_Subject_Indices.txt'
    with open(indices_save_path, 'w') as f:
        f.write(f"Total Dominant Subjects: {len(dominant_dmn_indices)}\n")
        f.write("0-Based Indices:\n")
        f.write(", ".join(map(str, dominant_dmn_indices)))
    print(f"Saved these indices to: {indices_save_path.name}\n")

    all_subjects_matrices_68_filtered_DMN = filtered_matrices_68
    all_subjects_matrices_14_filtered_DMN = filtered_matrices_14

    # =========================================================================
    # PERFORM PAIRED T-TEST
    # =========================================================================
    t_stat, p_val = stats.ttest_rel(mean_dmn_per_subj, mean_rest_per_subj)
    t_stat_pure, p_val_pure = stats.ttest_rel(mean_dmn_per_subj, mean_pure_rest_per_subj)

    group_mean_dmn = np.mean(mean_dmn_per_subj)
    group_mean_rest = np.mean(mean_rest_per_subj)
    group_mean_pure_rest = np.mean(mean_pure_rest_per_subj)
    group_std_dmn = np.std(mean_dmn_per_subj)
    group_std_rest = np.std(mean_rest_per_subj)
    group_std_pure_rest = np.std(mean_pure_rest_per_subj)


    print(f"DMN Mean Connectivity:           {group_mean_dmn:.5f} ± {group_std_dmn:.5f}")
    print(f"Rest of Brain with DMN regions Mean Connectivity: {group_mean_rest:.5f} ± {group_std_rest:.5f}")
    print(f"Rest of Brain without DMN regions Mean Connectivity: {group_mean_pure_rest:.5f} ± {group_std_pure_rest:.5f}")
    print(f"Paired t-test results with DMN regions in the rest of the brain:           t = {t_stat:.3f}, p = {p_val:.3e}")
    print(f"Paired t-test results without DMN regions in the rest of the brain:           t = {t_stat_pure:.3f}, p = {p_val_pure:.3e}")

    if p_val < 0.05:
        if group_mean_dmn > group_mean_rest:
            print("-> SIGNIFICANT: The DMN is more connected than the rest of the brain(with DMN regions).")
        else:
            print("-> SIGNIFICANT: The DMN is LESS connected than the rest of the brain(with DMN regions).")
    else:
        print("-> n.s.: No significant difference found between DMN and the rest of the brain(with DMN regions).")
    
    if p_val_pure < 0.05:
        if group_mean_dmn > group_mean_pure_rest:
            print("-> SIGNIFICANT: The DMN is more connected than the rest of the brain(without DMN regions).")
        else:
            print("-> SIGNIFICANT: The DMN is LESS connected than the rest of the brain(without DMN regions).")
    else:
        print("-> n.s.: No significant difference found between DMN and the rest of the brain(without DMN regions).")


    # =========================================================================
    # VISUALIZATION: VIOLIN / BOXPLOT
    # =========================================================================
    # Format data for Seaborn plotting
    plot_data = pd.DataFrame({
        'Subject': np.arange(1, len(mean_dmn_per_subj) + 1).tolist() * 2,
        'Mean Lagged Coherence': mean_dmn_per_subj + mean_rest_per_subj,
        'Network': ['DMN (14 Nodes)'] * len(mean_dmn_per_subj) + ['Rest of Brain(with DMN regions)'] * len(mean_rest_per_subj)
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=plot_data, x='Network', y='Mean Lagged Coherence', palette=['skyblue', 'lightgray'], inner=None, alpha=0.6, ax=ax)
    sns.stripplot(data=plot_data, x='Network', y='Mean Lagged Coherence', color='black', alpha=0.5, jitter=True, size=5, ax=ax)
    
    significance_text = f"p = {p_val:.3e}\n***" if p_val < 0.001 else (f"p = {p_val:.3f}\n*" if p_val < 0.05 else "n.s.")
    
    y_max = plot_data['Mean Lagged Coherence'].max()
    y_range = y_max - plot_data['Mean Lagged Coherence'].min()
    bracket_y = y_max + (y_range * 0.05)
    
    ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + (y_range*0.02), bracket_y + (y_range*0.02), bracket_y], lw=1.5, color='black')
    ax.text(0.5, bracket_y + (y_range*0.03), significance_text, ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

    plt.title("Resting-State Alpha Connectivity: DMN vs Rest of Brain(with DMN regions)", fontsize=14, pad=20)
    plt.ylabel("Mean Lagged Linear Coherence")
    plt.xlabel("")
    plt.ylim(bottom=plot_data['Mean Lagged Coherence'].min() * 0.9, top=bracket_y + (y_range*0.15))
    plt.tight_layout()
    
    stats_save_path = config.graph_dir / 'DMN_vs_Rest_of_Brain_Stats_with_DMN_regions.png'
    fig.savefig(stats_save_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved statistical comparison graph to: {stats_save_path.name}")