"""
visualization.py
Handles the visualization of connectivity matrices (Histograms, 3D Networks, Bar Charts).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plotly.graph_objects as go
import config

def analyze_consistency_and_consensus(all_matrices, labels, min_subjects_ratio=config.MIN_SUBJECTS_RATIO):
    """
    all_matrices: a list of 68x68 (or 14x14) connectivity matrices (one per subject).
    labels: a list of MNE label objects (contains names and 3D coordinates).
    min_subjects_ratio: the minimum ratio of subjects that must share a connection 
                        to be included in the 3D consensus graph.
    """
    # 1. Setup Data
    stacked_matrices = np.array(all_matrices)
    num_subjects, num_nodes, _ = stacked_matrices.shape
    
    # Extract names from the labels object
    label_names = [label.name for label in labels]
    
    print(f"\n--- Starting Consistency Analysis for {num_subjects} subjects ---")
    graph_dir = config.graph_dir

    # Binarize per-subject (Everyone contributes their personal top 10%)
    binarized_matrices = np.zeros_like(stacked_matrices)
    
    for s in range(num_subjects):
        subj_matrix = stacked_matrices[s]
        # Calculate the 90th percentile for THIS specific subject
        subj_thresh = np.percentile(subj_matrix[subj_matrix > 0], 90)
        # Binarize this subject's matrix
        binarized_matrices[s] = (subj_matrix > subj_thresh).astype(int)
        
    print(f"Binarized matrices using per-subject top 10% thresholds.")
    
    # =========================================================
    # PART 1: HISTOGRAMS
    # =========================================================
    iu = np.triu_indices(num_nodes, k=1)
    
    # --- Histogram 1: Node Degree ---
    node_degrees_per_subject = np.sum(binarized_matrices, axis=2)
    average_node_degree = np.mean(node_degrees_per_subject, axis=0)
    
    sorted_indices = np.argsort(average_node_degree)[::-1]
    sorted_labels = [label_names[idx] for idx in sorted_indices]
    sorted_degrees = average_node_degree[sorted_indices]
    
    fig1 = plt.figure(figsize=(15, 6))
    plt.bar(sorted_labels, sorted_degrees, color='skyblue', edgecolor='black')
    plt.title("Average Number of Connections per Region") 
    plt.xlabel("Brain Region")
    plt.ylabel("Average Degree (Number of Connections)")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    fig1.savefig(graph_dir / 'node_degree_histogram.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # --- Histogram 2: Edge Frequency ---
    edge_consensus_matrix = np.sum(binarized_matrices, axis=0) 

    # =========================================================
    # PART 2: CONSENSUS TEXT & 3D GRAPH
    # =========================================================
    min_required = int(num_subjects * min_subjects_ratio)
    
    print(f"\n--- Alpha Band Consensus Connections ---")
    print(f"Showing edges present in >= {min_required} out of {num_subjects} subjects ({min_subjects_ratio*100}%):")
    print("-" * 50)
    
    edges_to_plot = []
    # Find the edges that passed the ratio test
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            count = int(edge_consensus_matrix[i, j])
            if count >= min_required:
                edges_to_plot.append((i, j, count))
                print(f"[{count:2d} subjects] {label_names[i]} <--> {label_names[j]}")
                
    if not edges_to_plot:
        print("No consensus edges met the requirement for the 3D graph. Try lowering min_subjects_ratio.")
        return

    # --- 3D Plotly Network ---
    print("\nGenerating 3D interactive consensus network...")
    node_coords = []
    for label in labels:
        coord_mm = label.pos.mean(axis=0) * 1000 
        node_coords.append(coord_mm)
    node_coords = np.array(node_coords)
    
    traces = []
    
    # 1. Add Nodes
    nodes_trace = go.Scatter3d(
        x=node_coords[:, 0], y=node_coords[:, 1], z=node_coords[:, 2],
        mode='markers',
        marker=dict(size=7, color='skyblue', line=dict(color='darkblue', width=1)),
        text=label_names, hoverinfo='text', name='Regions'
    )
    traces.append(nodes_trace)
    
    # 2. Add Edges
    colormap = plt.get_cmap('winter')
    max_count = num_subjects
    
    for i, j, count in edges_to_plot:
        denom = max(1, max_count - min_required)
        normalized_weight = (count - min_required) / denom
        
        color_rgb = colormap(normalized_weight)[:3]
        color_hex = colors.rgb2hex(color_rgb)
        
        x0, y0, z0 = node_coords[i]
        x1, y1, z1 = node_coords[j]
        
        traces.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(color=color_hex, width=4), 
            hoverinfo='text',
            text=f"{label_names[i]} ↔ {label_names[j]}<br>Shared by: {count}/{num_subjects} subjects",
            showlegend=False
        ))

    # Format and save
    fig3 = go.Figure(data=traces)
    fig3.update_layout(
        title=f"3D Consensus Network (Shared by >= {int(min_subjects_ratio*100)}% of subjects)",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    save_path_3d = graph_dir / f'consensus_3d_network_{int(min_subjects_ratio*100)}pct.html'
    fig3.write_html(save_path_3d)
    print(f"Done! 3D network saved successfully as '{save_path_3d.name}'")

    # =========================================================
    # PART 3: CONSENSUS EDGES BAR CHART
    # =========================================================
    print("\nGenerating Consensus Edges Bar Chart...")
    
    edge_names = []
    edge_counts = []
    
    sorted_edges = sorted(edges_to_plot, key=lambda x: x[2], reverse=True)
    
    for i, j, count in sorted_edges:
        edge_string = f"{label_names[i]} ↔ {label_names[j]}"
        edge_names.append(edge_string)
        edge_counts.append(count)

    fig4 = plt.figure(figsize=(15, 6))
    plt.bar(edge_names, edge_counts, color='coral', edgecolor='black')
    plt.title(f"Most Common Consensus Connections (Shared by >= {int(min_subjects_ratio*100)}% of subjects)")
    plt.xlabel("Specific Brain Connections (Edges)")
    plt.ylabel("Number of Subjects Sharing the Connection")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    
    save_path_edges_bar = graph_dir / f'consensus_edges_barchart_{int(min_subjects_ratio*100)}pct.png'
    fig4.savefig(save_path_edges_bar, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"Consensus Edges Bar Chart saved successfully as '{save_path_edges_bar.name}'")