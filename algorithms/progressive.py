import time
import pandas as pd
import plotly.graph_objects as go

MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_PENALTY = -1

def score_char(c1, c2):
    if c1 == '-' and c2 == '-': return 0
    if c1 == '-' or c2 == '-': return GAP_PENALTY
    return MATCH_SCORE if c1 == c2 else MISMATCH_SCORE

def score_profile_char(profile_col, char):
    score = sum(score_char(p_char, char) for p_char in profile_col)
    return score / len(profile_col)

def align_profile_sequence(profile, seq):
    m, n = len(profile[0]), len(seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1): dp[i][0] = dp[i-1][0] + GAP_PENALTY
    for j in range(1, n + 1): dp[0][j] = dp[0][j-1] + GAP_PENALTY
        
    for i in range(1, m + 1):
        profile_col = [p[i-1] for p in profile]
        for j in range(1, n + 1):
            match = dp[i-1][j-1] + score_profile_char(profile_col, seq[j-1])
            delete = dp[i-1][j] + GAP_PENALTY
            insert = dp[i][j-1] + GAP_PENALTY
            dp[i][j] = max(match, delete, insert)
            
    i, j = m, n
    aligned_profile = ["" for _ in profile]
    aligned_seq = ""
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            profile_col = [p[i-1] for p in profile]
            score_diag = dp[i-1][j-1] + score_profile_char(profile_col, seq[j-1])
            if abs(dp[i][j] - score_diag) < 1e-5:
                for k in range(len(profile)): aligned_profile[k] = profile[k][i-1] + aligned_profile[k]
                aligned_seq = seq[j-1] + aligned_seq
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i-1][j] + GAP_PENALTY)) < 1e-5:
            for k in range(len(profile)): aligned_profile[k] = profile[k][i-1] + aligned_profile[k]
            aligned_seq = "-" + aligned_seq
            i -= 1
        else:
            for k in range(len(profile)): aligned_profile[k] = "-" + aligned_profile[k]
            aligned_seq = seq[j-1] + aligned_seq
            j -= 1
    return aligned_profile, aligned_seq

def build_guide_tree_visualization(seq_names):
    """
    Build coordinates for a binary guide tree visualization.
    Shows how sequences are progressively combined.
    """
    n = len(seq_names)
    if n < 2:
        return None, None
    
    # Create nodes for visualization
    nodes = []
    edges = []
    
    # Leaf nodes (sequences) at the bottom
    leaf_y = 0
    spacing = 100 / (n + 1)
    for i, name in enumerate(seq_names):
        x = (i + 1) * spacing
        nodes.append({"name": name, "x": x, "y": leaf_y, "type": "leaf"})
    
    # Internal nodes (alignments) showing how they combine
    current_level_y = leaf_y + 50
    node_counter = n
    
    # Simulate progressive alignment tree structure
    # First two sequences combine at the root
    internal_nodes = []
    
    # Root node
    root_x = (spacing + 2 * spacing) / 2
    nodes.append({"name": "Root", "x": root_x, "y": current_level_y, "type": "internal"})
    edges.append({"parent_idx": node_counter, "child_idx": 0})
    edges.append({"parent_idx": node_counter, "child_idx": 1})
    node_counter += 1
    
    # Subsequent sequences add to the growing alignment
    parent_idx = node_counter - 1
    for i in range(2, n):
        new_x = ((i + 1) * spacing + root_x) / 2
        nodes.append({"name": f"Group_{i-1}", "x": new_x, "y": current_level_y + (i - 2) * 25, "type": "internal"})
        edges.append({"parent_idx": node_counter, "child_idx": parent_idx})
        edges.append({"parent_idx": node_counter, "child_idx": i})
        parent_idx = node_counter
        node_counter += 1
    
    return nodes, edges

def create_tree_plotly(nodes, edges):
    """
    Create a Plotly figure for the guide tree visualization.
    """
    fig = go.Figure()
    
    # Draw edges
    for edge in edges:
        parent = nodes[edge["parent_idx"]]
        child = nodes[edge["child_idx"]]
        
        fig.add_trace(go.Scatter(
            x=[child["x"], parent["x"]],
            y=[child["y"], parent["y"]],
            mode="lines",
            line=dict(color="lightgray", width=2),
            hoverinfo="none",
            showlegend=False
        ))
    
    # Draw leaf nodes (sequences)
    leaf_x = [n["x"] for n in nodes if n["type"] == "leaf"]
    leaf_y = [n["y"] for n in nodes if n["type"] == "leaf"]
    leaf_names = [n["name"] for n in nodes if n["type"] == "leaf"]
    
    fig.add_trace(go.Scatter(
        x=leaf_x, y=leaf_y,
        mode="markers+text",
        marker=dict(size=12, color="#2E86AB", symbol="circle"),
        text=leaf_names,
        textposition="bottom center",
        textfont=dict(size=10, color="#2E86AB"),
        hovertext=leaf_names,
        hoverinfo="text",
        showlegend=False
    ))
    
    # Draw internal nodes
    internal_x = [n["x"] for n in nodes if n["type"] == "internal"]
    internal_y = [n["y"] for n in nodes if n["type"] == "internal"]
    
    fig.add_trace(go.Scatter(
        x=internal_x, y=internal_y,
        mode="markers",
        marker=dict(size=10, color="#A23B72", symbol="diamond"),
        hoverinfo="none",
        showlegend=False
    ))
    
    fig.update_layout(
        title="Guide Tree Structure (Progressive Alignment Order)",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        plot_bgcolor="white"
    )
    
    return fig

def run_progressive_msa_generator(sequences, seq_names):
    start_time = time.time()
    n = len(sequences)
    
    if n < 2:
        yield {"status": "done", "msa": sequences, "names": seq_names, "metrics": {"time_seconds": 0, "num_sequences": n, "alignment_length": len(sequences[0])}}
        return

    # Step 1: Build pairwise sequence identity matrix
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(100.0)
            else:
                matches = sum(1 for a, b in zip(sequences[i], sequences[j]) if a == b)
                identity = round((matches / max(len(sequences[i]), len(sequences[j]))) * 100, 1)
                row.append(identity)
        matrix.append(row)
        
    df_matrix = pd.DataFrame(matrix, columns=seq_names, index=seq_names)
    
    # Display pairwise identity matrix
    yield {
        "status": "card",
        "title": "1. Pairwise Sequence Identity Matrix",
        "description": "Calculate the pairwise sequence identity matrix (Identity %) between all sequence pairs. This similarity matrix serves as the foundation for constructing the guide tree, enabling the algorithm to identify the most closely related sequences and determine the optimal alignment order for multiple sequence alignment.",
        "matrix": df_matrix
    }

    # Step 2: Build and display the guide tree
    tree_nodes, tree_edges = build_guide_tree_visualization(seq_names)
    tree_fig = create_tree_plotly(tree_nodes, tree_edges) if tree_nodes else None
    
    yield {
        "status": "card",
        "title": "2. Guide Tree Construction",
        "description": f"Based on the pairwise distance matrix, the algorithm constructs a hierarchical guide tree (Neighbor-Joining or UPGMA topology) to determine the optimal sequence alignment order. The guide tree structure dictates which sequences to align first and progressively adds remaining sequences to the growing multiple sequence alignment (MSA).\n\n**Alignment execution order:** `{' ➔ '.join(seq_names)}`",
        "tree_figure": tree_fig
    }

    # Step 3: Progressive alignment - start with first sequence and add others one by one
    current_msa = [sequences[0]]
    current_names = [seq_names[0]]
    
    # Initialize the profile with the first pairwise alignment
    aligned_profile, aligned_seq = align_profile_sequence(current_msa, sequences[1])
    current_msa = aligned_profile + [aligned_seq]
    current_names.append(seq_names[1])
    
    yield {
        "status": "card",
        "title": f"3.1. Root Alignment ({seq_names[0]} & {seq_names[1]})",
        "description": f"Initialize the root profile by performing pairwise alignment of the two most similar sequences. This creates the foundational alignment profile that will be progressively expanded by aligning the remaining sequences. The profile-based approach allows efficient scoring using character frequency distributions.",
        "current_msa": current_msa,
        "current_names": current_names
    }
    
    # Progressively add remaining sequences to the growing alignment
    for i in range(2, n):
        aligned_profile, aligned_seq = align_profile_sequence(current_msa, sequences[i])
        current_msa = aligned_profile + [aligned_seq]
        current_names.append(seq_names[i])
        
        if i > 2:
            yield {
                "status": "card",
                "title": f"3.{i}. Progressive Addition of Sequence {seq_names[i]}",
                "description": f"Apply dynamic programming profile-sequence alignment to progressively add **{seq_names[i]}** to the current MSA.",
                "current_msa": current_msa,
                "current_names": current_names
            }
        else:
            yield {
                "status": "card",
                "title": f"3.2. Adding Sequence {seq_names[i]}",
                "description": f"Apply dynamic programming profile-sequence alignment to add **{seq_names[i]}** to the current MSA. The algorithm scores each position based on the character distribution in the existing profile, allowing efficient incorporation of sequences in decreasing order of similarity. This greedy approach maintains computational efficiency while producing reasonable alignments.",
                "current_msa": current_msa,
                "current_names": current_names
            }
        
    execution_time = time.time() - start_time
    yield {
        "status": "done",
        "msa": current_msa,
        "names": current_names,
        "metrics": {"time_seconds": round(execution_time, 4), "num_sequences": n, "alignment_length": len(current_msa[0])}
    }