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

    # ---------------------------------------------------------
    # KHẮC PHỤC 1: Tính ma trận khoảng cách bằng gióng hàng thực sự
    # ---------------------------------------------------------
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 100.0
            elif i < j:
                # Sử dụng hàm gióng hàng có sẵn (Profile 1 trình tự vs 1 trình tự)
                aligned_p, aligned_s = align_profile_sequence([sequences[i]], sequences[j])
                s1, s2 = aligned_p[0], aligned_s
                
                # Đếm số lượng match trên trình tự ĐÃ ĐƯỢC GIÓNG HÀNG
                matches = sum(1 for a, b in zip(s1, s2) if a == b and a != '-')
                # Tính % dựa trên chiều dài của bản gióng hàng
                identity = round((matches / len(s1)) * 100, 1)
                
                matrix[i][j] = identity
                matrix[j][i] = identity # Ma trận đối xứng
        
    df_matrix = pd.DataFrame(matrix, columns=seq_names, index=seq_names)
    
    yield {
        "status": "card",
        "title": "1. Pairwise Sequence Identity Matrix",
        "description": "Calculate the pairwise sequence identity matrix (Identity %) between all sequence pairs using actual dynamic programming. This similarity matrix serves as the foundation for constructing the guide tree.",
        "matrix": df_matrix
    }

    # ---------------------------------------------------------
    # KHẮC PHỤC 2: Xác định thứ tự gióng hàng (Guide Tree Order)
    # ---------------------------------------------------------
    # 2a. Tìm cặp trình tự giống nhau nhất
    max_id = -1
    best_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i][j] > max_id:
                max_id = matrix[i][j]
                best_pair = (i, j)
    
    # 2b. Xây dựng danh sách thứ tự tối ưu
    order_indices = list(best_pair)
    unaligned = set(range(n)) - set(best_pair)
    
    # Thuật toán Greedy: Chọn trình tự có độ tương đồng trung bình cao nhất với nhóm đã chọn
    while unaligned:
        best_seq = -1
        max_avg_id = -1
        for k in unaligned:
            avg_id = sum(matrix[k][m] for m in order_indices) / len(order_indices)
            if avg_id > max_avg_id:
                max_avg_id = avg_id
                best_seq = k
        order_indices.append(best_seq)
        unaligned.remove(best_seq)
        
    # 2c. Sắp xếp lại dữ liệu đầu vào theo thứ tự của Guide Tree
    ordered_sequences = [sequences[i] for i in order_indices]
    ordered_names = [seq_names[i] for i in order_indices]

    # Step 2: Build and display the guide tree (Dùng danh sách đã sắp xếp)
    tree_nodes, tree_edges = build_guide_tree_visualization(ordered_names)
    tree_fig = create_tree_plotly(tree_nodes, tree_edges) if tree_nodes else None
    
    yield {
        "status": "card",
        "title": "2. Guide Tree Construction",
        "description": f"Based on the pairwise distance matrix, the algorithm constructed a guide tree to determine the optimal sequence alignment order.\n\n**Alignment execution order:** `{' ➔ '.join(ordered_names)}`",
        "tree_figure": tree_fig
    }

    # ---------------------------------------------------------
    # Step 3: Progressive alignment - Chạy theo thứ tự đã tối ưu
    # ---------------------------------------------------------
    current_msa = [ordered_sequences[0]]
    current_names = [ordered_names[0]]
    
    # Khởi tạo profile bằng cách gióng hàng 2 trình tự giống nhau nhất
    aligned_profile, aligned_seq = align_profile_sequence(current_msa, ordered_sequences[1])
    current_msa = aligned_profile + [aligned_seq]
    current_names.append(ordered_names[1])
    
    yield {
        "status": "card",
        "title": f"3.1. Root Alignment ({ordered_names[0]} & {ordered_names[1]})",
        "description": f"Initialize the root profile by performing pairwise alignment of the two most similar sequences.",
        "current_msa": current_msa,
        "current_names": current_names
    }
    
    # Lần lượt thêm các trình tự còn lại vào Profile
    for i in range(2, n):
        aligned_profile, aligned_seq = align_profile_sequence(current_msa, ordered_sequences[i])
        current_msa = aligned_profile + [aligned_seq]
        current_names.append(ordered_names[i])
        
        if i > 2:
            yield {
                "status": "card",
                "title": f"3.{i}. Progressive Addition of Sequence {ordered_names[i]}",
                "description": f"Apply dynamic programming profile-sequence alignment to progressively add **{ordered_names[i]}** to the current MSA.",
                "current_msa": current_msa,
                "current_names": current_names
            }
        else:
            yield {
                "status": "card",
                "title": f"3.2. Adding Sequence {ordered_names[i]}",
                "description": f"Apply dynamic programming profile-sequence alignment to add **{ordered_names[i]}** to the current MSA.",
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