import time
import io
import matplotlib.pyplot as plt
from Bio import Phylo
from collections import Counter

MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_PENALTY = -1

def score_char(c1, c2):
    if c1 == '-' and c2 == '-': return 0
    if c1 == '-' or c2 == '-': return GAP_PENALTY
    return MATCH_SCORE if c1 == c2 else MISMATCH_SCORE

def score_profile_profile(col1, col2):
    score = 0
    for c1 in col1:
        for c2 in col2:
            score += score_char(c1, c2)
    return score / (len(col1) * len(col2))

def align_profiles(prof1, prof2):
    m, n = len(prof1[0]), len(prof2[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1): dp[i][0] = dp[i-1][0] + GAP_PENALTY
    for j in range(1, n + 1): dp[0][j] = dp[0][j-1] + GAP_PENALTY
        
    for i in range(1, m + 1):
        col1 = [p[i-1] for p in prof1]
        for j in range(1, n + 1):
            col2 = [p[j-1] for p in prof2]
            match = dp[i-1][j-1] + score_profile_profile(col1, col2)
            delete = dp[i-1][j] + GAP_PENALTY
            insert = dp[i][j-1] + GAP_PENALTY
            dp[i][j] = max(match, delete, insert)
            
    aligned_prof1 = ["" for _ in prof1]
    aligned_prof2 = ["" for _ in prof2]
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            col1 = [p[i-1] for p in prof1]
            col2 = [p[j-1] for p in prof2]
            if abs(dp[i][j] - (dp[i-1][j-1] + score_profile_profile(col1, col2))) < 1e-5:
                for k in range(len(prof1)): aligned_prof1[k] = prof1[k][i-1] + aligned_prof1[k]
                for k in range(len(prof2)): aligned_prof2[k] = prof2[k][j-1] + aligned_prof2[k]
                i -= 1; j -= 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i-1][j] + GAP_PENALTY)) < 1e-5:
            for k in range(len(prof1)): aligned_prof1[k] = prof1[k][i-1] + aligned_prof1[k]
            for k in range(len(prof2)): aligned_prof2[k] = "-" + aligned_prof2[k]
            i -= 1
        else:
            for k in range(len(prof1)): aligned_prof1[k] = "-" + aligned_prof1[k]
            for k in range(len(prof2)): aligned_prof2[k] = prof2[k][j-1] + aligned_prof2[k]
            j -= 1
    return aligned_prof1 + aligned_prof2

def build_matrix(sim_dict, active_ids, original_names):
    labels = []
    for cid in active_ids:
        if cid < len(original_names):
            labels.append(original_names[cid])
        else:
            labels.append(f"Node_{cid}")

    matrix = []
    for r in active_ids:
        row = []
        for c in active_ids:
            if r == c:
                row.append(0.0) 
            else:
                val = sim_dict.get((r, c), sim_dict.get((c, r), 0))
                row.append(round(val, 2))
        matrix.append(row)
    return matrix, labels

def build_full_newick(sim_dict, names):
    """Tính toán toàn bộ cấu trúc Cây Hướng Dẫn ngay từ đầu"""
    n = len(names)
    temp_sim = sim_dict.copy()
    temp_clusters = {i: names[i] for i in range(n)}
    temp_sizes = {i: 1 for i in range(n)}
    active_ids = list(range(n))
    next_id = n
    
    while len(active_ids) > 1:
        max_s = -float('inf')
        best_pair = (active_ids[0], active_ids[1])
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                id1, id2 = active_ids[i], active_ids[j]
                if temp_sim[(id1, id2)] > max_s:
                    max_s = temp_sim[(id1, id2)]; best_pair = (id1, id2)
        
        c1, c2 = best_pair
        new_id = next_id; next_id += 1
        new_newick = f"({temp_clusters[c1]},{temp_clusters[c2]})"
        temp_clusters[new_id] = new_newick
        
        new_size = temp_sizes[c1] + temp_sizes[c2]
        temp_sizes[new_id] = new_size
        
        active_ids.remove(c1); active_ids.remove(c2)
        for cid in active_ids:
            avg = (temp_sim[(c1, cid)] * temp_sizes[c1] + temp_sim[(c2, cid)] * temp_sizes[c2]) / new_size
            temp_sim[(new_id, cid)] = temp_sim[(cid, new_id)] = avg
        active_ids.append(new_id)
        
    return temp_clusters[active_ids[0]] + ";"

def create_tree_plot(newick_str):
    try:
        tree = Phylo.read(io.StringIO(newick_str), "newick")
        fig, ax = plt.subplots(figsize=(10, 4))
        # Ẩn trục tọa độ cho cây nhìn đẹp và gọn hơn
        ax.axis('off')
        Phylo.draw(tree, axes=ax, do_show=False)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None

def run_upgma_msa_generator(sequences, names):
    n = len(sequences)
    if n < 2: return
    start_time = time.time()
    original_names = names.copy()
    
    # 1. Xây dựng ma trận khoảng cách ban đầu
    sim = {}
    for i in range(n):
        for j in range(i+1, n):
            al_p1 = align_profiles([sequences[i]], [sequences[j]])
            score = sum(2 if c1==c2 else -1 for c1, c2 in zip(al_p1[0], al_p1[1]) if c1!='-' and c2!='-')
            sim[(i, j)] = sim[(j, i)] = score

    # 2. Xây dựng và vẽ toàn bộ Cây Hướng Dẫn ngay tại Step 0
    guide_tree_newick = build_full_newick(sim, original_names)
    tree_fig = create_tree_plot(guide_tree_newick)
    initial_matrix, initial_labels = build_matrix(sim, list(range(n)), original_names)

    # ĐẨY DATA STEP 0 LÊN GIAO DIỆN
    yield {
        "status": "card",
        "title": "Step 0: Distance Matrix & Guide Tree Construction",
        "description": "Calculated initial pairwise similarities and constructed the UPGMA Guide Tree. This tree dictates the progressive alignment order.",
        "matrix": initial_matrix, 
        "matrix_labels": initial_labels,
        "tree_fig": tree_fig,
        "tree_newick": guide_tree_newick,
        "current_msa": None
    }

    profiles_dict = {i: [sequences[i]] for i in range(n)}
    names_dict = {i: [names[i]] for i in range(n)}
    cluster_sizes = {i: 1 for i in range(n)}
    newick_strs = {i: names[i] for i in range(n)}
    
    next_id = n
    step_count = 1
    
    # 3. Vòng lặp gộp nhóm (Profile-Profile Alignment)
    while len(profiles_dict) > 1:
        active_ids = list(profiles_dict.keys())
        max_s = -float('inf')
        best_pair = (active_ids[0], active_ids[1])
        
        for x in range(len(active_ids)):
            for y in range(x+1, len(active_ids)):
                c1, c2 = active_ids[x], active_ids[y]
                if sim[(c1, c2)] > max_s:
                    max_s = sim[(c1, c2)]; best_pair = (c1, c2)
                    
        c1, c2 = best_pair
        new_id = next_id; next_id += 1
        
        merged_prof = align_profiles(profiles_dict[c1], profiles_dict[c2])
        merged_names = names_dict[c1] + names_dict[c2]
        newick_strs[new_id] = f"({newick_strs[c1]},{newick_strs[c2]})"
        
        profiles_dict[new_id] = merged_prof
        names_dict[new_id] = merged_names
        
        for cid in active_ids:
            if cid != c1 and cid != c2:
                avg = (sim[(c1, cid)] * cluster_sizes[c1] + sim[(c2, cid)] * cluster_sizes[c2]) / (cluster_sizes[c1] + cluster_sizes[c2])
                sim[(new_id, cid)] = sim[(cid, new_id)] = avg
        
        cluster_sizes[new_id] = cluster_sizes[c1] + cluster_sizes[c2]
        del profiles_dict[c1]; del profiles_dict[c2]
        
        # Lấy dữ liệu để xuất lên UI
        active_ids = list(profiles_dict.keys())
        step_matrix, step_labels = build_matrix(sim, active_ids, original_names)
        
        # Tút tát lại Tên để hiển thị đẹp hơn
        name1 = newick_strs[c1] if len(newick_strs[c1]) < 30 else f"Node_{c1}"
        name2 = newick_strs[c2] if len(newick_strs[c2]) < 30 else f"Node_{c2}"
        
        # ĐẨY DATA CÁC BƯỚC TIẾP THEO LÊN GIAO DIỆN
        yield {
            "status": "card",
            "title": f"Step {step_count}: Aligning {name1} and {name2}",
            "description": f"Merged into **Node_{new_id}**. The similarity matrix is updated using Average Linkage.",
            "matrix": step_matrix, 
            "matrix_labels": step_labels,
            "current_msa": merged_prof, 
            "current_names": merged_names
        }
        step_count += 1
            
    final_id = next_id - 1
    yield {
        "status": "done",
        "msa": profiles_dict[final_id],
        "names": names_dict[final_id],
        "execution_time": time.time() - start_time
    }