import time
from collections import Counter
import io
import matplotlib.pyplot as plt
from Bio import Phylo

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
            labels.append(f"Cluster_{cid}")

    matrix = []
    for r in active_ids:
        row = []
        for c in active_ids:
            if r == c:
                row.append(0.0) # Đường chéo chính
            else:
                val = sim_dict.get((r, c), sim_dict.get((c, r), 0))
                row.append(round(val, 2))
        matrix.append(row)
    return matrix, labels

def build_full_newick(sim_dict, names):
    """Xây dựng cấu trúc cây Newick hoàn chỉnh dựa trên ma trận similarity ban đầu"""
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
    
def run_upgma_msa_generator(sequences, names):
    n = len(sequences)
    if n < 2: return
    start_time = time.time()

    profiles_dict = {i: [sequences[i]] for i in range(n)}
    names_dict = {i: [names[i]] for i in range(n)}
    cluster_sizes = {i: 1 for i in range(n)}
    newick_strs = {i: names[i] for i in range(n)}
    original_names = names.copy()
    
    sim = {}
    for i in range(n):
        for j in range(i+1, n):
            al_p1 = align_profiles([sequences[i]], [sequences[j]])
            score = sum(2 if c1==c2 else -1 for c1, c2 in zip(al_p1[0], al_p1[1]) if c1!='-' and c2!='-')
            sim[(i, j)] = sim[(j, i)] = score

    # Initial Matrix yield...
    initial_matrix, initial_labels = build_matrix(sim, list(profiles_dict.keys()), original_names)
    yield {
        "status": "card",
        "title": "Step 0: Initial Similarity Matrix",
        "matrix": initial_matrix, "matrix_labels": initial_labels,
        "current_msa": None
    }

    next_id = n
    while len(profiles_dict) > 1:
        # ... (Logic gom cụm UPGMA giữ nguyên) ...
        active_ids = list(profiles_dict.keys())
        max_s = -float('inf')
        best_pair = (active_ids[0], active_ids[1])
        for x in range(len(active_ids)):
            for y in range(x+1, len(active_ids)):
                if sim[(active_ids[x], active_ids[y])] > max_s:
                    max_s = sim[(active_ids[x], active_ids[y])]; best_pair = (active_ids[x], active_ids[y])
        
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

        active_ids = list(profiles_dict.keys())
        step_matrix, step_labels = build_matrix(sim, active_ids, original_names)
        yield {
            "status": "card",
            "title": f"Cluster Merge: Node {new_id}",
            "matrix": step_matrix, "matrix_labels": step_labels,
            "current_msa": merged_prof, "current_names": merged_names
        }
            
    final_id = next_id - 1
    final_newick = newick_strs[final_id] + ";"
    
    # TẠO FIGURE TRƯỚC KHI TRẢ VỀ
    tree_fig = create_tree_plot(final_newick)

    yield {
        "status": "done",
        "msa": profiles_dict[final_id],
        "names": names_dict[final_id],
        "tree_newick": final_newick,
        "tree_fig": tree_fig, # Gửi đối tượng Figure về cho app.py
        "execution_time": time.time() - start_time
    }