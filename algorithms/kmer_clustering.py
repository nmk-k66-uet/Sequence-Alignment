import time
import math
from collections import Counter
from algorithms.progressive import align_profile_sequence, build_guide_tree_visualization, create_tree_plotly

def get_kmer_vector(seq, k):
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    return Counter(kmers)

def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([val**2 for val in vec1.values()])
    sum2 = sum([val**2 for val in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return float(numerator) / denominator if denominator else 0.0

def get_guide_tree_order(sequences, k):
    n = len(sequences)
    vectors = [get_kmer_vector(seq, k) for seq in sequences]
    
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            sim = cosine_similarity(vectors[i], vectors[j])
            sim_matrix[i][j] = sim_matrix[j][i] = sim
            
    max_sim = -1
    start_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if sim_matrix[i][j] > max_sim:
                max_sim = sim_matrix[i][j]
                start_pair = (i, j)
                
    order = [start_pair[0], start_pair[1]]
    unaligned = set(range(n)) - set(order)
    
    while unaligned:
        best_next, best_avg_sim = -1, -1
        for cand in unaligned:
            avg_sim = sum([sim_matrix[cand][idx] for idx in order]) / len(order)
            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_next = cand
        order.append(best_next)
        unaligned.remove(best_next)
        
    return order

def run_ml_clustering_msa_generator(sequences, seq_names, k=3):
    start_time = time.time()
    
    if len(sequences) < 2:
        yield {"status": "done", "msa": sequences, "names": seq_names, "metrics": {"time_seconds": 0, "num_sequences": len(sequences), "alignment_length": len(sequences[0])}}
        return

    # Build guide tree based on k-mer similarity clustering
    order = get_guide_tree_order(sequences, k)
    ordered_seqs = [sequences[i] for i in order]
    ordered_names = [seq_names[i] for i in order]
    
    # Create tree visualization
    tree_nodes, tree_edges = build_guide_tree_visualization(ordered_names)
    tree_fig = create_tree_plotly(tree_nodes, tree_edges) if tree_nodes else None
    
    yield {
        "status": "card",
        "title": "1. K-mer Feature Extraction & ML-Based Clustering",
        "description": f"Transform sequences into {k}-mer frequency vectors and apply machine learning-based clustering using Cosine Similarity. This approach captures sequence composition patterns and generates a guide tree by grouping sequences with similar k-mer distributions. The resulting tree optimizes alignment order by placing the most similar sequences adjacent to each other.\n\n**Optimized alignment order (ML Guide Tree):** `{' ➔ '.join(ordered_names)}`",
        "tree_figure": tree_fig
    }

    current_msa = [ordered_seqs[0]]
    current_names = [ordered_names[0]]
    
    # Progressive alignment phase: Add sequences one by one to the MSA
    for i in range(1, len(ordered_seqs)):
        aligned_profile, aligned_seq = align_profile_sequence(current_msa, ordered_seqs[i])
        current_msa = aligned_profile + [aligned_seq]
        current_names.append(ordered_names[i])

        if i > 1:    
            yield {
                "status": "card",
                "title": f"2.{i}. Progressive Alignment: Adding {ordered_names[i]}",
                "description": f"Integrate sequence **{ordered_names[i]}** into the current MSA using dynamic programming profile-to-sequence alignment.",
                "current_msa": current_msa,
                "current_names": current_names
            }
        else:    
            yield {
                "status": "card",
                "title": f"2.{i}. Progressive Alignment: Adding {ordered_names[i]}",
                "description": f"Integrate sequence **{ordered_names[i]}** into the current MSA using dynamic programming profile-to-sequence alignment. The ML-guided clustering tree ensures that sequences with similar compositional features are aligned in optimal order, improving alignment quality and reducing computational complexity.",
                "current_msa": current_msa,
                "current_names": current_names
            }
            
    execution_time = time.time() - start_time
    yield {
        "status": "done",
        "msa": current_msa,
        "names": current_names,
        "metrics": {"time_seconds": round(execution_time, 4), "num_sequences": len(sequences), "alignment_length": len(current_msa[0])}
    }