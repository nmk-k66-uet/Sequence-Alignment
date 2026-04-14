import time
import math
from collections import Counter

# Import the core DP alignment function from your progressive algorithm
from algorithms.progressive import align_profile_sequence

def get_kmer_vector(seq, k):
    """
    Description:
        Extracts features from a sequence by counting K-mer frequencies.
        This transforms a biological sequence into a mathematical vector.
    """
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    return Counter(kmers)

def cosine_similarity(vec1, vec2):
    """
    Description:
        Calculates the Cosine Similarity between two K-mer vectors.
        Returns a value between 0.0 (completely different) and 1.0 (identical).
    """
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([val**2 for val in vec1.values()])
    sum2 = sum([val**2 for val in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator: return 0.0
    return float(numerator) / denominator

def get_guide_tree_order(sequences, k):
    """
    Description:
        Unsupervised Machine Learning approach to find the optimal alignment order.
        Builds a distance matrix and uses a greedy agglomerative approach to cluster
        the most similar sequences first.
    """
    n = len(sequences)
    vectors = [get_kmer_vector(seq, k) for seq in sequences]
    
    # Compute Pairwise Similarity Matrix
    sim_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            sim = cosine_similarity(vectors[i], vectors[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim
            
    # Find the most similar pair to start the alignment
    max_sim = -1
    start_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if sim_matrix[i][j] > max_sim:
                max_sim = sim_matrix[i][j]
                start_pair = (i, j)
                
    order = [start_pair[0], start_pair[1]]
    unaligned = set(range(n)) - set(order)
    
    # Greedily build the rest of the order based on average cluster similarity
    while unaligned:
        best_next = -1
        best_avg_sim = -1
        
        for cand in unaligned:
            # Calculate average similarity to the already aligned group
            avg_sim = sum([sim_matrix[cand][aligned_idx] for aligned_idx in order]) / len(order)
            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_next = cand
                
        order.append(best_next)
        unaligned.remove(best_next)
        
    return order

def run_ml_clustering_msa_generator(sequences, seq_names, k=3):
    """
    Description:
        Generator function for the Hybrid ML MSA.
        1. Extracts features & builds guide tree.
        2. Progressively aligns following the ML-predicted order.
    """
    start_time = time.time()
    
    if len(sequences) < 2:
        yield {"status": "done", "msa": sequences, "names": seq_names, "metrics": {"time_seconds": 0, "num_sequences": len(sequences), "alignment_length": len(sequences[0]) if sequences else 0}}
        return

    # --- Phase 1: Machine Learning Feature Extraction & Clustering ---
    yield {"status": "running", "step": "ML Pipeline", "message": f"Extracting {k}-mer features and computing Cosine Similarity matrix...", "current_msa": []}
    time.sleep(0.5) # UI Buffer
    
    order = get_guide_tree_order(sequences, k)
    ordered_seqs = [sequences[i] for i in order]
    ordered_names = [seq_names[i] for i in order]
    
    yield {"status": "running", "step": "ML Pipeline", "message": f"Clustering complete. Optimal alignment order: {' ➔ '.join(ordered_names)}", "current_msa": []}
    time.sleep(1.0) # UI Buffer

    # --- Phase 2: Ordered Progressive Alignment ---
    current_msa = [ordered_seqs[0]]
    current_names = [ordered_names[0]]
    
    # Align the first pair
    yield {"status": "running", "step": 1, "message": f"Aligning root cluster: {ordered_names[0]} and {ordered_names[1]}...", "current_msa": current_msa, "current_names": current_names}
    aligned_profile, aligned_seq = align_profile_sequence(current_msa, ordered_seqs[1])
    current_msa = aligned_profile + [aligned_seq]
    current_names.append(ordered_names[1])
    
    # Progressively add the rest
    for i in range(2, len(ordered_seqs)):
        yield {"status": "running", "step": i, "message": f"Aligning next closest sequence: {ordered_names[i]}...", "current_msa": current_msa, "current_names": current_names}
        aligned_profile, aligned_seq = align_profile_sequence(current_msa, ordered_seqs[i])
        current_msa = aligned_profile + [aligned_seq]
        current_names.append(ordered_names[i])
            
    execution_time = time.time() - start_time
    
    yield {
        "status": "done",
        "msa": current_msa,
        "names": current_names,
        "metrics": {
            "time_seconds": round(execution_time, 4),
            "num_sequences": len(sequences),
            "alignment_length": len(current_msa[0])
        }
    }