import time

def apply_spaced_seed(sequence, start_index, seed_model):
    """
    Apply a spaced seed (mask pattern) to a sequence at a specific position.
    Extracts only positions marked with '1' in the seed model, ignoring '0' positions.
    This selective matching allows PSSM-like flexibility while maintaining seed specificity,
    improving sensitivity for distant homologs compared to contiguous k-mer matching.
    """
    if start_index + len(seed_model) > len(sequence):
        return None
        
    masked_string = "".join(
        [sequence[start_index + i] for i, bit in enumerate(seed_model) if bit == '1']
    )
    return masked_string

def build_reference_index(reference, seed_model):
    """
    Construct a hash table (index) for the reference sequence using a spaced seed pattern.
    For each position in the reference, applies the spaced seed and stores the position
    in a hash table indexed by the masked seed. This enables rapid lookup during query scanning.
    """
    index_table = {}
    for i in range(len(reference) - len(seed_model) + 1):
        seed_key = apply_spaced_seed(reference, i, seed_model)
        if seed_key:
            if seed_key not in index_table:
                index_table[seed_key] = []
            index_table[seed_key].append(i)
    return index_table

def run_patternhunter_generator(query, reference, seed_model="111010010100110111"):
    """
    Generator function implementing the PatternHunter algorithm.
    Scans the query sequence for matches of the spaced seed pattern in the reference index,
    and yields status updates at each step for real-time visualization.
    The spaced seed pattern improves sensitivity compared to contiguous k-mers,
    enabling detection of more distant sequence homologies.
    Default seed model: "111010010100110111" (weight=11, span=18)
    """
    start_time = time.time()
    
    # Step 1: Build the reference index (typically very fast)
    ref_index = build_reference_index(reference, seed_model)
    
    hits = []
    
    # Step 2: Scan the query sequence
    for query_pos in range(len(query) - len(seed_model) + 1):
        query_seed = apply_spaced_seed(query, query_pos, seed_model)
        current_step_hits = []
        
        # Hash table lookup O(1)
        if query_seed and query_seed in ref_index:
            for ref_pos in ref_index[query_seed]:
                hit_info = {
                    "query_pos": query_pos,
                    "ref_pos": ref_pos,
                    "matched_seed": query_seed
                }
                hits.append(hit_info)
                current_step_hits.append(hit_info)
                
        # PRIMARY FOCUS: Yield current status for Streamlit to render the UI
        yield {
            "status": "running",
            "step": query_pos + 1,
            "query_pos": query_pos,
            "current_seed": query_seed,
            "hits_in_this_step": current_step_hits,
            "total_hits_so_far": len(hits)
        }
        
    execution_time = time.time() - start_time
    
    # Yield final results
    yield {
        "status": "done",
        "final_hits": hits,
        "metrics": {
            "time_seconds": round(execution_time, 4),
            "total_hits": len(hits),
            "index_size": len(ref_index)
        }
    }