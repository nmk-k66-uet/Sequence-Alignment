import time

def apply_spaced_seed(sequence, start_index, seed_model):
    """
    Áp dụng mặt nạ (spaced seed) lên chuỗi tại một vị trí cụ thể.
    """
    if start_index + len(seed_model) > len(sequence):
        return None
        
    masked_string = "".join(
        [sequence[start_index + i] for i, bit in enumerate(seed_model) if bit == '1']
    )
    return masked_string

def build_reference_index(reference, seed_model):
    """
    Xây dựng Hash Table (Index) cho chuỗi Reference.
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
    Hàm Generator: Quét chuỗi Query và trả về trạng thái từng bước (yield).
    """
    start_time = time.time()
    
    # Bước 1: Xây dựng Index (Thường diễn ra rất nhanh)
    ref_index = build_reference_index(reference, seed_model)
    
    hits = []
    
    # Bước 2: Quét chuỗi Query
    for query_pos in range(len(query) - len(seed_model) + 1):
        query_seed = apply_spaced_seed(query, query_pos, seed_model)
        current_step_hits = []
        
        # Tra cứu hash table O(1)
        if query_seed and query_seed in ref_index:
            for ref_pos in ref_index[query_seed]:
                hit_info = {
                    "query_pos": query_pos,
                    "ref_pos": ref_pos,
                    "matched_seed": query_seed
                }
                hits.append(hit_info)
                current_step_hits.append(hit_info)
                
        # TRỌNG TÂM: Trả về trạng thái hiện tại để Streamlit vẽ UI
        yield {
            "status": "running",
            "step": query_pos + 1,
            "query_pos": query_pos,
            "current_seed": query_seed,
            "hits_in_this_step": current_step_hits,
            "total_hits_so_far": len(hits)
        }
        
    execution_time = time.time() - start_time
    
    # Yield kết quả cuối cùng
    yield {
        "status": "done",
        "final_hits": hits,
        "metrics": {
            "time_seconds": round(execution_time, 4),
            "total_hits": len(hits),
            "index_size": len(ref_index)
        }
    }