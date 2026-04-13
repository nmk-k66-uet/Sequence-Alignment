import time

def build_kmer_index(reference, w):
    """
    Cắt chuỗi Reference thành các k-mer liên tiếp và đưa vào Hash Table.
    """
    index = {}
    for i in range(len(reference) - w + 1):
        kmer = reference[i:i+w]
        if kmer not in index:
            index[kmer] = []
        index[kmer].append(i)
    return index

def extend_hit(query, reference, q_pos, r_pos, w):
    """
    Mở rộng kết quả sang hai bên mà không cho phép gap (Gapless Extension).
    """
    q_start, q_end = q_pos, q_pos + w
    r_start, r_end = r_pos, r_pos + w
    score = w # Điểm cơ bản là độ dài k-mer
    
    # Mở rộng sang phải
    while q_end < len(query) and r_end < len(reference) and query[q_end] == reference[r_end]:
        score += 1
        q_end += 1
        r_end += 1
        
    # Mở rộng sang trái
    while q_start > 0 and r_start > 0 and query[q_start-1] == reference[r_start-1]:
        score += 1
        q_start -= 1
        r_start -= 1
        
    return {
        "q_range": (q_start, q_end),
        "r_range": (r_start, r_end),
        "score": score,
        "alignment": query[q_start:q_end]
    }

def run_blast_generator(query, reference, w=11):
    """
    Hàm Generator: Quét, tìm k-mer, mở rộng và trả về trạng thái từng bước.
    """
    start_time = time.time()
    
    ref_index = build_kmer_index(reference, w)
    hits = []
    
    for query_pos in range(len(query) - w + 1):
        kmer = query[query_pos:query_pos+w]
        current_step_extensions = []
        
        if kmer in ref_index:
            for ref_pos in ref_index[kmer]:
                # Nếu tìm thấy k-mer, tiến hành mở rộng (extension)
                extended_hit = extend_hit(query, reference, query_pos, ref_pos, w)
                
                # Lọc trùng lặp đơn giản
                if extended_hit not in hits:
                    hits.append(extended_hit)
                    current_step_extensions.append(extended_hit)
                    
        # Yield trạng thái để Streamlit cập nhật UI
        yield {
            "status": "running",
            "step": query_pos + 1,
            "query_pos": query_pos,
            "current_kmer": kmer,
            "extensions_in_this_step": current_step_extensions,
            "total_hits_so_far": len(hits)
        }
        
    execution_time = time.time() - start_time
    
    yield {
        "status": "done",
        "final_hits": hits,
        "metrics": {
            "time_seconds": round(execution_time, 4),
            "total_hits": len(hits),
            "index_size": len(ref_index)
        }
    }