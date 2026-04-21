import time
import pandas as pd
import plotly.graph_objects as go
from algorithms.progressive import align_profile_sequence 

def strip_gaps(seq):
    return seq.replace("-", "")

def run_iterative_msa_generator(sequences, seq_names, iterations=2):
    start_time = time.time()
    n = len(sequences)
    
    if n < 2:
        yield {"status": "done", "msa": sequences, "names": seq_names, "metrics": {"time_seconds": 0, "num_sequences": n, "alignment_length": len(sequences[0])}}
        return

    # ---------------------------------------------------------
    # GIAI ĐOẠN 1: TẠO BẢN MSA BAN ĐẦU BẰNG PROGRESSIVE ALIGNMENT
    # ---------------------------------------------------------
    # 1a. Tính ma trận khoảng cách chính xác
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 100.0
            elif i < j:
                aligned_p, aligned_s = align_profile_sequence([sequences[i]], sequences[j])
                s1, s2 = aligned_p[0], aligned_s
                matches = sum(1 for a, b in zip(s1, s2) if a == b and a != '-')
                identity = round((matches / len(s1)) * 100, 1)
                matrix[i][j] = identity
                matrix[j][i] = identity
    
    # 1b. Tạo Guide Tree bằng thuật toán tham lam (Greedy)
    max_id = -1
    best_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i][j] > max_id:
                max_id = matrix[i][j]
                best_pair = (i, j)
    
    order_indices = list(best_pair)
    unaligned = set(range(n)) - set(best_pair)
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
        
    ordered_sequences = [sequences[i] for i in order_indices]
    ordered_names = [seq_names[i] for i in order_indices]

    # 1c. Thực thi gióng hàng tăng dần ban đầu
    current_msa = [ordered_sequences[0]]
    current_names = [ordered_names[0]]
    aligned_profile, aligned_seq = align_profile_sequence(current_msa, ordered_sequences[1])
    current_msa = aligned_profile + [aligned_seq]
    current_names.append(ordered_names[1])
    
    for i in range(2, n):
        aligned_profile, aligned_seq = align_profile_sequence(current_msa, ordered_sequences[i])
        current_msa = aligned_profile + [aligned_seq]
        current_names.append(ordered_names[i])

    yield {
        "status": "card",
        "title": "Phase 1: Initial Progressive MSA Construction",
        "description": "Generate an initial multiple sequence alignment using the progressive alignment method. This rapid construction produces a foundational alignment structure that captures the primary homology relationships between sequences.",
        "current_msa": current_msa,
        "current_names": current_names
    }

    # ---------------------------------------------------------
    # GIAI ĐOẠN 2: TINH CHỈNH LẶP LẠI (ITERATIVE REFINEMENT)
    # ---------------------------------------------------------
    for cycle in range(iterations):
        # Lặp qua từng trình tự trong bản gióng hàng hiện tại
        for i in range(n):
            target_name = current_names[i]
            target_seq_with_gaps = current_msa[i]
            
            # Rút trình tự mục tiêu ra khỏi Profile chung
            remaining_profile = current_msa[:i] + current_msa[i+1:]
            
            # BƯỚC QUAN TRỌNG: Dọn dẹp các cột trống (Gaps-only columns)
            # Nếu không làm bước này, bản gióng hàng sẽ bị phình to vô hạn và không thoát được bẫy Gap.
            clean_profile = [""] * len(remaining_profile)
            num_columns = len(remaining_profile[0])
            for col in range(num_columns):
                # Kiểm tra xem cột này có ký tự nào khác '-' không
                column_chars = [p[col] for p in remaining_profile]
                if any(c != '-' for c in column_chars):
                    for r in range(len(remaining_profile)):
                        clean_profile[r] += remaining_profile[r][col]
            
            # Xóa các gap khỏi trình tự mục tiêu để tạo thành chuỗi thô (raw sequence)
            ungapped_target = strip_gaps(target_seq_with_gaps)
            
            # Gióng hàng lại chuỗi thô với Profile đã được dọn dẹp
            aligned_profile, aligned_target = align_profile_sequence(clean_profile, ungapped_target)
            
            # Cập nhật lại bản MSA tổng thể với kết quả mới
            current_msa = aligned_profile[:i] + [aligned_target] + aligned_profile[i:]
            
            yield {
                "status": "card",
                "title": f"Phase 2 - Iteration {cycle+1}: Refining Sequence {target_name}",
                "description": f"Remove all gaps from **{target_name}** and re-align the ungapped sequence against the profile containing the remaining {len(remaining_profile)} sequences. The profile is cleaned from empty gap columns before realignment to escape the 'Once a gap, always a gap' trap.",
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