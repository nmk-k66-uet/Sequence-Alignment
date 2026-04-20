import time
from algorithms.progressive import align_profile_sequence

def strip_gaps(seq):
    return seq.replace("-", "")

def run_iterative_msa_generator(sequences, seq_names, iterations=2):
    start_time = time.time()
    
    if len(sequences) < 2:
        yield {"status": "done", "msa": sequences, "names": seq_names, "metrics": {"time_seconds": 0, "num_sequences": len(sequences), "alignment_length": len(sequences[0])}}
        return

    # --- Phase 1: Initial Progressive Alignment ---
    current_msa = [sequences[0]]
    current_names = [seq_names[0]]
    for i in range(1, len(sequences)):
        aligned_profile, aligned_seq = align_profile_sequence(current_msa, sequences[i])
        current_msa = aligned_profile + [aligned_seq]
        current_names.append(seq_names[i])
        
    yield {
        "status": "card",
        "title": "Phase 1: Initial Progressive MSA Construction",
        "description": "Generate an initial multiple sequence alignment using the progressive alignment method. This rapid construction produces a foundational alignment structure that captures the primary homology relationships between sequences. While the initial alignment provides a reasonable starting point, it may be suboptimal due to the greedy nature of progressive alignment. This initial MSA serves as the foundation for iterative refinement in subsequent phases.",
        "current_msa": current_msa,
        "current_names": current_names
    }

    # --- Phase 2: Iterative Refinement ---
    for cycle in range(iterations):
        for i in range(len(current_msa)):
            target_name = current_names[i]
            target_seq = strip_gaps(current_msa[i])
            
            # Remove the target sequence from the current MSA
            remaining_profile = current_msa[:i] + current_msa[i+1:]
            
            # Re-align the target sequence against the remaining profile
            aligned_profile, aligned_seq = align_profile_sequence(remaining_profile, target_seq)
            
            # Merge back to the correct position
            current_msa = aligned_profile[:i] + [aligned_seq] + aligned_profile[i:]
            
            if i > 0:
                yield {
                    "status": "card",
                    "title": f"Phase 2 - Iteration {cycle+1}: Refining Sequence {target_name}",
                    "description": f"Remove all gaps from **{target_name}** and re-align the ungapped sequence against the profile containing the remaining {len(remaining_profile)} sequences.",
                    "current_msa": current_msa,
                    "current_names": current_names
                }
            else:
                yield {
                    "status": "card",
                    "title": f"Phase 2 - Iteration {cycle+1}: Refining Sequence {target_name}",
                    "description": f"Remove all gaps from **{target_name}** and re-align the ungapped sequence against the profile containing the remaining {len(remaining_profile)} sequences. This iterative refinement process allows sequences previously positioned based on suboptimal early decisions to find better alignment positions. Multiple iterations enable the algorithm to escape local optima and improve overall MSA quality through dynamic repositioning.",
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