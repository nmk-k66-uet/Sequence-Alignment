import time

# --- Scoring System ---
MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_PENALTY = -1

def score_char(c1, c2):
    """Calculates the alignment score between two characters."""
    if c1 == '-' and c2 == '-': return 0
    if c1 == '-' or c2 == '-': return GAP_PENALTY
    return MATCH_SCORE if c1 == c2 else MISMATCH_SCORE

def score_profile_char(profile_col, char):
    """Calculates the average score of aligning a character to a profile column."""
    if not profile_col: return 0
    score = sum(score_char(p_char, char) for p_char in profile_col)
    return score / len(profile_col)

def strip_gaps(seq):
    """Removes gap characters from a sequence."""
    return seq.replace("-", "")

def align_profile_sequence(profile, seq):
    """
    Aligns a single sequence (without gaps) to an existing MSA profile 
    using Needleman-Wunsch DP.
    """
    m = len(profile[0])
    n = len(seq)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + GAP_PENALTY
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + GAP_PENALTY
        
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
                for k in range(len(profile)):
                    aligned_profile[k] = profile[k][i-1] + aligned_profile[k]
                aligned_seq = seq[j-1] + aligned_seq
                i -= 1
                j -= 1
                continue
                
        if i > 0 and abs(dp[i][j] - (dp[i-1][j] + GAP_PENALTY)) < 1e-5:
            for k in range(len(profile)):
                aligned_profile[k] = profile[k][i-1] + aligned_profile[k]
            aligned_seq = "-" + aligned_seq
            i -= 1
        else:
            for k in range(len(profile)):
                aligned_profile[k] = "-" + aligned_profile[k]
            aligned_seq = seq[j-1] + aligned_seq
            j -= 1
            
    return aligned_profile, aligned_seq

def run_iterative_msa_generator(sequences, seq_names, iterations=2):
    """
    Description:
        Generator function for Iterative Refinement MSA.
        1. Generates initial alignment (Progressive approach).
        2. Iteratively refines by taking out one sequence, stripping gaps, 
           and realigning it to the remaining profile.
           
    Inputs:
        - sequences (list of str): The raw DNA/Protein sequences.
        - seq_names (list of str): Sequence identifiers.
        - iterations (int): Number of refinement cycles.
        
    Outputs:
        - dict: State updates to be yielded to the UI.
    """
    start_time = time.time()
    
    if len(sequences) < 2:
        yield {"status": "done", "msa": sequences, "names": seq_names, "metrics": {"time_seconds": 0, "num_sequences": len(sequences), "alignment_length": len(sequences[0]) if sequences else 0}}
        return

    # --- Phase 1: Initial Progressive Alignment ---
    yield {"status": "running", "step": "Initial", "message": "Building initial progressive alignment...", "current_msa": []}
    
    current_msa = [sequences[0]]
    current_names = [seq_names[0]]
    
    for i in range(1, len(sequences)):
        aligned_profile, aligned_seq = align_profile_sequence(current_msa, sequences[i])
        current_msa = aligned_profile + [aligned_seq]
        current_names.append(seq_names[i])
        
    yield {"status": "running", "step": "Initial", "message": "Initial progressive alignment completed.", "current_msa": current_msa, "current_names": current_names}

    # --- Phase 2: Iterative Refinement ---
    step_count = 1
    for cycle in range(iterations):
        for i in range(len(current_msa)):
            target_name = current_names[i]
            yield {"status": "running", "step": f"Cycle {cycle+1}", "message": f"Refining {target_name}...", "current_msa": current_msa, "current_names": current_names}
            
            # Extract target sequence and strip gaps
            target_seq = strip_gaps(current_msa[i])
            
            # Create a profile without the target sequence
            remaining_profile = current_msa[:i] + current_msa[i+1:]
            
            # Realign target to the remaining profile
            aligned_profile, aligned_seq = align_profile_sequence(remaining_profile, target_seq)
            
            # Reconstruct the MSA
            current_msa = aligned_profile[:i] + [aligned_seq] + aligned_profile[i:]
            
            yield {"status": "running", "step": f"Cycle {cycle+1}", "message": f"Realigned {target_name} to the profile.", "current_msa": current_msa, "current_names": current_names}
            step_count += 1
            
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