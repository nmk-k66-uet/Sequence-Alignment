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
    """
    Calculates the score between a new character and a column of existing aligned characters (profile).
    Takes the average score of comparing the new character to each character in the column.
    """
    score = 0
    for p_char in profile_col:
        score += score_char(p_char, char)
    return score / len(profile_col)

def align_profile_sequence(profile, seq):
    """
    Aligns a single sequence to an existing MSA profile using Needleman-Wunsch Dynamic Programming.
    - profile: List of already aligned strings (must be equal length).
    - seq: The new string to align.
    """
    m = len(profile[0])
    n = len(seq)
    
    # Initialize DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill base cases (gap penalties)
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + GAP_PENALTY
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + GAP_PENALTY
        
    # Fill DP table
    for i in range(1, m + 1):
        profile_col = [p[i-1] for p in profile]
        for j in range(1, n + 1):
            match = dp[i-1][j-1] + score_profile_char(profile_col, seq[j-1])
            delete = dp[i-1][j] + GAP_PENALTY
            insert = dp[i][j-1] + GAP_PENALTY
            dp[i][j] = max(match, delete, insert)
            
    # Traceback
    i, j = m, n
    aligned_profile = ["" for _ in profile]
    aligned_seq = ""
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            profile_col = [p[i-1] for p in profile]
            score_diag = dp[i-1][j-1] + score_profile_char(profile_col, seq[j-1])
            
            # Check if it came from diagonal (Match/Mismatch)
            if abs(dp[i][j] - score_diag) < 1e-5:
                for k in range(len(profile)):
                    aligned_profile[k] = profile[k][i-1] + aligned_profile[k]
                aligned_seq = seq[j-1] + aligned_seq
                i -= 1
                j -= 1
                continue
                
        # Check if it came from Top (Gap in new sequence)
        if i > 0 and abs(dp[i][j] - (dp[i-1][j] + GAP_PENALTY)) < 1e-5:
            for k in range(len(profile)):
                aligned_profile[k] = profile[k][i-1] + aligned_profile[k]
            aligned_seq = "-" + aligned_seq
            i -= 1
        # It came from Left (Gap in profile)
        else:
            for k in range(len(profile)):
                aligned_profile[k] = "-" + aligned_profile[k]
            aligned_seq = seq[j-1] + aligned_seq
            j -= 1
            
    return aligned_profile, aligned_seq

def run_progressive_msa_generator(sequences, seq_names):
    """
    Generator function that yields the step-by-step progress of the Multiple Sequence Alignment.
    """
    start_time = time.time()
    
    if len(sequences) < 2:
        yield {"status": "done", "msa": sequences, "names": seq_names, "metrics": {"time_seconds": 0, "num_sequences": len(sequences), "alignment_length": len(sequences[0]) if sequences else 0}}
        return
        
    # Step 1: Initialize with the first two sequences
    yield {"status": "running", "step": 1, "message": f"Aligning {seq_names[0]} and {seq_names[1]}...", "current_msa": []}
    
    # Treat the first sequence as a profile of size 1
    profile = [sequences[0]]
    aligned_profile, aligned_seq = align_profile_sequence(profile, sequences[1])
    
    current_msa = aligned_profile + [aligned_seq]
    current_names = [seq_names[0], seq_names[1]]
    
    yield {"status": "running", "step": 1, "message": f"Successfully aligned {seq_names[0]} and {seq_names[1]}.", "current_msa": current_msa, "current_names": current_names}
    
    # Step 2: Progressively add the remaining sequences to the profile
    for i in range(2, len(sequences)):
        yield {"status": "running", "step": i, "message": f"Adding {seq_names[i]} to the existing alignment profile...", "current_msa": current_msa, "current_names": current_names}
        
        aligned_profile, aligned_seq = align_profile_sequence(current_msa, sequences[i])
        current_msa = aligned_profile + [aligned_seq]
        current_names.append(seq_names[i])
        
        yield {"status": "running", "step": i, "message": f"Successfully integrated {seq_names[i]}.", "current_msa": current_msa, "current_names": current_names}
        
    execution_time = time.time() - start_time
    
    # Final Result Yield
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