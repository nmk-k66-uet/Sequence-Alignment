import streamlit as st
import os
import time
import tracemalloc
import math
import pandas as pd
import altair as alt
from Bio import SeqIO
from collections import Counter

# Import algorithms directly
from algorithms.patternhunter import run_patternhunter_generator
from algorithms.blast import run_blast_generator
from algorithms.progressive import run_progressive_msa_generator
from algorithms.iterative import run_iterative_msa_generator

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Sequence Alignment Simulator", layout="wide")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_e_value(score, query_len, ref_len):
    """
    Description:
        Simulates the calculation of the E-value (Expect value) for an alignment hit.
        Uses the formula: E = K * m * n * exp(-lambda * S).
        
    Inputs:
        - score (int or float): The raw alignment score.
        - query_len (int): Length of the query sequence.
        - ref_len (int): Length of the database reference sequence.
        
    Outputs:
        - float: The calculated E-value.
    """
    K = 0.1
    lam = 0.3
    return K * query_len * ref_len * math.exp(-lam * score)

def render_alignment(q_str, r_str, organism_name="", file_name=""):
    """
    Description:
        Generates an HTML string to visualize a pairwise sequence alignment.
        Applies color coding (Green: Match, Gray: Gap, Red: Mismatch) and 
        displays the source organism and file name.
        
    Inputs:
        - q_str (str): Query aligned substring.
        - r_str (str): Reference aligned substring.
        - organism_name (str): Name of the organism database (optional).
        - file_name (str): Name of the fasta file (optional).
        
    Outputs:
        - str: HTML formatted string ready for Streamlit rendering.
    """
    html = '<div style="font-family: Courier, monospace; font-size: 16px; background-color: rgba(128, 128, 128, 0.1); padding: 10px; border-radius: 5px; overflow-x: auto; white-space: pre; margin-bottom: 5px;">'
    
    if organism_name or file_name:
        html += f'<div style="font-size: 14px; font-family: sans-serif; color: #495057; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 10px;"><b>Organism:</b> {organism_name} &nbsp;|&nbsp; <b>File:</b> {file_name}</div>'
        
    q_html = r_html = m_html = ""
    
    for q, r in zip(q_str, r_str):
        if q == r and q != '-':
            color, m_char = "#28a745", "|"  # Match: Green
        elif q == '-' or r == '-':
            color, m_char = "#6c757d", " "  # Gap: Gray
        else:
            color, m_char = "#dc3545", "x"  # Mismatch: Red
            
        q_html += f'<span style="color: {color}; font-weight: bold;">{q}</span>'
        r_html += f'<span style="color: {color}; font-weight: bold;">{r}</span>'
        m_html += f'<span style="color: {color};">{m_char}</span>'
        
    html += f'<div>Query: {q_html}</div>'
    html += f'<div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{m_html}</div>'
    html += f'<div>Ref&nbsp;&nbsp;: {r_html}</div></div>'
    return html

def render_msa(aligned_sequences, seq_names):
    """
    Description:
        Generates an HTML string to visualize a Multiple Sequence Alignment (MSA).
        Calculates conservation ratio for each column to determine the background color.
        
    Inputs:
        - aligned_sequences (list of str): List of aligned sequences (equal lengths).
        - seq_names (list of str): List of sequence names/identifiers.
        
    Outputs:
        - str: HTML formatted string containing the MSA visualization.
    """
    if not aligned_sequences:
        return "<div>No alignment data available</div>"

    seq_length = len(aligned_sequences[0])
    num_seqs = len(aligned_sequences)
    
    html = '<div style="font-family: Courier, monospace; font-size: 14px; background-color: rgba(128, 128, 128, 0.05); padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre; border: 1px solid #ddd;">'
    
    column_colors = []
    consensus_str = ""
    
    # Calculate conservation for each column
    for col_idx in range(seq_length):
        col_chars = [seq[col_idx] for seq in aligned_sequences if seq[col_idx] != '-']
        if not col_chars:
            column_colors.append("transparent")
            consensus_str += "-"
            continue
            
        most_common_char, count = Counter(col_chars).most_common(1)[0]
        ratio = count / num_seqs
        
        if ratio == 1.0:
            column_colors.append("rgba(40, 167, 69, 0.4)") # 100% Match: Green
            consensus_str += most_common_char
        elif ratio >= 0.6:
            column_colors.append("rgba(255, 193, 7, 0.3)") # >60% Match: Yellow
            consensus_str += most_common_char.lower()
        else:
            column_colors.append("transparent")
            consensus_str += "."
            
    max_name_len = max([len(name) for name in seq_names] + [len("Consensus")])
    
    # Render individual sequences
    for seq_name, seq in zip(seq_names, aligned_sequences):
        padded_name = seq_name.ljust(max_name_len + 2)
        html += f'<div style="margin-bottom: 2px;"><strong>{padded_name}</strong> '
        for i, char in enumerate(seq):
            html += f'<span style="background-color: {column_colors[i]};">{char}</span>'
        html += '</div>'
        
    # Render Consensus string
    html += '<hr style="margin: 8px 0; border-color: #ccc;">'
    padded_consensus = "Consensus".ljust(max_name_len + 2)
    html += f'<div style="color: #666;"><strong>{padded_consensus}</strong> '
    for char in consensus_str:
        html += f'<span>{char}</span>'
    html += '</div></div>'
    
    return html

@st.cache_data
def get_available_organisms(data_dir="Data"):
    """
    Description:
        Scans the local directory to find subdirectories representing organism databases.
        
    Inputs:
        - data_dir (str): The root path containing data folders. Default is "Data".
        
    Outputs:
        - list: A list of strings representing organism names.
    """
    if not os.path.exists(data_dir):
        os.makedirs(os.path.join(data_dir, "Demo_Organism"))
        return ["Demo_Organism"]
    organisms = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return organisms if organisms else ["No data available"]

@st.cache_data
def load_fasta(organism_name, data_dir="Data"):
    """
    Description:
        Parses the first FASTA file found inside the specified organism's folder.
        
    Inputs:
        - organism_name (str): The folder name of the selected organism.
        - data_dir (str): The root path containing data folders. Default is "Data".
        
    Outputs:
        - tuple: (Parsed sequence as uppercase string, File name as string).
                 Returns mock data if the file or directory does not exist.
    """
    org_path = os.path.join(data_dir, organism_name)
    if not os.path.exists(org_path):
        return "ACGT" * 50, "mock_data.fasta"
    
    for file in os.listdir(org_path):
        if file.endswith((".fasta", ".fa")):
            file_path = os.path.join(org_path, file)
            record = next(SeqIO.parse(file_path, "fasta"))
            return str(record.seq).upper(), file
            
    return "ACGT" * 100, "mock_data.fasta"

def parse_multi_fasta_text(fasta_text):
    """
    Description:
        Parses raw text in Multi-FASTA format into names and sequences.
        
    Inputs:
        - fasta_text (str): The raw text inputted by the user.
        
    Outputs:
        - tuple: (List of sequence names, List of nucleotide sequences).
    """
    names, seqs = [], []
    curr_name, curr_seq = "", []
    for line in fasta_text.strip().split('\n'):
        line = line.strip()
        if not line: continue
        if line.startswith('>'):
            if curr_name:
                names.append(curr_name)
                seqs.append("".join(curr_seq))
            curr_name = line[1:]
            curr_seq = []
        else:
            curr_seq.append(line.upper())
    if curr_name:
        names.append(curr_name)
        seqs.append("".join(curr_seq))
    return names, seqs


# ==========================================
# MAIN APPLICATION UI
# ==========================================
st.title("Sequence Alignment Simulator")

# --- SIDEBAR: MODE NAVIGATION ---
st.sidebar.title("App Navigation")
app_mode = st.sidebar.radio(
    "Select Analysis Mode:",
    ["Pairwise Alignment (1 vs 1)", "Multiple Sequence Alignment (MSA)"]
)
st.sidebar.divider()

# ==========================================
# MODE 1: PAIRWISE ALIGNMENT
# ==========================================
if app_mode == "Pairwise Alignment (1 vs 1)":
    
    st.markdown("### 1. Data Preparation")
    col_data1, col_data2 = st.columns([1, 2])

    with col_data1:
        organism = st.selectbox("Select Database (Organism):", get_available_organisms())
        reference_seq, ref_filename = load_fasta(organism)
        if len(reference_seq) > 50:
            st.caption(f"Loaded Database: {len(reference_seq)} bp from {ref_filename}")
        else:
            st.warning("Using mock data.")
            
    with col_data2:
        query_seq = st.text_input("Enter Query Sequence (DNA):", "AGCTAGCAAGCTGATC").upper()

    st.divider()

    st.markdown("### 2. Analysis & Alignment")
    tab1, tab2 = st.tabs(["Single Run", "Performance Comparison"])

    # ====== TAB 1: SINGLE RUN ======
    with tab1:
        st.markdown("This mode helps you understand the algorithms by visualizing them step-by-step.")
        col_t1_left, col_t1_right = st.columns([1, 2.5])
        
        with col_t1_left:
            st.markdown("**Parameter Settings**")
            algo_choice = st.selectbox("Select Algorithm:", ("PatternHunter", "BLAST"), key="t1_algo")
            
            if algo_choice == "PatternHunter":
                seed_input = st.text_input("Spaced Seed Model:", value="111010010100110111", key="t1_seed")
            else:
                w_input = st.number_input("K-mer Size (w):", min_value=2, max_value=50, value=11, key="t1_w")
                
            btn_run_single = st.button("Start Scanning", width='content')

        with col_t1_right:
            st.markdown("**Visualization Screen**")
            if btn_run_single:
                viz_container = st.empty() 
                tracemalloc.start()
                
                if algo_choice == "PatternHunter":
                    generator = run_patternhunter_generator(query_seq, reference_seq, seed_model=seed_input) 
                else:
                    generator = run_blast_generator(query_seq, reference_seq, w=w_input)
                    
                for step_data in generator:
                    if step_data["status"] == "running":
                        with viz_container.container():
                            st.markdown(f"**Scanning (Step {step_data['step']}/{len(query_seq)})**")
                            if algo_choice == "PatternHunter":
                                st.code(f"Query Position: {step_data['query_pos']}\nExtracted Seed: {step_data['current_seed']}")
                            else:
                                st.code(f"Query Position: {step_data['query_pos']}\nCurrent K-mer: {step_data['current_kmer']}")
                            st.progress(min(step_data['step'] / len(query_seq), 1.0))
                            st.info(f"Found: {step_data['total_hits_so_far']} Hits")
                        time.sleep(0.1) 
                        
                    elif step_data["status"] == "done":
                        _, peak_mem = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        peak_mem_kb = peak_mem / 1024
                        
                        viz_container.empty()
                        st.success("Search completed successfully!")
                        
                        metrics = step_data["metrics"]
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Execution Time", f"{metrics['time_seconds']}s")
                        m2.metric("Total Hits", metrics['total_hits'])
                        m3.metric("Index Size", metrics['index_size'])
                        m4.metric("Peak Memory", f"{peak_mem_kb:.2f} KB")
                        
                        if step_data["final_hits"]:
                            st.subheader("Result Details")
                            for i, hit in enumerate(step_data["final_hits"]):
                                st.markdown(f"**Result #{i+1}**")
                                if algo_choice == "BLAST":
                                    q_start, q_end = hit["q_range"]
                                    r_start, r_end = hit["r_range"]
                                    score = hit['score']
                                    q_str = query_seq[q_start:q_end]
                                    r_str = reference_seq[r_start:r_end]
                                    e_val = calculate_e_value(score, len(query_seq), len(reference_seq))
                                    
                                    st.markdown(render_alignment(q_str, r_str, organism_name=organism, file_name=ref_filename), unsafe_allow_html=True)
                                    st.caption(f"Score: {score} | E-value: {e_val:.2e} | Query Range: [{q_start}:{q_end}]")
                                    
                                elif algo_choice == "PatternHunter":
                                    seed_len = len(seed_input)
                                    q_start = hit["query_pos"]
                                    r_start = hit["ref_pos"]
                                    score = seed_len 
                                    q_str = query_seq[q_start:q_start+seed_len]
                                    r_str = reference_seq[r_start:r_start+seed_len]
                                    e_val = calculate_e_value(score, len(query_seq), len(reference_seq))
                                    
                                    st.markdown(render_alignment(q_str, r_str, organism_name=organism, file_name=ref_filename), unsafe_allow_html=True)
                                    st.caption(f"E-value: {e_val:.2e} | Query Position: {q_start} | Seed Model: {hit['matched_seed']}")
                                st.divider()
                        else:
                            st.warning("No matching results found.")

    # ====== TAB 2: COMPARISON ======
    with tab2:
        st.markdown("Set parameters and race two algorithms to compare performance.")
        col_algoA, col_algoB = st.columns(2)
        
        with col_algoA:
            st.info("**Algorithm A: PatternHunter**")
            t2_ph_seed = st.text_input("Spaced Seed:", value="111010010100110111", key="t2_seed")
        with col_algoB:
            st.warning("**Algorithm B: BLAST**")
            t2_blast_w = st.number_input("K-mer Size (w):", min_value=2, max_value=50, value=11, key="t2_w")
            
        st.markdown("<br>", unsafe_allow_html=True)
        btn_compare = st.button("START COMPARISON", width='content', type="primary")
        
        if btn_compare:
            with st.spinner("Racing 2 algorithms..."):
                tracemalloc.start()
                ph_gen = run_patternhunter_generator(query_seq, reference_seq, seed_model=t2_ph_seed)
                for ph_res in ph_gen: pass 
                _, ph_peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                ph_peak_kb = ph_peak_mem / 1024
                
                tracemalloc.start()
                blast_gen = run_blast_generator(query_seq, reference_seq, w=t2_blast_w)
                for blast_res in blast_gen: pass
                _, blast_peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                blast_peak_kb = blast_peak_mem / 1024
                
            st.success("Comparison completed!")
            st.divider()
            
            st.subheader("Performance Report")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.info(f"**PatternHunter:** Found {ph_res['metrics']['total_hits']} Hits | Index Size: {ph_res['metrics']['index_size']}")
            with col_s2:
                st.warning(f"**BLAST:** Found {blast_res['metrics']['total_hits']} Hits | Index Size: {blast_res['metrics']['index_size']}")
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            df_metrics = pd.DataFrame({
                "Algorithm": ["PatternHunter", "BLAST"],
                "Execution Time (s)": [ph_res['metrics']['time_seconds'], blast_res['metrics']['time_seconds']],
                "Peak Memory (KB)": [ph_peak_kb, blast_peak_kb]
            })
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("**Execution Time Comparison**")
                chart_time = alt.Chart(df_metrics).mark_bar(color="#1f77b4").encode(
                    x=alt.X("Algorithm", axis=alt.Axis(labelAngle=0)), 
                    y="Execution Time (s)",
                    tooltip=["Algorithm", "Execution Time (s)"]
                ).properties(height=350)
                st.altair_chart(chart_time, width='content')
                
            with col_chart2:
                st.markdown("**Memory Usage Comparison**")
                chart_mem = alt.Chart(df_metrics).mark_bar(color="#ff7f0e").encode(
                    x=alt.X("Algorithm", axis=alt.Axis(labelAngle=0)),
                    y="Peak Memory (KB)",
                    tooltip=["Algorithm", "Peak Memory (KB)"]
                ).properties(height=350)
                st.altair_chart(chart_mem, width='content')

            st.divider()
            st.subheader("Best Hit Details")
            col_hit1, col_hit2 = st.columns(2)
            
            with col_hit1:
                st.markdown("##### PatternHunter")
                if ph_res["final_hits"]:
                    best_ph_hit = ph_res["final_hits"][0] 
                    seed_len = len(t2_ph_seed)
                    q_start = best_ph_hit["query_pos"]
                    r_start = best_ph_hit["ref_pos"]
                    q_str = query_seq[q_start:q_start+seed_len]
                    r_str = reference_seq[r_start:r_start+seed_len]
                    e_val = calculate_e_value(seed_len, len(query_seq), len(reference_seq))
                    st.markdown(render_alignment(q_str, r_str, organism_name=organism, file_name=ref_filename), unsafe_allow_html=True)
                    st.caption(f"E-value: {e_val:.2e} | Query Position: {q_start}")
                else:
                    st.write("No results.")

            with col_hit2:
                st.markdown("##### BLAST")
                if blast_res["final_hits"]:
                    best_blast_hit = max(blast_res["final_hits"], key=lambda x: x["score"])
                    q_start, q_end = best_blast_hit["q_range"]
                    r_start, r_end = best_blast_hit["r_range"]
                    score = best_blast_hit['score']
                    q_str = query_seq[q_start:q_end]
                    r_str = reference_seq[r_start:r_end]
                    e_val = calculate_e_value(score, len(query_seq), len(reference_seq))
                    st.markdown(render_alignment(q_str, r_str, organism_name=organism, file_name=ref_filename), unsafe_allow_html=True)
                    st.caption(f"Score: {score} | E-value: {e_val:.2e} | Query Range: [{q_start}:{q_end}]")
                else:
                    st.write("No results.")

# ==========================================
# MODE 2: MULTIPLE SEQUENCE ALIGNMENT (MSA)
# ==========================================
elif app_mode == "Multiple Sequence Alignment (MSA)":
    
    st.markdown("### Multiple Sequence Alignment (MSA)")
    st.markdown("Analyze evolutionary relationships and find conserved regions across multiple sequences.")
    
    default_fasta = """>Seq1_Human
ATGCTAGCTAGCTAGCTGATCGCAT
>Seq2_Mouse
ATGCTAGCTAGCAAGCTGATCGCAT
>Seq3_Rat
ATGCTAGCTAGCAAGCTGATCGC-T
>Seq4_Chicken
ATGCTAGCTAG-AAGCTGATCGCAT
"""
    st.markdown("#### Input Data (Multi-FASTA format)")
    fasta_input = st.text_area("Paste your sequences here:", value=default_fasta, height=180)
    
    st.divider()

    st.markdown("### 2. Analysis & Alignment")
    tab1_msa, tab2_msa = st.tabs(["Single Run", "Performance Comparison"])

    # ====== TAB 1: SINGLE RUN (MSA) ======
    with tab1_msa:
        col_msa1, col_msa2 = st.columns([1, 2.5])
        
        with col_msa1:
            st.markdown("**Parameter Settings**")
            algo_msa = st.selectbox("MSA Algorithm:", ["Progressive Alignment", "Iterative Refinement", "ClustalW (Coming Soon)"])
            
            t1_iterations = 2
            if algo_msa == "Iterative Refinement":
                t1_iterations = st.number_input("Refinement Iterations:", min_value=1, max_value=10, value=2, key="t1_msa_iter")
                
            btn_run_msa = st.button("Run Alignment", type="primary", width='content')
        
        with col_msa2:
            st.markdown("**Visualization Screen**")
            if btn_run_msa:
                names, seqs = parse_multi_fasta_text(fasta_input)
                
                if len(seqs) < 2:
                    st.error("Please provide at least 2 sequences to align.")
                else:
                    viz_container = st.empty()
                    
                    # --- Call the appropriate MSA Algorithm ---
                    if algo_msa == "Progressive Alignment":
                        generator = run_progressive_msa_generator(seqs, names)
                    elif algo_msa == "Iterative Refinement":
                        generator = run_iterative_msa_generator(seqs, names, iterations=t1_iterations)
                    else:
                        st.warning("This algorithm is coming soon. Running Progressive Alignment instead.")
                        generator = run_progressive_msa_generator(seqs, names)
                    
                    for step_data in generator:
                        if step_data["status"] == "running":
                            with viz_container.container():
                                st.markdown(f"**Step {step_data['step']}:** {step_data['message']}")
                                if step_data.get("current_msa"):
                                    st.markdown(render_msa(step_data["current_msa"], step_data["current_names"]), unsafe_allow_html=True)
                            time.sleep(0.5) # Delay for animation effect
                            
                        elif step_data["status"] == "done":
                            viz_container.empty()
                            st.success(f"{algo_msa} completed!")
                            
                            st.subheader("Final Alignment Results")
                            st.markdown(render_msa(step_data["msa"], step_data["names"]), unsafe_allow_html=True)
                            
                            st.info("💡 **Legend:** Green background indicates 100% conservation. Yellow background indicates high conservation (>60%).")
                            
                            # Performance Metrics
                            st.divider()
                            st.subheader("Performance Metrics")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Execution Time", f"{step_data['metrics']['time_seconds']}s")
                            m2.metric("Sequences Aligned", step_data['metrics']['num_sequences'])
                            m3.metric("MSA Profile Length", step_data['metrics']['alignment_length'])

    # ====== TAB 2: COMPARISON (MSA) ======
    with tab2_msa:
        st.markdown("Compare the performance and final alignment quality of Progressive Alignment vs. Iterative Refinement.")
        
        col_algoA, col_algoB = st.columns(2)
        
        with col_algoA:
            st.info("**Algorithm A: Progressive Alignment**")
            st.write("Fast, greedy approach. Good for closely related sequences.")
            
        with col_algoB:
            st.warning("**Algorithm B: Iterative Refinement**")
            st.write("Repeatedly refines the alignment. Slower but potentially more accurate.")
            t2_iterations = st.number_input("Refinement Iterations:", min_value=1, max_value=10, value=2, key="t2_msa_iter")
            
        st.markdown("<br>", unsafe_allow_html=True)
        btn_compare_msa = st.button("START MSA COMPARISON", width='content', type="primary")
        
        if btn_compare_msa:
            names, seqs = parse_multi_fasta_text(fasta_input)
            
            if len(seqs) < 2:
                st.error("Please provide at least 2 sequences to align.")
            else:
                with st.spinner("Racing MSA algorithms..."):
                    # --- Run Progressive ---
                    tracemalloc.start()
                    prog_gen = run_progressive_msa_generator(seqs, names)
                    for prog_res in prog_gen: pass 
                    _, prog_peak_mem = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    prog_peak_kb = prog_peak_mem / 1024
                    
                    # --- Run Iterative ---
                    tracemalloc.start()
                    iter_gen = run_iterative_msa_generator(seqs, names, iterations=t2_iterations)
                    for iter_res in iter_gen: pass
                    _, iter_peak_mem = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    iter_peak_kb = iter_peak_mem / 1024
                    
                st.success("Comparison completed!")
                st.divider()
                
                st.subheader("Performance Report")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.info(f"**Progressive:** {prog_res['metrics']['alignment_length']} columns | Time: {prog_res['metrics']['time_seconds']}s")
                with col_s2:
                    st.warning(f"**Iterative ({t2_iterations} cycles):** {iter_res['metrics']['alignment_length']} columns | Time: {iter_res['metrics']['time_seconds']}s")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                df_metrics = pd.DataFrame({
                    "Algorithm": ["Progressive Alignment", "Iterative Refinement"],
                    "Execution Time (s)": [prog_res['metrics']['time_seconds'], iter_res['metrics']['time_seconds']],
                    "Peak Memory (KB)": [prog_peak_kb, iter_peak_kb]
                })
                
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.markdown("**Execution Time Comparison**")
                    chart_time = alt.Chart(df_metrics).mark_bar(color="#1f77b4").encode(
                        x=alt.X("Algorithm", axis=alt.Axis(labelAngle=0)), 
                        y="Execution Time (s)",
                        tooltip=["Algorithm", "Execution Time (s)"]
                    ).properties(height=350)
                    st.altair_chart(chart_time, width='content')
                    
                with col_chart2:
                    st.markdown("**Memory Usage Comparison**")
                    chart_mem = alt.Chart(df_metrics).mark_bar(color="#ff7f0e").encode(
                        x=alt.X("Algorithm", axis=alt.Axis(labelAngle=0)),
                        y="Peak Memory (KB)",
                        tooltip=["Algorithm", "Peak Memory (KB)"]
                    ).properties(height=350)
                    st.altair_chart(chart_mem, width='content')

                st.divider()
                st.subheader("Final Alignment Comparison")
                st.markdown("Visual comparison of the final MSA profiles generated by each algorithm.")
                
                st.markdown("##### Progressive Alignment")
                st.markdown(render_msa(prog_res["msa"], prog_res["names"]), unsafe_allow_html=True)
                
                st.markdown("##### Iterative Refinement")
                st.markdown(render_msa(iter_res["msa"], iter_res["names"]), unsafe_allow_html=True)