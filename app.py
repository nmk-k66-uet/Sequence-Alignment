import streamlit as st
import os
import time
import tracemalloc
import math
import pandas as pd
import plotly.express as px
from Bio import SeqIO
from collections import Counter

# Import algorithms directly
from algorithms.patternhunter import run_patternhunter_generator
from algorithms.blast import run_blast_generator
from algorithms.progressive import run_progressive_msa_generator
from algorithms.iterative import run_iterative_msa_generator
from algorithms.kmer_clustering import run_ml_clustering_msa_generator
from algorithms.upgma import run_upgma_msa_generator

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Sequence Alignment Simulator", layout="wide")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_e_value(score, query_len, ref_len):
    K = 0.1
    lam = 0.3
    return K * query_len * ref_len * math.exp(-lam * score)

def calculate_pairwise_stats(q_str, r_str):
    MATCH_SCORE = 2
    MISMATCH_SCORE = -1
    GAP_PENALTY = -1

    score, matches, mismatches, gaps = 0, 0, 0, 0
    
    for q, r in zip(q_str, r_str):
        if q == '-' or r == '-':
            score += GAP_PENALTY
            gaps += 1
        elif q == r:
            score += MATCH_SCORE
            matches += 1
        else:
            score += MISMATCH_SCORE
            mismatches += 1
            
    return score, matches, mismatches, gaps

def calculate_sp_score(msa_profile):
    MATCH_SCORE = 2
    MISMATCH_SCORE = -1
    GAP_PENALTY = -1

    num_seqs = len(msa_profile)
    if num_seqs < 2:
        return 0, 0, 0, 0
        
    seq_len = len(msa_profile[0])
    total_score, total_matches, total_mismatches, total_gaps = 0, 0, 0, 0
    
    for i in range(num_seqs):
        for j in range(i + 1, num_seqs):
            seq1, seq2 = msa_profile[i], msa_profile[j]
            for k in range(seq_len):
                c1, c2 = seq1[k], seq2[k]
                if c1 == '-' and c2 == '-':
                    pass 
                elif c1 == '-' or c2 == '-':
                    total_score += GAP_PENALTY
                    total_gaps += 1
                elif c1 == c2:
                    total_score += MATCH_SCORE
                    total_matches += 1
                else:
                    total_score += MISMATCH_SCORE
                    total_mismatches += 1
                    
    return total_score, total_matches, total_mismatches, total_gaps

def render_alignment(q_str, r_str, organism_name="", file_name=""):
    html = '<div style="font-family: Courier, monospace; font-size: 16px; background-color: rgba(128, 128, 128, 0.1); padding: 10px; border-radius: 5px; overflow-x: auto; white-space: pre; margin-bottom: 5px;">'
    
    if organism_name or file_name:
        html += f'<div style="font-size: 14px; font-family: sans-serif; color: #495057; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 10px;"><b>Database:</b> {organism_name} &nbsp;|&nbsp; <b>File:</b> {file_name}</div>'
        
    q_html = r_html = m_html = ""
    
    for q, r in zip(q_str, r_str):
        if q == r and q != '-':
            color, m_char = "#28a745", "|"
        elif q == '-' or r == '-':
            color, m_char = "#6c757d", " "
        else:
            color, m_char = "#dc3545", "x"
            
        q_html += f'<span style="color: {color}; font-weight: bold;">{q}</span>'
        r_html += f'<span style="color: {color}; font-weight: bold;">{r}</span>'
        m_html += f'<span style="color: {color};">{m_char}</span>'
        
    html += f'<div>Query: {q_html}</div>'
    html += f'<div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{m_html}</div>'
    html += f'<div>Ref&nbsp;&nbsp;: {r_html}</div></div>'
    return html

def render_msa(aligned_sequences, seq_names):
    if not aligned_sequences:
        return "<div>No alignment data available</div>"

    seq_length = len(aligned_sequences[0])
    num_seqs = len(aligned_sequences)
    
    html = '<div style="font-family: Courier, monospace; font-size: 14px; background-color: rgba(128, 128, 128, 0.05); padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre; border: 1px solid #ddd;">'
    
    column_colors = []
    consensus_str = ""
    
    for col_idx in range(seq_length):
        col_chars = [seq[col_idx] for seq in aligned_sequences if seq[col_idx] != '-']
        if not col_chars:
            column_colors.append("transparent")
            consensus_str += "-"
            continue
            
        most_common_char, count = Counter(col_chars).most_common(1)[0]
        ratio = count / num_seqs
        
        if ratio == 1.0:
            column_colors.append("rgba(40, 167, 69, 0.4)")
            consensus_str += most_common_char
        elif ratio >= 0.6:
            column_colors.append("rgba(255, 193, 7, 0.3)")
            consensus_str += most_common_char.lower()
        else:
            column_colors.append("transparent")
            consensus_str += "."
            
    max_name_len = max([len(name) for name in seq_names] + [len("Consensus")])
    
    for seq_name, seq in zip(seq_names, aligned_sequences):
        padded_name = seq_name.ljust(max_name_len + 2)
        html += f'<div style="margin-bottom: 2px;"><strong>{padded_name}</strong> '
        for i, char in enumerate(seq):
            html += f'<span style="background-color: {column_colors[i]};">{char}</span>'
        html += '</div>'
        
    html += '<hr style="margin: 8px 0; border-color: #ccc;">'
    padded_consensus = "Consensus".ljust(max_name_len + 2)
    html += f'<div style="color: #666;"><strong>{padded_consensus}</strong> '
    for char in consensus_str:
        html += f'<span>{char}</span>'
    html += '</div></div>'
    
    return html

def render_performance_charts(df_perf, df_stats):
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        fig_time = px.bar(df_perf, x="Algorithm", y="Execution Time (s)", 
                          title="Execution Time (s)", text_auto='.3f',
                          color_discrete_sequence=["#1f77b4"])
        fig_time.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
        st.plotly_chart(fig_time, width='stretch')
        
    with c2:
        fig_mem = px.bar(df_perf, x="Algorithm", y="Peak Memory (KB)", 
                         title="Peak Memory (KB)", text_auto='.1f',
                         color_discrete_sequence=["#ff7f0e"])
        fig_mem.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
        st.plotly_chart(fig_mem, width='stretch')

    with c3:
        fig_score = px.bar(df_perf, x="Algorithm", y="Alignment Score", 
                           title="Alignment Score", text_auto=True, 
                           color_discrete_sequence=["#8c564b"])
        fig_score.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
        st.plotly_chart(fig_score, width='stretch')
        
    with c4:
        fig_stats = px.bar(df_stats, x="Algorithm", y="Count", color="Type",
                           title="Matches, Mismatches & Gaps", barmode='group', text_auto=True,
                           color_discrete_map={"Matches": "#28a745", "Mismatches": "#dc3545", "Gaps": "#6c757d"})
        fig_stats.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
        st.plotly_chart(fig_stats, width='stretch')

@st.cache_data
def get_available_organisms(data_dir="Data"):
    if not os.path.exists(data_dir):
        os.makedirs(os.path.join(data_dir, "Demo_Organism"))
        return ["Demo_Organism"]
    organisms = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return organisms if organisms else ["No data available"]

@st.cache_data
def load_fasta(organism_name, data_dir="Data"):
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
    names, seqs = [], []
    curr_name, curr_seq = "" , []
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
        available_orgs = get_available_organisms()
        
        selected_orgs = st.multiselect(
            "Select Target Database(s):", 
            options=available_orgs, 
            default=available_orgs[:1] if available_orgs else None
        )
            
        if not selected_orgs:
            st.warning("Please select at least one database to proceed.")
            
    with col_data2:
        query_seq = st.text_input("Enter Query Sequence (DNA):", "AGCTAGCAAGCTGATC").upper()

    st.divider()

    st.markdown("### 2. Analysis & Alignment")
    tab1, tab2 = st.tabs(["Single Run", "Performance Comparison"])

    # ====== TAB 1: SINGLE RUN ======
    with tab1:
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
            if btn_run_single and selected_orgs:
                viz_container = st.empty() 
                progress_bar = st.progress(0.0)
                tracemalloc.start()
                
                all_final_hits = []
                total_time = 0
                total_index_size = 0
                
                for idx, org in enumerate(selected_orgs):
                    ref_seq, ref_filename = load_fasta(org)
                    
                    if algo_choice == "PatternHunter":
                        generator = run_patternhunter_generator(query_seq, ref_seq, seed_model=seed_input) 
                    else:
                        generator = run_blast_generator(query_seq, ref_seq, w=w_input)
                        
                    for step_data in generator:
                        if step_data["status"] == "running":
                            with viz_container.container():
                                st.markdown(f"**Scanning Database: {org} ({idx+1}/{len(selected_orgs)})**")
                                if algo_choice == "PatternHunter":
                                    st.code(f"Query Position: {step_data['query_pos']}\nExtracted Seed: {step_data['current_seed']}")
                                else:
                                    st.code(f"Query Position: {step_data['query_pos']}\nCurrent K-mer: {step_data['current_kmer']}")
                                st.info(f"Target Hits Found: {step_data['total_hits_so_far']} | Cumulative Hits: {len(all_final_hits) + step_data['total_hits_so_far']}")
                            time.sleep(0.02) 
                            
                        elif step_data["status"] == "done":
                            total_time += step_data["metrics"]["time_seconds"]
                            total_index_size += step_data["metrics"]["index_size"]
                            
                            # Inject source tracking into hits
                            for hit in step_data["final_hits"]:
                                hit["organism"] = org
                                hit["filename"] = ref_filename
                                hit["ref_seq"] = ref_seq
                            all_final_hits.extend(step_data["final_hits"])
                            
                    progress_bar.progress((idx + 1) / len(selected_orgs))
                        
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_mem_kb = peak_mem / 1024
                
                viz_container.empty()
                progress_bar.empty()
                st.success(f"Search completed across {len(selected_orgs)} databases.")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Execution Time", f"{total_time:.4f}s")
                m2.metric("Total Hits", len(all_final_hits))
                m3.metric("Combined Index Size", total_index_size)
                m4.metric("Peak Memory", f"{peak_mem_kb:.2f} KB")
                
                if all_final_hits:
                    if algo_choice == "BLAST":
                        all_final_hits.sort(key=lambda x: x["score"], reverse=True)
                    
                    st.subheader("Top Result Details (Showing top 20)")
                    for i, hit in enumerate(all_final_hits[:20]):
                        st.markdown(f"**Result #{i+1}**")
                        if algo_choice == "BLAST":
                            q_start, q_end = hit["q_range"]
                            r_start, r_end = hit["r_range"]
                            score = hit['score']
                            q_str = query_seq[q_start:q_end]
                            r_str = hit["ref_seq"][r_start:r_end]
                            e_val = calculate_e_value(score, len(query_seq), len(hit["ref_seq"]))
                            
                            st.markdown(render_alignment(q_str, r_str, organism_name=hit["organism"], file_name=hit["filename"]), unsafe_allow_html=True)
                            st.caption(f"Score: {score} | E-value: {e_val:.2e} | Query Range: [{q_start}:{q_end}]")
                            
                        elif algo_choice == "PatternHunter":
                            seed_len = len(seed_input)
                            q_start = hit["query_pos"]
                            r_start = hit["ref_pos"]
                            score = seed_len 
                            q_str = query_seq[q_start:q_start+seed_len]
                            r_str = hit["ref_seq"][r_start:r_start+seed_len]
                            e_val = calculate_e_value(score, len(query_seq), len(hit["ref_seq"]))
                            
                            st.markdown(render_alignment(q_str, r_str, organism_name=hit["organism"], file_name=hit["filename"]), unsafe_allow_html=True)
                            st.caption(f"E-value: {e_val:.2e} | Query Position: {q_start} | Seed Model: {hit['matched_seed']}")
                        st.divider()
                else:
                    st.warning("No matching results found in the selected databases.")

    # ====== TAB 2: COMPARISON ======
    with tab2:
        col_algoA, col_algoB = st.columns(2)
        
        with col_algoA:
            st.info("**Algorithm A: PatternHunter**")
            t2_ph_seed = st.text_input("Spaced Seed:", value="111010010100110111", key="t2_seed")
        with col_algoB:
            st.warning("**Algorithm B: BLAST**")
            t2_blast_w = st.number_input("K-mer Size (w):", min_value=2, max_value=50, value=11, key="t2_w")
            
        st.markdown("<br>", unsafe_allow_html=True)
        btn_compare = st.button("START COMPARISON", width='content', type="primary")
        
        if btn_compare and selected_orgs:
            with st.spinner(f"Executing algorithms across {len(selected_orgs)} databases..."):
                
                # --- Algorithm A Execution ---
                tracemalloc.start()
                ph_total_time = 0
                ph_index_size = 0
                ph_all_hits = []
                for org in selected_orgs:
                    ref_seq, ref_filename = load_fasta(org)
                    ph_gen = run_patternhunter_generator(query_seq, ref_seq, seed_model=t2_ph_seed)
                    for step in ph_gen:
                        if step["status"] == "done":
                            ph_total_time += step["metrics"]["time_seconds"]
                            ph_index_size += step["metrics"]["index_size"]
                            for h in step["final_hits"]:
                                h["organism"] = org
                                h["filename"] = ref_filename
                                h["ref_seq"] = ref_seq
                            ph_all_hits.extend(step["final_hits"])
                _, ph_peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                ph_peak_kb = ph_peak_mem / 1024
                
                # --- Algorithm B Execution ---
                tracemalloc.start()
                blast_total_time = 0
                blast_index_size = 0
                blast_all_hits = []
                for org in selected_orgs:
                    ref_seq, ref_filename = load_fasta(org)
                    blast_gen = run_blast_generator(query_seq, ref_seq, w=t2_blast_w)
                    for step in blast_gen:
                        if step["status"] == "done":
                            blast_total_time += step["metrics"]["time_seconds"]
                            blast_index_size += step["metrics"]["index_size"]
                            for h in step["final_hits"]:
                                h["organism"] = org
                                h["filename"] = ref_filename
                                h["ref_seq"] = ref_seq
                            blast_all_hits.extend(step["final_hits"])
                _, blast_peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                blast_peak_kb = blast_peak_mem / 1024
                
            st.success("Comparison completed.")
            st.divider()
            
            # --- CALCULATE BEST HIT STATS ---
            if ph_all_hits:
                best_ph_hit = ph_all_hits[0] 
                seed_len = len(t2_ph_seed)
                ph_q_str = query_seq[best_ph_hit["query_pos"]:best_ph_hit["query_pos"]+seed_len]
                ph_r_str = best_ph_hit["ref_seq"][best_ph_hit["ref_pos"]:best_ph_hit["ref_pos"]+seed_len]
                scoreA, matchA, mismatchA, gapA = calculate_pairwise_stats(ph_q_str, ph_r_str)
            else:
                scoreA, matchA, mismatchA, gapA = 0, 0, 0, 0
                
            if blast_all_hits:
                best_blast_hit = max(blast_all_hits, key=lambda x: x["score"])
                blast_q_str = query_seq[best_blast_hit["q_range"][0]:best_blast_hit["q_range"][1]]
                blast_r_str = best_blast_hit["ref_seq"][best_blast_hit["r_range"][0]:best_blast_hit["r_range"][1]]
                scoreB, matchB, mismatchB, gapB = calculate_pairwise_stats(blast_q_str, blast_r_str)
            else:
                scoreB, matchB, mismatchB, gapB = 0, 0, 0, 0

            # --- PERFORMANCE REPORT ---
            st.subheader("Performance Report")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.info(f"**PatternHunter:** Score: {scoreA} | Hits: {len(ph_all_hits)} | Time: {ph_total_time:.4f}s")
            with col_s2:
                st.warning(f"**BLAST:** Score: {scoreB} | Hits: {len(blast_all_hits)} | Time: {blast_total_time:.4f}s")
            st.markdown("<br>", unsafe_allow_html=True)
            
            df_perf_psa = pd.DataFrame({
                "Algorithm": ["PatternHunter", "BLAST"],
                "Execution Time (s)": [ph_total_time, blast_total_time],
                "Peak Memory (KB)": [ph_peak_kb, blast_peak_kb],
                "Alignment Score": [scoreA, scoreB]
            })
            
            df_stats_psa = pd.DataFrame([
                {"Algorithm": "PatternHunter", "Type": "Matches", "Count": matchA},
                {"Algorithm": "PatternHunter", "Type": "Mismatches", "Count": mismatchA},
                {"Algorithm": "PatternHunter", "Type": "Gaps", "Count": gapA},
                {"Algorithm": "BLAST", "Type": "Matches", "Count": matchB},
                {"Algorithm": "BLAST", "Type": "Mismatches", "Count": mismatchB},
                {"Algorithm": "BLAST", "Type": "Gaps", "Count": gapB}
            ])

            render_performance_charts(df_perf_psa, df_stats_psa)

            st.divider()
            st.subheader("Best Hit Comparison")
            col_hit1, col_hit2 = st.columns(2)
            
            with col_hit1:
                st.markdown("##### PatternHunter")
                if ph_all_hits:
                    e_val_A = calculate_e_value(scoreA, len(query_seq), len(best_ph_hit["ref_seq"]))
                    st.markdown(render_alignment(ph_q_str, ph_r_str, organism_name=best_ph_hit["organism"], file_name=best_ph_hit["filename"]), unsafe_allow_html=True)
                    st.caption(f"Score: {scoreA} | E-value: {e_val_A:.2e}")
                else:
                    st.write("No results.")

            with col_hit2:
                st.markdown("##### BLAST")
                if blast_all_hits:
                    e_val_B = calculate_e_value(scoreB, len(query_seq), len(best_blast_hit["ref_seq"]))
                    st.markdown(render_alignment(blast_q_str, blast_r_str, organism_name=best_blast_hit["organism"], file_name=best_blast_hit["filename"]), unsafe_allow_html=True)
                    st.caption(f"Score: {scoreB} | E-value: {e_val_B:.2e}")
                else:
                    st.write("No results.")

# ==========================================
# MODE 2: MULTIPLE SEQUENCE ALIGNMENT (MSA)
# ==========================================
elif app_mode == "Multiple Sequence Alignment (MSA)":
    
    st.markdown("### Multiple Sequence Alignment (MSA)")
    
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

    tab1_msa, tab2_msa = st.tabs(["Single Run", "Performance Comparison"])
    msa_algo_list = ["Progressive Alignment", "UPGMA Hierarchical", "Iterative Refinement", "K-mer Clustering (Hybrid ML)"]

    # ====== TAB 1: SINGLE RUN (MSA) ======
    with tab1_msa:
        col_msa1, col_msa2 = st.columns([1, 2.5])
        
        with col_msa1:
            st.markdown("**Parameter Settings**")
            algo_msa = st.selectbox("MSA Algorithm:", msa_algo_list, key="t1_algo_msa")
            
            t1_params = {}
            if algo_msa == "Iterative Refinement":
                t1_params['iterations'] = st.number_input("Refinement Iterations:", min_value=1, max_value=10, value=2, key="t1_msa_iter")
            elif algo_msa == "K-mer Clustering (Hybrid ML)":
                t1_params['k'] = st.number_input("K-mer Size (Feature Extraction):", min_value=2, max_value=6, value=3, key="t1_msa_kmer")
            elif algo_msa == "UPGMA Hierarchical":
                t1_params['method'] = st.selectbox("Clustering Method:", ["UPGMA", "NJ"], key="t1_msa_method")

            btn_run_msa = st.button("Run Alignment", type="primary", width='content')
        
        with col_msa2:
            st.markdown("**Visualization Screen**")
            if btn_run_msa:
                names, seqs = parse_multi_fasta_text(fasta_input)
                
                if len(seqs) < 2:
                    st.error("Please provide at least 2 sequences to align.")
                else:
                    st.markdown("### Detailed Alignment Process")
                    
                    if algo_msa == "Progressive Alignment":
                        generator = run_progressive_msa_generator(seqs, names)
                    elif algo_msa == "Iterative Refinement":
                        generator = run_iterative_msa_generator(seqs, names, iterations=t1_params['iterations'])
                    elif algo_msa == "K-mer Clustering (Hybrid ML)":
                        generator = run_ml_clustering_msa_generator(seqs, names, k=t1_params['k'])
                    elif algo_msa == "UPGMA Hierarchical":
                        generator = run_upgma_msa_generator(seqs, names)
                    
                    for step_data in generator:
                        if step_data["status"] == "card":
                            with st.expander(f"{step_data['title']}", expanded=True):
                                if "description" in step_data:
                                    st.markdown(step_data["description"])
                                if "tree_figure" in step_data and step_data["tree_figure"] is not None:
                                    st.plotly_chart(step_data["tree_figure"], width='stretch')

                                if "matrix" in step_data and step_data["matrix"]:
                                    df_matrix = pd.DataFrame(
                                        step_data["matrix"], 
                                        columns=step_data["matrix_labels"], 
                                        index=step_data["matrix_labels"]
                                    )
                                    st.markdown("**Similarity Matrix:**")
                                    st.dataframe(df_matrix.style.background_gradient(cmap='Blues'), use_container_width=True)
                                
                                if "current_msa" in step_data:
                                    st.markdown(render_msa(step_data["current_msa"], step_data["current_names"]), unsafe_allow_html=True)
                            
                            time.sleep(0.2)
                            
                        elif step_data["status"] == "done":
                            st.success(f"{algo_msa} completed.")
                            
                            st.subheader("Final Alignment Result")
                            st.markdown(render_msa(step_data["msa"], step_data["names"]), unsafe_allow_html=True)
                            st.info("**Legend:** Green background indicates 100% conservation. Yellow background indicates high conservation (>60%).")
                            
                            st.divider()
                            st.subheader("Performance & Alignment Metrics")
                            
                            score, matches, mismatches, gaps = calculate_sp_score(step_data["msa"])
                            
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Execution Time", f"{step_data['metrics']['time_seconds']}s")
                            m2.metric("MSA Profile Length", step_data['metrics']['alignment_length'])
                            m3.metric("Alignment Score (SP)", score)
                            m4.metric("Match / Mismatch / Gap", f"{matches} / {mismatches} / {gaps}")

    # ====== TAB 2: COMPARISON (MSA) ======
    with tab2_msa:
        col_algoA, col_algoB = st.columns(2)
        
        with col_algoA:
            st.info("**Algorithm A**")
            t2_algoA = st.selectbox("Select Algorithm A:", msa_algo_list, index=0, key="t2_algoA")
            t2_paramsA = {}
            if t2_algoA == "Iterative Refinement":
                t2_paramsA['iterations'] = st.number_input("Refinement Iterations (A):", min_value=1, max_value=10, value=2, key="t2_iterA")
            elif t2_algoA == "K-mer Clustering (Hybrid ML)":
                t2_paramsA['k'] = st.number_input("K-mer Size (A):", min_value=2, max_value=6, value=3, key="t2_kmerA")
                
        with col_algoB:
            st.warning("**Algorithm B**")
            t2_algoB = st.selectbox("Select Algorithm B:", msa_algo_list, index=2, key="t2_algoB")
            t2_paramsB = {}
            if t2_algoB == "Iterative Refinement":
                t2_paramsB['iterations'] = st.number_input("Refinement Iterations (B):", min_value=1, max_value=10, value=2, key="t2_iterB")
            elif t2_algoB == "K-mer Clustering (Hybrid ML)":
                t2_paramsB['k'] = st.number_input("K-mer Size (B):", min_value=2, max_value=6, value=3, key="t2_kmerB")
            
        st.markdown("<br>", unsafe_allow_html=True)
        btn_compare_msa = st.button("START MSA COMPARISON", width='content', type="primary")
        
        if btn_compare_msa:
            names, seqs = parse_multi_fasta_text(fasta_input)
            
            if len(seqs) < 2:
                st.error("Please provide at least 2 sequences to align.")
            else:
                with st.spinner("Executing algorithms..."):
                    
                    def run_selected_generator(algo_name, params):
                        if algo_name == "Progressive Alignment":
                            return run_progressive_msa_generator(seqs, names)
                        elif algo_name == "Iterative Refinement":
                            return run_iterative_msa_generator(seqs, names, iterations=params['iterations'])
                        elif algo_name == "K-mer Clustering (Hybrid ML)":
                            return run_ml_clustering_msa_generator(seqs, names, k=params['k'])
                    
                    tracemalloc.start()
                    genA = run_selected_generator(t2_algoA, t2_paramsA)
                    for resA in genA: pass
                    _, peakA = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peakA_kb = peakA / 1024
                    
                    tracemalloc.start()
                    genB = run_selected_generator(t2_algoB, t2_paramsB)
                    for resB in genB: pass
                    _, peakB = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peakB_kb = peakB / 1024
                    
                st.success("Comparison completed.")
                st.divider()
                
                scoreA, matchA, mismatchA, gapA = calculate_sp_score(resA["msa"])
                scoreB, matchB, mismatchB, gapB = calculate_sp_score(resB["msa"])
                
                st.subheader("Performance Report")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.info(f"**A ({t2_algoA}):** Score: {scoreA} | Length: {resA['metrics']['alignment_length']} | Time: {resA['metrics']['time_seconds']}s")
                with col_s2:
                    st.warning(f"**B ({t2_algoB}):** Score: {scoreB} | Length: {resB['metrics']['alignment_length']} | Time: {resB['metrics']['time_seconds']}s")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                df_perf_msa = pd.DataFrame({
                    "Algorithm": ["A", "B"],
                    "Execution Time (s)": [resA['metrics']['time_seconds'], resB['metrics']['time_seconds']],
                    "Peak Memory (KB)": [peakA_kb, peakB_kb],
                    "Alignment Score": [scoreA, scoreB]
                })
                
                df_stats_msa = pd.DataFrame([
                    {"Algorithm": "A", "Type": "Matches", "Count": matchA},
                    {"Algorithm": "A", "Type": "Mismatches", "Count": mismatchA},
                    {"Algorithm": "A", "Type": "Gaps", "Count": gapA},
                    {"Algorithm": "B", "Type": "Matches", "Count": matchB},
                    {"Algorithm": "B", "Type": "Mismatches", "Count": mismatchB},
                    {"Algorithm": "B", "Type": "Gaps", "Count": gapB}
                ])
                
                render_performance_charts(df_perf_msa, df_stats_msa)

                st.divider()
                st.subheader("Final Alignment Comparison")
                
                st.markdown(f"##### Algorithm A: {t2_algoA}")
                st.markdown(render_msa(resA["msa"], resA["names"]), unsafe_allow_html=True)
                
                st.markdown(f"##### Algorithm B: {t2_algoB}")
                st.markdown(render_msa(resB["msa"], resB["names"]), unsafe_allow_html=True)