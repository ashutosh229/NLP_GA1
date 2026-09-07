"""
app.py
------
Part 5: Interactive Streamlit Web Application
Hosts the Integrated Background Editor with:
1. Live-typing mode (incremental user input processing)
2. Simulated fast-typing stream mode (with space-drop merge probability p)
3. Live alert stream ([SEGMENT-ALERT], [SPELL-ALERT], [GRAMMAR-ALERT])
4. Real-time latency tracking (ms)
5. Final Passage Analysis table (PCFG vs Trigram vs Bigram + Decision Rule)
6. Interactive Speed Demon benchmark runner
"""

import sys
import os
import time
import re
import streamlit as st
import pandas as pd
import nltk

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from editor_engine import IntegratedEditorEngine
from evaluator import run_speed_demon_benchmark, generate_speed_demon_batch


st.set_page_config(
    page_title="Integrated Background Editor — NLP GA1",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern dark/light mode aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-seg {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #BFDBFE;
    }
    .badge-spell {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #FDE68A;
    }
    .badge-gram {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #FECACA;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing AI Grammar & Segmentation Engine...")
def load_editor_engine():
    """Initializes and caches models to avoid re-training on app reloads."""
    for c in ['brown', 'treebank', 'gutenberg', 'punkt', 'punkt_tab']:
        try:
            nltk.download(c, quiet=True)
        except Exception:
            pass
    engine = IntegratedEditorEngine(p_merge=0.08, trigger_interval_n=5, k_smoothing=0.05)
    engine.initialize(max_train_sents=2500)
    return engine


engine = load_editor_engine()

# Sidebar Configuration
st.sidebar.title("⚙️ Editor Settings")
p_merge = st.sidebar.slider("Spacebar Drop Probability (p)", min_value=0.0, max_value=0.25, value=0.08, step=0.01,
                            help="Probability that two consecutive words merge without a space, simulating fast typing.")
trigger_n = st.sidebar.slider("Grammar Trigger Interval (N words)", min_value=2, max_value=10, value=5, step=1,
                              help="Word count interval at which sliding-window perplexity and real-word context checks trigger.")
engine.p_merge = p_merge
engine.trigger_interval_n = trigger_n

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧩 Sub-System Architecture")
st.sidebar.markdown("""
- **Q1 Segmenter:** Trigram DP + HMM
- **Q3 Corrector:** SymSpell & Edit-Distance-1
- **PCFG Parser:** CKY Viterbi (Penn Treebank)
- **Shared LM:** Add-$k$ Smoothed ($k=0.05$)
""")

# App Header
st.markdown("<div class='main-title'>✍️ Integrated Background Editor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Live Word Segmentation, Spelling Correction, and Constituency-Based Grammar Checking</div>", unsafe_allow_html=True)

tab_live, tab_sim, tab_bench = st.tabs(["⚡ Live Typing Mode", "🎬 Simulated Typing Stream", "🚀 Speed Demon Benchmark"])

# -------------------------------------------------------------
# TAB 1: LIVE TYPING MODE
# -------------------------------------------------------------
with tab_live:
    st.subheader("Interactive Real-Time Typing")
    st.caption("Type or paste sentences below. The editor processes tokens incrementally, firing segmentation, spelling, and grammar alerts live.")

    default_text = "thequick brown fox jumped over the lazy dog. please meat me at the station tomorrow."
    user_input = st.text_area("Live Input Stream", value=default_text, height=130)

    if st.button("Process Live Input", type="primary"):
        tokens = user_input.strip().split()
        if not tokens:
            st.warning("Please enter text above.")
        else:
            accumulated_tokens = []
            alerts_log = []
            tok_latencies = []
            gram_latencies = []
            merges = 0
            spells = 0

            with st.spinner("Analyzing text incrementally..."):
                for idx, tok in enumerate(tokens):
                    res = engine.check_token(tok)
                    tok_latencies.append(res["latency_ms"])
                    if res["is_segmented"]:
                        merges += 1
                    if res["is_spelled"]:
                        spells += 1
                    for a in res["alerts"]:
                        alerts_log.append(a)

                    accumulated_tokens.extend(res["resulting_tokens"])

                    if len(accumulated_tokens) % engine.trigger_interval_n == 0:
                        words_so_far = [w for w, _ in accumulated_tokens]
                        g_res = engine.check_grammar_window(words_so_far)
                        gram_latencies.append(g_res["latency_ms"])
                        for ga in g_res["alerts"]:
                            alerts_log.append(ga)

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Merges Resolved", merges)
            with c2:
                st.metric("Spelling Corrections", spells)
            with c3:
                avg_tok = sum(tok_latencies) / max(1, len(tok_latencies))
                st.metric("Avg Token Latency", f"{avg_tok:.2f} ms")
            with c4:
                avg_gram = sum(gram_latencies) / max(1, len(gram_latencies))
                st.metric("Avg Grammar Latency", f"{avg_gram:.2f} ms")

            # Alert Stream
            st.markdown("#### 🚨 Real-Time Alerts")
            if not alerts_log:
                st.success("No segmentation, spelling, or grammatical errors detected!")
            else:
                for a in alerts_log:
                    if a["type"] == "SEGMENT-ALERT":
                        st.markdown(f"<span class='badge-seg'>[SEGMENT-ALERT]</span> {a['message']}", unsafe_allow_html=True)
                    elif a["type"] == "SPELL-ALERT":
                        st.markdown(f"<span class='badge-spell'>[SPELL-ALERT]</span> {a['message']}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='badge-gram'>[GRAMMAR-ALERT]</span> {a['message']}", unsafe_allow_html=True)

            # Reconciled Output
            st.markdown("#### 📝 Reconciled Passage")
            corrected_str = " ".join(w for w, _ in accumulated_tokens)
            st.info(corrected_str)

            # Final Sentence Table
            st.markdown("#### 📊 Part 4: Final Sentence Structural Analysis")
            summary = engine.analyze_final_passage(accumulated_tokens, merges, spells)
            df_summary = pd.DataFrame(summary)[["sentence_idx", "sentence_text", "pcfg_result", "bigram_score", "trigram_score", "chosen_method", "final_verdict", "decision_reason"]]
            df_summary.columns = ["Sent #", "Sentence Text", "PCFG Score", "Bigram (PPL)", "Trigram (PPL)", "Method", "Verdict", "Rationale"]
            st.dataframe(df_summary, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: SIMULATED TYPING STREAM
# -------------------------------------------------------------
with tab_sim:
    st.subheader("Simulated Typist Word-by-Word Stream")
    st.caption("Streams a sampled paragraph word-by-word, randomly dropping spaces between words to test live segmentation and alerting.")

    sample_choice = st.selectbox(
        "Choose Passage Source",
        [
            "Gutenberg Classic Excerpt",
            "Brown News Paragraph",
            "Custom Injected Errors Paragraph"
        ]
    )

    if sample_choice == "Custom Injected Errors Paragraph":
        sim_passage = (
            "The quick brownfox jumps over the lazydog near the quiet riverbank. "
            "Please meat me at the station because I hav a great plan for us. "
            "Every morning the old man walked to themarket to buy fresh bread. "
            "This is an extraordinary test sentnce that validates our editor. "
            "They will travel together across the ancient countryside by train."
        )
    elif sample_choice == "Brown News Paragraph":
        sim_passage = (
            "The committee announced that the financial budget would be approved soon. "
            "Several representative members voiced strong objections during the afternoon session. "
            "A comprehensive review of the entire economic plan will begin next Monday. "
            "Officials stated that public cooperation remains essential for sustainable growth. "
            "Both parties agreed to resume constructive discussions later this month."
        )
    else:
        sim_passage = (
            "The journey through the dense northern forest was long and demanding. "
            "Cold winds blew relentlessly across the open rocky plains for days. "
            "The weary travelers finally arrived at the warm village before sunset. "
            "Friendly villagers offered wholesome food and shelter near the hearth. "
            "Peaceful silence returned to the mountains as night descended slowly."
        )

    st.text_area("Passage to Stream", value=sim_passage, height=100, disabled=True)
    speed_delay = st.slider("Typing Simulation Delay (seconds per token)", min_value=0.01, max_value=0.30, value=0.05, step=0.02)

    if st.button("▶️ Start Simulated Streaming", type="primary"):
        merged_tokens = engine.simulate_typing_stream(sim_passage)
        st.markdown(f"**Merged Stream Input ({len(merged_tokens)} tokens, space-drop p={engine.p_merge}):**")
        st.code(" ".join(merged_tokens))

        stream_box = st.empty()
        alert_box = st.empty()
        metric_box = st.empty()

        accumulated_tokens = []
        alerts_log = []
        tok_latencies = []
        gram_latencies = []
        merges = 0
        spells = 0

        for idx, tok in enumerate(merged_tokens):
            res = engine.check_token(tok)
            tok_latencies.append(res["latency_ms"])
            if res["is_segmented"]:
                merges += 1
            if res["is_spelled"]:
                spells += 1
            for a in res["alerts"]:
                alerts_log.append(a)

            accumulated_tokens.extend(res["resulting_tokens"])

            if len(accumulated_tokens) % engine.trigger_interval_n == 0:
                words_so_far = [w for w, _ in accumulated_tokens]
                g_res = engine.check_grammar_window(words_so_far)
                gram_latencies.append(g_res["latency_ms"])
                for ga in g_res["alerts"]:
                    alerts_log.append(ga)

            # Update live stream view
            current_stream_text = " ".join(w for w, _ in accumulated_tokens)
            stream_box.markdown(f"**Live Reconstructed Text:**\n> {current_stream_text}")

            time.sleep(speed_delay)

        st.success("Typing simulation complete!")

        # Final Analysis
        summary = engine.analyze_final_passage(accumulated_tokens, merges, spells)
        df_summary = pd.DataFrame(summary)[["sentence_idx", "sentence_text", "pcfg_result", "bigram_score", "trigram_score", "chosen_method", "final_verdict", "decision_reason"]]
        df_summary.columns = ["Sent #", "Sentence Text", "PCFG Score", "Bigram (PPL)", "Trigram (PPL)", "Method", "Verdict", "Rationale"]
        st.dataframe(df_summary, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: SPEED DEMON BENCHMARK
# -------------------------------------------------------------
with tab_bench:
    st.subheader("🚀 Speed Demon Benchmark (1,000 Corrupted Words)")
    st.caption("Part 5.2 requirement: passes 1,000 corrupted tokens through the full live-check pipeline (segmentation + spelling) vs the grammar check in isolation to quantify latency overhead.")

    if st.button("Run Speed Demon Benchmark (1,000 Words)", type="primary"):
        with st.spinner("Generating 1,000 corrupted tokens and benchmarking..."):
            bench = run_speed_demon_benchmark(engine, batch_size=1000)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Full Pipeline Latency", f"{bench['avg_full_ms']:.3f} ms / word", f"{bench['total_full_ms']:.1f} ms total")
        with c2:
            st.metric("Grammar-Only Latency", f"{bench['avg_gram_ms']:.3f} ms / word", f"{bench['total_gram_ms']:.1f} ms total")
        with c3:
            st.metric("Latency Added by Seg+Spell", f"{bench['latency_added_ms']:.3f} ms / word", f"{bench['latency_ratio']:.2f}x ratio")

        st.markdown("#### 🔬 Benchmark Analysis & Conclusion")
        st.markdown(f"""
        - **Batch Size:** {bench['batch_size']} simulated words (single-edit typos, merged words, clean words).
        - **Resolved Segments:** {bench['segmented_count']} merged tokens successfully identified and split by Q1 beam decoder.
        - **Resolved Spelling Errors:** {bench['spelled_count']} non-word typos corrected via Q3 SymSpell / ED1.
        - **Per-token Latency:** Full live-check averages **{bench['avg_full_ms']:.3f} ms/word**, which is **far below human typing speed (typically ~150–250 ms per keystroke)**.
        - **Conclusion:** Because the segmentation check only triggers when words are not found in the vocabulary or are unusually long, and SymSpell operates in $O(1)$ dictionary lookup time, the combined layer adds minimal latency and runs smoothly in real time without lag.
        """)
