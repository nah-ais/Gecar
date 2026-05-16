"""
=============================================================================
GECAR NLP DASHBOARD — Streamlit Application
=============================================================================
Dashboard interaktif untuk:
  - Heatmap Wilayah × Pilar Sektoral
  - Diverging Bar Chart (Masyarakat vs Pemerintah)
  - Topic Modeling (NMF / BERTopic)
  - Gap Analysis Visual
  - Verbatim Quotes Explorer
  - Benang Merah per Wilayah
  - Narrative / Creative View untuk tim desain
=============================================================================
Jalankan dengan:
  streamlit run app.py
=============================================================================
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
# Sesuaikan path ke file CSV kamu di sini:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Coba cari CSV di beberapa lokasi umum
POSSIBLE_PATHS = [
    BASE_DIR,
    os.path.join(BASE_DIR, "data"),
    "/mnt/user-data/uploads",
    ".",
]

def find_file(filename):
    for path in POSSIBLE_PATHS:
        full = os.path.join(path, filename)
        if os.path.exists(full):
            return full
    return None

PATH_KELOMPOK = find_file("Gecar_-_Kelompok.csv") or "Gecar_-_Kelompok.csv"
PATH_KII      = find_file("Gecar_-_KII.csv")      or "Gecar_-_KII.csv"

# ── Import modul analisis ────────────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
try:
    from gecar_nlp import (
        run_full_pipeline,
        run_nmf_topics,
        run_bertopic_topics,
        extract_tfidf_phrases,
        PILAR_KEYWORDS,
    )
    NLP_AVAILABLE = True
except ImportError as e:
    NLP_AVAILABLE = False
    st.error(f"Gagal import gecar_nlp.py: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GECAR NLP Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
.css-1d391kg { background: #0f172a; }
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; font-size: 12px; }

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label { color: #94a3b8; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { color: #f1f5f9; font-size: 28px; font-weight: 700; margin: 4px 0; }
.metric-delta { font-size: 11px; }

/* ── Section Headers ── */
.section-header {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 4px;
}

/* ── Gap Badge ── */
.gap-critical { background: #7f1d1d; color: #fca5a5; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
.gap-high     { background: #7c2d12; color: #fdba74; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
.gap-med      { background: #713f12; color: #fde68a; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
.gap-low      { background: #14532d; color: #86efac; padding: 2px 8px; border-radius: 4px; font-size: 11px; }

/* ── Verbatim Box ── */
.verbatim-box {
    border-left: 3px solid #3b82f6;
    background: #1e293b;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-style: italic;
    color: #cbd5e1;
    font-size: 13px;
    line-height: 1.6;
}
.verbatim-meta {
    font-size: 10px;
    color: #64748b;
    margin-top: 6px;
    font-style: normal;
}

/* ── Tabs ── */
button[data-baseweb="tab"] { font-weight: 600; }

/* ── Narsum Badge ── */
.narsum-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & CACHE DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="⚙️ Memproses data NLP — harap tunggu...")
def load_pipeline():
    return run_full_pipeline(PATH_KELOMPOK, PATH_KII, top_n_tfidf=12)


@st.cache_data(show_spinner="🔎 Menjalankan Topic Modeling...")
def get_topics_cached(source_label: str, wilayah: str, pilar: str):
    if source_label == "Masyarakat":
        df = results["df_masy"]
    else:
        df = results["df_pem"]
    topics, method = run_bertopic_topics(df, wilayah, pilar)
    return topics, method


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 GECAR NLP")
    st.markdown("##### Analisis Konteks & Konflik")
    st.markdown("---")

    # Cek apakah file ditemukan
    if not os.path.exists(PATH_KELOMPOK):
        st.error(f"File tidak ditemukan:\n`{PATH_KELOMPOK}`\n\nLetakkan CSV di folder yang sama dengan `app.py`.")
        st.stop()

    # Load data
    try:
        results = load_pipeline()
    except Exception as e:
        st.error(f"Error saat memproses data: {e}")
        st.exception(e)
        st.stop()

    df_masy = results["df_masy"]
    df_pem  = results["df_pem"]
    gap_df  = results["gap_df"]
    wilayah_all = results["wilayah_list"]
    pilar_all   = results["pilar_list"]

    st.markdown("### 🗺️ Filter Global")

    sel_wilayah = st.selectbox(
        "Wilayah",
        options=["Semua"] + wilayah_all,
        key="sel_wilayah",
    )
    sel_pilar = st.selectbox(
        "Pilar Sektoral",
        options=["Semua"] + pilar_all,
        key="sel_pilar",
    )

    st.markdown("---")
    st.markdown("### 📊 Info Dataset")

    total_masy = len(df_masy)
    total_pem  = len(df_pem)
    total_wil  = len(wilayah_all)

    st.metric("Total Tanggapan Masyarakat", total_masy)
    st.metric("Total Tanggapan Pemerintah", total_pem)
    st.metric("Wilayah Tercakup", total_wil)
    st.metric("Pilar Sektoral", 5)

    st.markdown("---")
    st.markdown("### 🔧 Pengaturan Analisis")
    top_n_show = st.slider("Top N Frasa TF-IDF ditampilkan", 5, 15, 10)
    n_topics   = st.slider("Jumlah Topik (NMF)", 3, 8, 5)

    st.markdown("---")
    st.caption("GECAR NLP v1.0 | Senior Data Analyst Pipeline")
    st.caption("Powered by: TF-IDF + NMF + BERTopic")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown("# 🔍 GECAR Context Analysis Dashboard")
    st.markdown(
        "**Gap Analysis** antara Keluhan Masyarakat *(Bottom-Up)* "
        "dan Respons Pemerintah *(Top-Down)* | "
        f"Wilayah: **{sel_wilayah}** | Pilar: **{sel_pilar}**"
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TABS NAVIGASI
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Overview & Gap Matrix",
    "📈 TF-IDF Diverging Chart",
    "🧠 Topic Modeling",
    "🧵 Benang Merah",
    "💬 Verbatim Explorer",
    "🎨 Narrative / Creative View",
])


# ═══════════════════════════════════════════════
# TAB 1: OVERVIEW & GAP MATRIX
# ═══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="section-header">📊 Matriks Gap: Wilayah × Pilar Sektoral</p>', unsafe_allow_html=True)
    st.markdown("Gap Score mendekati **1.0** = prioritas sangat berbeda antara Masyarakat & Pemerintah.")

    # ── Heatmap ──
    pivot_gap = gap_df.pivot(index="Wilayah", columns="Pilar_Sektoral", values="Gap_Score")

    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot_gap.values,
        x=pivot_gap.columns.tolist(),
        y=pivot_gap.index.tolist(),
        colorscale=[
            [0.0, "#14532d"],
            [0.25, "#15803d"],
            [0.5, "#ca8a04"],
            [0.75, "#dc2626"],
            [1.0, "#7f1d1d"],
        ],
        zmin=0, zmax=1,
        text=pivot_gap.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 13, "color": "white"},
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>Gap Score: %{z:.3f}<extra></extra>",
    ))
    fig_heat.update_layout(
        title="🔴 Heatmap Gap Score — Merah = Kesenjangan Tinggi",
        height=400,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(tickangle=-30, title="Pilar Sektoral"),
        yaxis=dict(title="Wilayah"),
        coloraxis_colorbar=dict(
            title="Gap Score",
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["0 Selaras", "0.25", "0.5", "0.75", "1.0 Kritis"],
        ),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Summary Metrics ──
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    avg_gap = gap_df["Gap_Score"].mean()
    max_gap_row = gap_df.loc[gap_df["Gap_Score"].idxmax()]
    min_gap_row = gap_df.loc[gap_df["Gap_Score"].idxmin()]
    kritis_count = (gap_df["Gap_Score"] >= 0.7).sum()

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-rata Gap</div>
            <div class="metric-value" style="color:{'#ef4444' if avg_gap>0.6 else '#f59e0b' if avg_gap>0.4 else '#22c55e'}">{avg_gap:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sel Kritis (≥0.7)</div>
            <div class="metric-value" style="color:#ef4444">{kritis_count}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Gap Tertinggi</div>
            <div class="metric-value" style="color:#ef4444">{max_gap_row['Gap_Score']:.2f}</div>
            <div class="metric-delta" style="color:#94a3b8">{max_gap_row['Wilayah']} / {max_gap_row['Pilar_Sektoral'][:20]}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Gap Terendah</div>
            <div class="metric-value" style="color:#22c55e">{min_gap_row['Gap_Score']:.2f}</div>
            <div class="metric-delta" style="color:#94a3b8">{min_gap_row['Wilayah']} / {min_gap_row['Pilar_Sektoral'][:20]}</div>
        </div>""", unsafe_allow_html=True)

    # ── Tabel Detail Gap ──
    st.markdown("---")
    st.markdown("#### 📋 Tabel Detail Gap Analysis")

    gap_filtered = gap_df.copy()
    if sel_wilayah != "Semua":
        gap_filtered = gap_filtered[gap_filtered["Wilayah"] == sel_wilayah]
    if sel_pilar != "Semua":
        gap_filtered = gap_filtered[gap_filtered["Pilar_Sektoral"] == sel_pilar]

    gap_display = gap_filtered[[
        "Wilayah", "Pilar_Sektoral", "Gap_Score", "N_Overlap",
        "Frasa_Top_Masy", "Frasa_Top_Pem", "Interpretasi"
    ]].sort_values("Gap_Score", ascending=False)

    def color_gap(val):
        try:
            v = float(val)
            if v >= 0.9: return "background-color: #7f1d1d; color: #fca5a5"
            elif v >= 0.7: return "background-color: #78350f; color: #fdba74"
            elif v >= 0.5: return "background-color: #713f12; color: #fde68a"
            elif v >= 0.25: return "background-color: #14532d; color: #86efac"
            else: return "background-color: #052e16; color: #4ade80"
        except (TypeError, ValueError):
            return ""

    st.dataframe(
        gap_display.style.map(color_gap, subset=["Gap_Score"]),
        use_container_width=True,
        height=350,
    )


# ═══════════════════════════════════════════════
# TAB 2: TF-IDF DIVERGING BAR CHART
# ═══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="section-header">📈 TF-IDF Diverging Bar Chart</p>', unsafe_allow_html=True)
    st.markdown(
        "Frasa dominan versi **Masyarakat** (kiri, biru) vs **Pemerintah** (kanan, merah). "
        "Semakin ke luar = semakin dominan frasa tersebut dalam narasi masing-masing."
    )

    # Ambil data TF-IDF sesuai filter
    frasa_masy_df = extract_tfidf_phrases(
        df_masy,
        wilayah=sel_wilayah if sel_wilayah != "Semua" else None,
        pilar=sel_pilar if sel_pilar != "Semua" else None,
        top_n=top_n_show,
        ngram_range=(2, 3),
    )
    frasa_pem_df = extract_tfidf_phrases(
        df_pem,
        wilayah=sel_wilayah if sel_wilayah != "Semua" else None,
        pilar=sel_pilar if sel_pilar != "Semua" else None,
        top_n=top_n_show,
        ngram_range=(2, 3),
    )

    if frasa_masy_df.empty and frasa_pem_df.empty:
        st.warning("Tidak ada data untuk kombinasi filter ini.")
    else:
        # Buat diverging chart
        fig_div = go.Figure()

        if not frasa_masy_df.empty:
            fig_div.add_trace(go.Bar(
                y=frasa_masy_df["Frasa"],
                x=-frasa_masy_df["Skor_TFIDF"],  # negative for left side
                orientation="h",
                name="Masyarakat",
                marker_color="#3b82f6",
                hovertemplate="<b>%{y}</b><br>Skor: %{customdata:.4f}<extra>Masyarakat</extra>",
                customdata=frasa_masy_df["Skor_TFIDF"],
                text=frasa_masy_df["Frasa"],
                textposition="outside",
            ))

        if not frasa_pem_df.empty:
            fig_div.add_trace(go.Bar(
                y=frasa_pem_df["Frasa"],
                x=frasa_pem_df["Skor_TFIDF"],  # positive for right side
                orientation="h",
                name="Pemerintah (KII)",
                marker_color="#ef4444",
                hovertemplate="<b>%{y}</b><br>Skor: %{x:.4f}<extra>Pemerintah</extra>",
                text=frasa_pem_df["Frasa"],
                textposition="outside",
            ))

        max_val = max(
            frasa_masy_df["Skor_TFIDF"].max() if not frasa_masy_df.empty else 0.01,
            frasa_pem_df["Skor_TFIDF"].max() if not frasa_pem_df.empty else 0.01,
        )

        fig_div.update_layout(
            barmode="overlay",
            height=max(400, top_n_show * 38),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                title="← Masyarakat | Pemerintah →",
                range=[-max_val * 1.4, max_val * 1.4],
                tickformat=".3f",
                zeroline=True,
                zerolinecolor="#475569",
                zerolinewidth=2,
            ),
            yaxis=dict(title="", automargin=True),
            title=f"Perbandingan Frasa Dominan | {sel_wilayah} | {sel_pilar}",
            showlegend=True,
        )

        # Garis tengah label
        fig_div.add_vline(x=0, line_width=2, line_color="#475569")
        fig_div.add_annotation(x=-max_val * 0.6, y=1.06, text="← MASYARAKAT",
                               showarrow=False, font=dict(color="#3b82f6", size=12),
                               xref="x", yref="paper")
        fig_div.add_annotation(x=max_val * 0.6, y=1.06, text="PEMERINTAH →",
                               showarrow=False, font=dict(color="#ef4444", size=12),
                               xref="x", yref="paper")

        st.plotly_chart(fig_div, use_container_width=True)

    # ── Side-by-side table ──
    st.markdown("---")
    st.markdown("#### 📋 Tabel Perbandingan Frasa")
    col_m, col_p = st.columns(2)

    with col_m:
        st.markdown("**🔵 Masyarakat — Frasa Teratas**")
        if not frasa_masy_df.empty:
            st.dataframe(frasa_masy_df[["Frasa", "Skor_TFIDF", "Frekuensi"]],
                         use_container_width=True, height=350)
        else:
            st.info("Tidak ada data.")

    with col_p:
        st.markdown("**🔴 Pemerintah (KII) — Frasa Teratas**")
        if not frasa_pem_df.empty:
            st.dataframe(frasa_pem_df[["Frasa", "Skor_TFIDF", "Frekuensi"]],
                         use_container_width=True, height=350)
        else:
            st.info("Tidak ada data.")

    # ── Frasa overlap highlight ──
    if not frasa_masy_df.empty and not frasa_pem_df.empty:
        set_m = set(frasa_masy_df["Frasa"].tolist())
        set_p = set(frasa_pem_df["Frasa"].tolist())
        overlap = set_m & set_p
        if overlap:
            st.success(f"✅ **Frasa yang sama di kedua pihak ({len(overlap)}):** {' | '.join(sorted(overlap))}")
        else:
            st.error("❌ **Tidak ada frasa yang tumpang tindih** — indikasi mismatch narasi yang kuat.")


# ═══════════════════════════════════════════════
# TAB 3: TOPIC MODELING
# ═══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-header">🧠 Topic Modeling — Klasterisasi Semantik</p>', unsafe_allow_html=True)
    st.markdown(
        "Menggunakan **NMF** (selalu tersedia) atau **BERTopic + IndoBERT** (jika terinstall). "
        "Topik mengagregasi tanggapan dengan *makna laten* yang sama meski kosakata berbeda."
    )

    col_src, col_meth = st.columns([2, 2])
    with col_src:
        topic_source = st.radio("Sumber data:", ["Masyarakat", "Pemerintah (KII)"], horizontal=True)
    with col_meth:
        use_bertopic = st.toggle("Coba BERTopic + IndoBERT", value=False,
                                  help="Membutuhkan: pip install bertopic sentence-transformers")

    df_topic = df_masy if topic_source == "Masyarakat" else df_pem

    if st.button("▶ Jalankan Topic Modeling", type="primary"):
        with st.spinner("Menjalankan analisis topik..."):
            if use_bertopic:
                topics, method_used = run_bertopic_topics(
                    df_topic,
                    wilayah=sel_wilayah if sel_wilayah != "Semua" else None,
                    pilar=sel_pilar if sel_pilar != "Semua" else None,
                )
            else:
                from gecar_nlp import run_nmf_topics
                topics = run_nmf_topics(
                    df_topic,
                    wilayah=sel_wilayah if sel_wilayah != "Semua" else None,
                    pilar=sel_pilar if sel_pilar != "Semua" else None,
                    n_topics=n_topics,
                )
                method_used = "NMF (Sklearn)"

        st.info(f"✅ Metode yang digunakan: **{method_used}**")

        if not topics:
            st.warning("Tidak cukup data untuk membuat topik dengan filter ini.")
        else:
            # ── Bubble Chart topik ──
            topic_labels = [t["label_otomatis"] for t in topics]
            topic_counts = [t["n_dokumen"] for t in topics]
            topic_kw     = [", ".join(t["kata_kunci"][:5]) for t in topics]

            fig_bubble = px.scatter(
                x=range(len(topics)),
                y=topic_counts,
                size=topic_counts,
                color=topic_labels,
                hover_name=topic_labels,
                hover_data={"Keywords": topic_kw},
                labels={"x": "Klaster ID", "y": "Jumlah Dokumen"},
                title=f"Peta Topik — {topic_source} | {sel_wilayah} | {sel_pilar}",
                size_max=60,
            )
            fig_bubble.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#e2e8f0", height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

            # ── Detail per topik ──
            st.markdown("---")
            for t in topics:
                with st.expander(f"🗂 {t['label_otomatis']} — {t['n_dokumen']} dokumen"):
                    st.markdown(f"**Kata Kunci:** `{'` | `'.join(t['kata_kunci'])}`")
                    if t["dokumen_sampel"]:
                        st.markdown("**Contoh Tanggapan:**")
                        for doc in t["dokumen_sampel"]:
                            st.markdown(f'<div class="verbatim-box">{doc}</div>',
                                       unsafe_allow_html=True)
    else:
        st.info("👆 Klik **Jalankan Topic Modeling** untuk memulai analisis klaster semantik.")


# ═══════════════════════════════════════════════
# TAB 4: BENANG MERAH
# ═══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-header">🧵 Benang Merah — Tema Berulang Lintas Pilar</p>', unsafe_allow_html=True)
    st.markdown(
        "Identifikasi kata/frasa yang muncul secara konsisten di berbagai Pilar Sektoral — "
        "ini adalah **isu sistemik** yang melampaui satu kategori saja."
    )

    benang_merah = results["benang_merah"]

    # ── Bar chart Benang Merah ──
    bm_key = sel_wilayah if sel_wilayah in benang_merah else "Semua Wilayah"
    bm_data = benang_merah.get(bm_key, {})

    if bm_data:
        bm_df = pd.DataFrame(list(bm_data.items()), columns=["Tema", "Skor"])
        bm_df = bm_df.sort_values("Skor", ascending=True).tail(20)

        fig_bm = go.Figure(go.Bar(
            x=bm_df["Skor"],
            y=bm_df["Tema"],
            orientation="h",
            marker=dict(
                color=bm_df["Skor"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="TF-IDF Score"),
            ),
            hovertemplate="<b>%{y}</b><br>Skor: %{x:.4f}<extra></extra>",
        ))
        fig_bm.update_layout(
            title=f"🧵 Benang Merah — {bm_key}",
            height=500,
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
            xaxis_title="TF-IDF Score (semakin tinggi = semakin dominan)",
            yaxis_title="",
        )
        st.plotly_chart(fig_bm, use_container_width=True)
    else:
        st.warning("Tidak ada data benang merah untuk wilayah ini.")

    # ── Komparasi lintas wilayah ──
    st.markdown("---")
    st.markdown("#### 🗺️ Perbandingan Benang Merah Lintas Wilayah")

    all_themes = {}
    for wil in results["wilayah_list"]:
        bm = benang_merah.get(wil, {})
        for tema, skor in list(bm.items())[:10]:
            if tema not in all_themes:
                all_themes[tema] = {}
            all_themes[tema][wil] = skor

    # Tema yang muncul di > 1 wilayah = sistemik
    sistemik = {t: v for t, v in all_themes.items() if len(v) >= 2}

    if sistemik:
        sistemik_df_rows = []
        for tema, wil_dict in sistemik.items():
            for wil, skor in wil_dict.items():
                sistemik_df_rows.append({"Tema": tema, "Wilayah": wil, "Skor": skor})

        sistemik_df = pd.DataFrame(sistemik_df_rows)
        pivot_sis = sistemik_df.pivot(index="Tema", columns="Wilayah", values="Skor").fillna(0)

        fig_sis = px.imshow(
            pivot_sis,
            color_continuous_scale="Blues",
            title="Tema Sistemik (Muncul di ≥2 Wilayah) — Intensitas TF-IDF",
            labels=dict(x="Wilayah", y="Tema Berulang", color="Skor"),
            aspect="auto",
        )
        fig_sis.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#e2e8f0", height=400,
        )
        st.plotly_chart(fig_sis, use_container_width=True)

        st.success(f"✅ **{len(sistemik)} tema sistemik** ditemukan lintas wilayah: "
                   f"`{'` | `'.join(list(sistemik.keys())[:8])}`")
    else:
        st.info("Belum cukup data untuk menemukan pola lintas-wilayah.")

    # ── Distribusi Pilar per Wilayah ──
    st.markdown("---")
    st.markdown("#### 📊 Distribusi Tanggapan per Pilar & Wilayah")

    dist_df = df_masy.groupby(["Wilayah", "Pilar_Sektoral"]).size().reset_index(name="N")
    fig_dist = px.bar(
        dist_df, x="Wilayah", y="N", color="Pilar_Sektoral",
        barmode="group",
        title="Volume Tanggapan Masyarakat per Pilar & Wilayah",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_dist.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font_color="#e2e8f0", height=350,
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 5: VERBATIM EXPLORER
# ═══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="section-header">💬 Verbatim Explorer — Kutipan Langsung</p>', unsafe_allow_html=True)
    st.markdown("Telusuri tanggapan mentah (verbatim) dengan filter multi-dimensi.")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        vb_source = st.selectbox("Sumber", ["Masyarakat", "Pemerintah (KII)", "Keduanya"])
    with col_f2:
        vb_wilayah = st.selectbox("Wilayah", ["Semua"] + wilayah_all, key="vb_wil")
    with col_f3:
        vb_pilar = st.selectbox("Pilar", ["Semua"] + pilar_all, key="vb_pil")
    with col_f4:
        vb_search = st.text_input("🔍 Cari kata kunci:", placeholder="misal: konflik, pangan")

    # Filter data
    def filter_verbatim(df_in, wil, pil, search_kw):
        df_f = df_in.copy()
        if wil != "Semua":
            df_f = df_f[df_f["Wilayah"] == wil]
        if pil != "Semua":
            df_f = df_f[df_f["Pilar_Sektoral"] == pil]
        if search_kw:
            mask = df_f["Tanggapan"].str.lower().str.contains(search_kw.lower(), na=False)
            df_f = df_f[mask]
        return df_f

    if vb_source in ["Masyarakat", "Keduanya"]:
        vb_masy = filter_verbatim(df_masy, vb_wilayah, vb_pilar, vb_search)
    else:
        vb_masy = pd.DataFrame()

    if vb_source in ["Pemerintah (KII)", "Keduanya"]:
        vb_pem = filter_verbatim(df_pem, vb_wilayah, vb_pilar, vb_search)
    else:
        vb_pem = pd.DataFrame()

    # Statistik filter
    total_found = len(vb_masy) + len(vb_pem)
    st.markdown(f"**{total_found} tanggapan ditemukan**")

    if total_found == 0:
        st.warning("Tidak ada tanggapan yang sesuai dengan filter.")
    else:
        # Tampilkan dalam expander per sumber
        if not vb_masy.empty:
            st.markdown("##### 🔵 Tanggapan Masyarakat")
            show_n = min(20, len(vb_masy))
            for _, row in vb_masy.head(show_n).iterrows():
                demo_info = ""
                if "Kelompok_Usia" in row:
                    demo_info = f"{row.get('Kelompok_Usia','')} · {row.get('Jenis_Kelamin','')} · {row.get('Kategori_Responden','')}"

                st.markdown(f"""
                <div class="verbatim-box">
                    {row['Tanggapan']}
                    <div class="verbatim-meta">
                        📍 {row['Wilayah']} &nbsp;|&nbsp; 🏷 {row['Pilar_Sektoral']}
                        &nbsp;|&nbsp; 👤 {demo_info}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if len(vb_masy) > show_n:
                st.caption(f"... dan {len(vb_masy) - show_n} tanggapan lainnya")

        if not vb_pem.empty:
            st.markdown("##### 🔴 Tanggapan Pemerintah (KII)")
            show_n_p = min(15, len(vb_pem))
            for _, row in vb_pem.head(show_n_p).iterrows():
                narsum = row.get("Narsum", "")
                st.markdown(f"""
                <div class="verbatim-box" style="border-left-color: #ef4444;">
                    {row['Tanggapan']}
                    <div class="verbatim-meta">
                        📍 {row['Wilayah']} &nbsp;|&nbsp; 🏷 {row['Pilar_Sektoral']}
                        &nbsp;|&nbsp; <span class="narsum-badge">{narsum}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if len(vb_pem) > show_n_p:
                st.caption(f"... dan {len(vb_pem) - show_n_p} tanggapan lainnya")

    # ── Demografi breakdown ──
    if not vb_masy.empty and "Kelompok_Usia" in vb_masy.columns:
        st.markdown("---")
        st.markdown("#### 📊 Breakdown Demografi (Masyarakat)")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            usia_count = vb_masy["Kelompok_Usia"].value_counts().reset_index()
            usia_count.columns = ["Kelompok_Usia", "N"]
            fig_usia = px.pie(usia_count, names="Kelompok_Usia", values="N",
                              title="Distribusi Usia", hole=0.4,
                              color_discrete_sequence=["#3b82f6", "#8b5cf6"])
            fig_usia.update_layout(paper_bgcolor="#0f172a", font_color="#e2e8f0",
                                   height=280, showlegend=True)
            st.plotly_chart(fig_usia, use_container_width=True)

        with col_d2:
            gender_count = vb_masy["Jenis_Kelamin"].value_counts().reset_index()
            gender_count.columns = ["Jenis_Kelamin", "N"]
            fig_gender = px.bar(gender_count, x="Jenis_Kelamin", y="N",
                                title="Distribusi Jenis Kelamin",
                                color="Jenis_Kelamin",
                                color_discrete_sequence=["#06b6d4", "#f472b6"])
            fig_gender.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                                     font_color="#e2e8f0", height=280, showlegend=False)
            st.plotly_chart(fig_gender, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 6: NARRATIVE / CREATIVE VIEW
# ═══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="section-header">🎨 Narrative & Creative View</p>', unsafe_allow_html=True)
    st.markdown("Panel khusus untuk **tim kreatif/desainer** — mengubah temuan analisis menjadi konsep visual naratif.")

    creative_view = st.radio(
        "Pilih konsep:",
        ["Two-Faces Illustration (Infografis Karikatur)",
         "Comic Strip (Berbasis Verbatim)",
         "Gap Score Radial Chart",
         "Sistem Peringatan Dini (Status Board)"],
        horizontal=False,
    )

    # ── Two Faces Illustration ──
    if "Two-Faces" in creative_view:
        st.markdown("---")
        st.markdown("### 🎭 'Two Faces' — Kontras Realita vs Narasi Makro")
        st.markdown("""
        > **Konsep Infografis Karikatur** untuk tim desainer.
        > Dua wajah bertolak belakang: wajah kiri = warga lapangan, wajah kanan = pejabat pemerintah.
        > Di antara keduanya = benang merah isu yang seharusnya menjadi jembatan.
        """)

        # Ambil data untuk dua faces
        col_tw1, col_tw2, col_tw3 = st.columns([5, 1, 5])

        # Kiri: voice of people
        subset_m = df_masy.copy()
        if sel_wilayah != "Semua":
            subset_m = subset_m[subset_m["Wilayah"] == sel_wilayah]
        if sel_pilar != "Semua":
            subset_m = subset_m[subset_m["Pilar_Sektoral"] == sel_pilar]

        sample_masy = subset_m["Tanggapan"].dropna().sample(min(3, len(subset_m)),
                                                              random_state=42).tolist()

        # Kanan: voice of government
        subset_p = df_pem.copy()
        if sel_wilayah != "Semua":
            subset_p = subset_p[subset_p["Wilayah"] == sel_wilayah]
        if sel_pilar != "Semua":
            subset_p = subset_p[subset_p["Pilar_Sektoral"] == sel_pilar]

        sample_pem = subset_p["Tanggapan"].dropna().sample(min(2, len(subset_p)),
                                                             random_state=42).tolist()

        with col_tw1:
            st.markdown("#### 👥 Suara Warga")
            st.markdown("*(Realita Lapangan — Bottom-Up)*")
            for i, quote in enumerate(sample_masy):
                short_q = quote[:180] + "..." if len(quote) > 180 else quote
                st.markdown(f"""
                <div class="verbatim-box" style="background: #172554; border-left-color: #60a5fa;">
                    <em>"{short_q}"</em>
                    <div class="verbatim-meta">— Warga {sel_wilayah if sel_wilayah != 'Semua' else ''}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: #1e3a5f; border-radius: 8px; padding: 12px; margin-top: 12px;">
                <strong style="color: #93c5fd;">Isu Utama Warga:</strong><br>
                <span style="color: #bfdbfe; font-size: 13px;">
                    {frasa_masy_df['Frasa'].head(5).str.title().tolist() if not frasa_masy_df.empty else ["(data tidak cukup)"]}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col_tw2:
            st.markdown("<br><br><br><br>", unsafe_allow_html=True)
            st.markdown("### ⟺")
            st.markdown("**GAP**")

            # Gap score untuk kombinasi ini
            gap_row = gap_df[
                (gap_df["Pilar_Sektoral"] == sel_pilar if sel_pilar != "Semua" else True) &
                (gap_df["Wilayah"] == sel_wilayah if sel_wilayah != "Semua" else True)
            ]
            if not gap_row.empty:
                avg_g = gap_row["Gap_Score"].mean()
                color = "#ef4444" if avg_g > 0.6 else "#f59e0b" if avg_g > 0.4 else "#22c55e"
                st.markdown(f"<h2 style='color:{color};text-align:center'>{avg_g:.2f}</h2>", unsafe_allow_html=True)

        with col_tw3:
            st.markdown("#### 🏛️ Narasi Pemerintah")
            st.markdown("*(Perspektif Makro — Top-Down)*")
            for i, quote in enumerate(sample_pem):
                short_q = quote[:180] + "..." if len(quote) > 180 else quote
                st.markdown(f"""
                <div class="verbatim-box" style="background: #450a0a; border-left-color: #f87171;">
                    <em>"{short_q}"</em>
                    <div class="verbatim-meta">— KII, {sel_wilayah if sel_wilayah != 'Semua' else ''}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: #3f1d1d; border-radius: 8px; padding: 12px; margin-top: 12px;">
                <strong style="color: #fca5a5;">Framing Pemerintah:</strong><br>
                <span style="color: #fecaca; font-size: 13px;">
                    {frasa_pem_df['Frasa'].head(5).str.title().tolist() if not frasa_pem_df.empty else ["(data tidak cukup)"]}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.info("""
        **📋 Brief untuk Tim Desainer:**
        - **Format:** Infografis A3 landscape, dua kolom, gaya karikatur editorial
        - **Visual kiri:** Ilustrasi wajah warga (ekspresi lelah/khawatir), latar kampung
        - **Visual kanan:** Ilustrasi wajah pejabat (ekspresi formal), latar kantor/peta
        - **Tengah:** Tembok dengan celah — di celah tertulis kata-kata yang jadi Gap
        - **Palet:** Biru dingin (warga) vs Merah birokrasi, dengan jembatan warna hijau untuk overlap
        - **Tipografi:** Verbatim dikutip langsung dengan font serif, data angka dengan sans-serif bold
        """)

    # ── Comic Strip ──
    elif "Comic Strip" in creative_view:
        st.markdown("---")
        st.markdown("### 📖 Comic Strip — Storyboard Berbasis Data")

        # Pilih wilayah spesifik untuk comic
        comic_wil = st.selectbox("Wilayah untuk Comic Strip:", wilayah_all, key="comic_wil")

        df_m_comic = df_masy[df_masy["Wilayah"] == comic_wil].copy()
        df_p_comic = df_pem[df_pem["Wilayah"] == comic_wil].copy() if comic_wil in df_pem["Wilayah"].values else pd.DataFrame()

        panels = []
        for pilar in pilar_all[:4]:  # 4 panel comic
            m_quotes = df_m_comic[df_m_comic["Pilar_Sektoral"] == pilar]["Tanggapan"].dropna()
            p_quotes = df_p_comic[df_p_comic["Pilar_Sektoral"] == pilar]["Tanggapan"].dropna() if not df_p_comic.empty else pd.Series()

            m_q = m_quotes.iloc[0][:120] + "..." if len(m_quotes) > 0 else "Data tidak tersedia"
            p_q = p_quotes.iloc[0][:120] + "..." if len(p_quotes) > 0 else "Tidak ada respons pemerintah"

            panels.append({"pilar": pilar, "warga": m_q, "pemerintah": p_q})

        cols_comic = st.columns(2)
        for i, panel in enumerate(panels):
            with cols_comic[i % 2]:
                st.markdown(f"""
                <div style="background: #1e293b; border: 2px solid #334155; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                    <div style="text-align:center; color: #64748b; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;">
                        Panel {i+1} · {panel['pilar']}
                    </div>
                    <hr style="border-color:#334155; margin: 8px 0">
                    <div style="background:#172554; padding:10px; border-radius:8px; margin-bottom:8px;">
                        <strong style="color:#93c5fd; font-size:11px;">👥 WARGA BERKATA:</strong><br>
                        <em style="color:#bfdbfe; font-size:12px;">"{panel['warga']}"</em>
                    </div>
                    <div style="text-align:center; font-size:18px; margin: 4px 0;">⬇️</div>
                    <div style="background:#1c1917; padding:10px; border-radius:8px;">
                        <strong style="color:#fca5a5; font-size:11px;">🏛️ PEMERINTAH MERESPONS:</strong><br>
                        <em style="color:#fecaca; font-size:12px;">"{panel['pemerintah']}"</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.info("**📋 Brief Comic Strip:** Setiap panel = 1 pilar isu. Karakter tetap: warga bernama 'Pak Nopius' (pria Jayawijaya, 40th) dan 'Bu Yane' (perempuan Asmat). Gaya: panel hitam-putih dengan highlight warna pada kata kunci kritis. Format: 4-6 panel horizontal, A4 landscape.")

    # ── Radial Gap Chart ──
    elif "Radial" in creative_view:
        st.markdown("---")
        st.markdown("### 🎯 Radar Chart — Profil Gap per Pilar")

        gap_filtered_r = gap_df.copy()
        if sel_wilayah != "Semua":
            gap_filtered_r = gap_filtered_r[gap_filtered_r["Wilayah"] == sel_wilayah]

        gap_by_pilar = gap_filtered_r.groupby("Pilar_Sektoral")["Gap_Score"].mean().reset_index()

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=gap_by_pilar["Gap_Score"].tolist(),
            theta=gap_by_pilar["Pilar_Sektoral"].tolist(),
            fill="toself",
            fillcolor="rgba(239,68,68,0.2)",
            line=dict(color="#ef4444", width=2),
            name="Gap Score",
            hovertemplate="<b>%{theta}</b><br>Gap: %{r:.2f}<extra></extra>",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#334155",
                                tickcolor="#64748b", tickfont=dict(color="#64748b")),
                angularaxis=dict(gridcolor="#334155"),
                bgcolor="#0f172a",
            ),
            paper_bgcolor="#0f172a",
            font_color="#e2e8f0",
            title=f"Radar Gap Profile — {sel_wilayah}",
            height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Rekomendasi intervensi berdasarkan pilar dengan gap tertinggi
        if not gap_by_pilar.empty:
            top_gap_pilar = gap_by_pilar.loc[gap_by_pilar["Gap_Score"].idxmax()]
            st.error(f"""
            🚨 **Rekomendasi Intervensi Prioritas:**
            Pilar **{top_gap_pilar['Pilar_Sektoral']}** memiliki gap tertinggi (skor: {top_gap_pilar['Gap_Score']:.2f}).
            Diperlukan dialog kebijakan segera antara masyarakat dan pemangku kebijakan di area ini.
            """)

    # ── Status Board ──
    else:
        st.markdown("---")
        st.markdown("### 🚦 Sistem Peringatan Dini — Status Board")
        st.markdown("Monitoring real-time status gap per wilayah dan pilar.")

        for wil in wilayah_all:
            st.markdown(f"#### 📍 {wil}")
            cols_sb = st.columns(len(pilar_all))
            for i, pil in enumerate(pilar_all):
                gap_row = gap_df[(gap_df["Wilayah"] == wil) & (gap_df["Pilar_Sektoral"] == pil)]
                gap_val = gap_row["Gap_Score"].values[0] if not gap_row.empty else 0.0

                if gap_val >= 0.9:
                    icon, color, status = "🔴", "#7f1d1d", "KRITIS"
                elif gap_val >= 0.7:
                    icon, color, status = "🟠", "#7c2d12", "TINGGI"
                elif gap_val >= 0.5:
                    icon, color, status = "🟡", "#713f12", "SEDANG"
                else:
                    icon, color, status = "🟢", "#14532d", "AMAN"

                with cols_sb[i]:
                    st.markdown(f"""
                    <div style="background:{color}; border-radius:8px; padding:10px; text-align:center; margin-bottom:4px;">
                        <div style="font-size:20px">{icon}</div>
                        <div style="color:white; font-size:9px; font-weight:700;">{pil[:15]}</div>
                        <div style="color:white; font-size:14px; font-weight:700;">{gap_val:.2f}</div>
                        <div style="color:rgba(255,255,255,0.7); font-size:9px;">{status}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Legenda:**")
        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        col_l1.markdown("🔴 **KRITIS** ≥0.9 — Tindakan segera")
        col_l2.markdown("🟠 **TINGGI** ≥0.7 — Perlu perhatian")
        col_l3.markdown("🟡 **SEDANG** ≥0.5 — Monitor rutin")
        col_l4.markdown("🟢 **AMAN** <0.5 — Relatif selaras")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:11px; padding: 8px;">
    GECAR NLP Analysis Dashboard &nbsp;|&nbsp;
    Pipeline: <strong>Sastrawi Stopwords → TF-IDF N-Gram(2,3) → NMF/BERTopic → Gap Analysis</strong> &nbsp;|&nbsp;
    ⚠️ PERINGATAN: Tidak ada stemming — struktur kata berimbuhan dijaga penuh untuk sensitivitas konteks
</div>
""", unsafe_allow_html=True)
