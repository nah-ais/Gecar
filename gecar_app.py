"""
GECAR - Analisa Dataset Masyarakat & KII
Streamlit Dashboard - Data Scientist Edition
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="GECAR Dashboard Analisa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ─ Main container ─ */
        .main { 
            background-color: #F7F9FC; 
        }

        /* ─ Metric cards ─ */
        div[data-testid="metric-container"] {
            background: white;
            border-radius: 12px;
            padding: 18px 22px;
            border-left: 5px solid #1A73E8;
            box-shadow: 0 2px 8px rgba(0,0,0,.07);
        }

        /* ─ Section headers ─ */
        .section-header {
            background: linear-gradient(135deg, #1A73E8 0%, #0D47A1 100%);
            color: white;
            padding: 14px 22px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 18px;
        }

        /* ─ Insight boxes ─ */
        .insight-box {
            background: #E8F4FD;
            border-left: 4px solid #1A73E8;
            padding: 14px 18px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 14px;
            color: #1A237E;
        }
        .insight-box-warning {
            background: #FFF3E0;
            border-left: 4px solid #FF6F00;
            padding: 14px 18px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 14px;
            color: #E65100;
        }
        .insight-box-success {
            background: #E8F5E9;
            border-left: 4px solid #2E7D32;
            padding: 14px 18px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 14px;
            color: #1B5E20;
        }

        /* ─ Dimension badge ─ */
        .dim-badge {
            display: inline-block;
            background: #1A73E8;
            color: white;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            margin: 3px;
        }

        /* ─ Tab styling ─ */
        div[data-testid="stTabs"] button {
            font-weight: 600;
        }

        /* ─ Sidebar ─ */
        section[data-testid="stSidebar"] {
            background: #0D47A1;
        }
        section[data-testid="stSidebar"] * {
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df_k = pd.read_csv("Gecar_-_Kelompok.csv")
    df_kii = pd.read_csv("Gecar_-_KII.csv")
    return df_k, df_kii


# ──────────────────────────────────────────────
# DIMENSION MAPPING
# ──────────────────────────────────────────────
KELOMPOK_DIMS = {
    "🏠 Kehidupan Sehari-hari": [
        "Kehidupan sehari-hari bikin apa?",
        "Bisakah Anda menggambarkan seperti apa kehidupan sehari-harimu?",
    ],
    "🔄 Perubahan Terkini": [
        "Apa yang berubah dalam beberapa bulan terakhir?",
        "Apa yang telah berubah dalam beberapa bulan terakhir? Bagaimana menurut Anda hal itu telah mempengaruhi Anda dan komunitas?",
    ],
    "🎯 Kebutuhan Utama": [
        "Apa saja kebutuhan utama saat ini dan yang akan datang di komunitas Anda?",
        "Apa kebutuhan utama di kampung",
        "Siapa yang akan Anda minta bantuan untuk mengatasi kebutuhan-kebutuhan ini?",
        "Siapa yang akan Anda minta bantuan untuk mengatasi kebutuhan kebutuhan ini?",
    ],
    "⚡ Ketegangan & Konflik": [
        "Apa saja ketegangan yang terjadi di komunitas Anda yang mempengaruhi keluarga Anda?",
        "Bagaimana ketegangan-ketegangan tersebut muncul di komunitas Anda?",
        "Mengapa ketegangan ini terjadi?",
        "Bagaimana ketegangan ini berubah dalam beberapa bulan terakhir?",
        "Bagaimana ketegangan berubah dalam beberapa bulan terakhir?",
        "Bagaimana ketegangan tersebut muncul?",
    ],
    "🛡️ Keamanan & Ketakutan": [
        "Apa yang biasa bikin kamu rasa aman?",
        "Apa yang membuat Anda merasa aman?",
        "Apa ketakutanmu?",
        "Apa ketakutan Anda? Bagaimana Anda mengatasi situasi tersebut (misalnya, kepada siapa Anda pergi, apa yang Anda lakukan)?",
        "Apa ketakutanmu? Bagaimana kamu mengatasi situasi tersebut (misalnya, kepada siapa kamu pergi, apa yang kamu lakukan)?",
    ],
    "👥 Kelompok Rentan": [
        "Siapa saja kelompok rentan di komunitas Anda? Bagaimana peningkatan ketegangan akan berdampak pada kelompok-kelompok ini (terutama anak-anak dan remaja)?",
        "Siapa saja kelompok rentan/berisiko tinggi di komunitas Anda? Bagaimana peningkatan ketegangan akan berdampak pada kelompok-kelompok ini (terutama anak-anak dan remaja)?",
        "Siapa kelompok rentan/berisiko tinggi?",
    ],
    "🤝 Kohesi Sosial": [
        "Apa yang menyatukan atau memecah belah masyarakat di negara ini?",
        "Apa yang menyatukan atau memecah belah masyarakat di wilayah ini?",
        "Apa yang menyatukan/memecah belah masyarakat di wilayah ini?",
        "Di daerah-daerah yang mengalami ketegangan yang meningkat, apa saja contoh kelompok dan/atau kegiatan yang bertujuan untuk mengurangi ketegangan dan mempromosikan perdamaian?",
        "Di daerah-daerah yang mengalami ketegangan yang meningkat, apa saja contoh kelompok dan/atau kegiatan yang bertujuan untuk mengurangi ketegangan dan mempromosikan perdamaian??",
    ],
    "🏛️ Aktor & Tokoh": [
        "Siapa saja tokoh yang paling berpengaruh di wilayah ini?",
        "Siapa tokoh paling berpengaruh?",
        "Bagaimana masyarakat memandang para aktor tersebut",
        "Bagaimana masyarakat memandang para aktor tersebut (misalnya PBB, LSM, kelompok agama, pemerintah, kelompok bersenjata, pemimpin politik, dll.)?",
        "Bagaimana masyarakat memandang WVI?",
    ],
    "🔮 Skenario Masa Depan": [
        "Apa yang akan terjadi di lingkunganmu kedepan dalam waktu dekat?",
        "Apa yang menurut Anda akan terjadi di komunitas Anda dalam waktu dekat?",
        "Apa yang menurutmu akan terjadi di komunitasmu dalam waktu dekat?",
        "Apa yang menurut Anda mungkin terjadi dalam 6 bulan ke depan terkait dengan situasi politik",
        "Apa yang menurut Anda mungkin terjadi dalam 6 bulan ke depan terkait dengan situasi ekonomi",
        "Apa yang menurut Anda mungkin terjadi dalam 6 bulan ke depan terkait dengan situasi sosial",
        "Apa yang menurut Anda mungkin terjadi dalam 6 bulan ke depan terkait dengan Situasi Keamanan ",
        "Apa yang menurut Anda mungkin terjadi dalam 6 bulan ke depan terkait dengan Situasi politik",
        "Apa yang menurut Anda mungkin terjadi dalam 6 bulan ke depan terkait dengan Situasi ekonomi",
        "Apa yang menurut Anda mungkin terjadi dalam 6 bulan ke depan terkait dengan Situasi Sosial",
        "Apa yang menurut Anda mungkin terjadi dalam 6 bulan ke depan terkait dengan situasi keamanan",
        "Skenario 6 bulan ke depan:",
    ],
    "🌟 Harapan & Kontribusi": [
        "Apa harapanmu untuk komunitas di masa depan?",
        "Apa harapanmu untuk komunitasmu (kampung) di masa depan?",
        "Apa harapan Anda untuk komunitas Anda di masa depan?",
        "Bagaimana kamu sebagai anak muda bisa berkontribusi untuk mengubah situasi?",
        "Bagaimana menurut kamu pemuda dapat berkontribusi dalam mengubah situasi?",
        "Bagaimana menurut Anda pemuda dapat berkontribusi dalam mengubah situasi?",
    ],
    "📋 Rekomendasi & Program": [
        "Apa yang perlu dilakukan untuk mengatasi dampak merugikan yang mungkin timbul dari skenario-skenario tersebut? Dan oleh siapa?",
        "Apa yang perlu dilakukan dan oleh siapa?",
        "Masukan Program:",
        "Masukan dan Harapan untuk WVI",
    ],
}

KII_DIMS = {
    "🎯 Kebutuhan Utama": [
        "Apa saja kebutuhan utama saat ini dan yang akan datang?",
        "Apa saja kebutuhan utama saat ini dan yang akan datang??",
        "Apa saja kebutuhan utama saat ini dan yang akan datang di komunitas Anda?",
        "Siapa yang akan Anda minta bantuan untuk mengatasi kebutuhan-kebutuhan ini?",
    ],
    "⚡ Ketegangan & Konflik": [
        "Apa saja ketegangan yang ada di negara ini dan Bagaimana ketegangan-ketegangan tersebut muncul?",
        "Apa saja ketegangan yang ada di negara ini?",
        "Bagaimana ketegangan-ketegangan tersebut muncul?",
        "Bagaimana ketegangan tersebut muncul?",
        "Mengapa ketegangan ini terjadi?",
        "Mengapa ketegangan ini bisa terjadi?",
        "Bagaimana ketegangan ini berubah dalam beberapa bulan terakhir?",
        "Apa saja ketegangan yang terjadi di komunitas Anda yang mempengaruhi keluarga Anda?",
    ],
    "👥 Kelompok Rentan": [
        "Siapa saja kelompok rentan.berisiko tinggi ? Bagaimana peningkatan ketegangan akan mempengaruhi kelompok-kelompok tersebut, terutama anak-anak dan remaja?",
        "Siapa saja kelompok rentan berisiko tinggi ? Bagaimana peningkatan ketegangan akan mempengaruhi kelompok-kelompok tersebut, terutama anak-anak dan remaja?",
        "Siapa kelompok rentan/berisiko tinggi?",
        " Contoh kelompok/kegiatan yang mempromosikan perdamaian?",
    ],
    "🤝 Kohesi Sosial": [
        "Apa yang menyatukan atau memecah belah masyarakat di wilayah ini?",
        "Apa yang menyatukan masyarakat di wilayah ini?",
        "Apa yang memecah belah masyarakat di wilayah ini?",
        "Di daerah-daerah yang mengalami ketegangan yang meningkat, apa saja contoh kelompok dan/atau kegiatan yang bertujuan untuk mengurangi ketegangan dan mempromosikan perdamaian?",
    ],
    "🏛️ Aktor & Tokoh": [
        "Kelompok mana yang paling berpengaruh dan di mana mereka berada?",
        "Apa tujuan dan kemampuan yang mereka miliki?",
        "Bagaimana hubungan di antara mereka?",
        "Bagaimana Anda/bagaimana Anda melihat mereka?",
        "Bagaimana masyarakat memandang aktor (Pemerintah, LSM, dll)?",
    ],
    "🔮 Skenario Masa Depan": [
        "Apa yang Anda perkirakan akan terjadi dalam 6 bulan ke depan terkait dengan",
        "Apa yang Anda perkirakan akan terjadi dalam 6 bulan ke depan",
        "Skenario 6 bulan ke depan:",
    ],
    "🏢 Dampak & Operasional LSM": [
        "Apa konsekuensi/dampak konteks ini bagi operasional LSM",
        "Apa konsekuensi/dampak konteks ini bagi operasional LSM?",
        " Apa konsekuensinya terhadap Kegiatan respon bencana konflik/kegiatan pemberdayaan/kegiatan yang sifatnya membangun perdamaian",
        "Pesan publik dan swasta eksternal seperti apa yang perlu kami sampaikan",
        "Apa jenis kegiatan Pembinaan perdamaian yang dapat kami kontribusikan?",
        "Apa jenis kegiatan  Pembinaan perdamaian yang dapat kami kontribusikan?",
    ],
    "📋 Rekomendasi & Program": [
        "Apa yang perlu dilakukan dan oleh siapa?",
        "Apa yang bisa dilakukan Bersama dengan WVI:",
        "Pesan untuk WVI:",
    ],
}

# Shared dimensions for comparison
SHARED_DIMS = [
    "🎯 Kebutuhan Utama",
    "⚡ Ketegangan & Konflik",
    "👥 Kelompok Rentan",
    "🤝 Kohesi Sosial",
    "🏛️ Aktor & Tokoh",
    "🔮 Skenario Masa Depan",
    "📋 Rekomendasi & Program",
]

# Color palette
COLORS = {
    "kelompok": "#1A73E8",
    "kii": "#EA4335",
    "wilayah": ["#1A73E8", "#34A853", "#FBBC04"],
    "gender": {"Perempuan": "#E91E63", "Laki laki": "#1A73E8"},
    "usia": {"Anak": "#FF6D00", "Dewasa": "#6200EA"},
    "dim": px.colors.qualitative.Set3,
}


def assign_dimension(df, dim_map):
    """Assign dimension label to each row."""
    question_to_dim = {}
    for dim, questions in dim_map.items():
        for q in questions:
            question_to_dim[q.strip()] = dim

    df["Dimensi"] = df["Pertanyaan"].apply(
        lambda x: question_to_dim.get(str(x).strip(), "🔍 Lainnya")
    )
    return df


# ──────────────────────────────────────────────
# LOAD & PREPARE
# ──────────────────────────────────────────────
df_k, df_kii = load_data()
df_k = assign_dimension(df_k, KELOMPOK_DIMS)
df_kii = assign_dimension(df_kii, KII_DIMS)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 GECAR Dashboard")
    st.markdown("**Analisa Konteks Kemanusiaan**")
    st.markdown("---")

    page = st.radio(
        "📑 Navigasi",
        [
            "🏠 Ringkasan Eksekutif",
            "📋 Dataset Masyarakat",
            "🏛️ Dataset KII (Pemerintah)",
            "🔍 Analisa Dimensi",
            "⚖️ Perbandingan",
            "🗺️ Analisa Wilayah",
        ],
    )

    st.markdown("---")
    st.markdown("**Filter Dataset Masyarakat**")
    sel_wilayah_k = st.multiselect(
        "Wilayah",
        df_k["Wilayah"].unique(),
        default=list(df_k["Wilayah"].unique()),
    )
    sel_usia = st.multiselect(
        "Kelompok Usia",
        df_k["Kelompok_Usia"].unique(),
        default=list(df_k["Kelompok_Usia"].unique()),
    )

    st.markdown("**Filter Dataset KII**")
    sel_wilayah_kii = st.multiselect(
        "Wilayah KII",
        df_kii["Wilayah"].unique(),
        default=list(df_kii["Wilayah"].unique()),
    )

    st.markdown("---")
    st.caption("📅 Sumber: GECAR Survey 2024–2025")
    st.caption("🌏 Wilayah: Papua")


# Apply filters
df_k_f = df_k[
    (df_k["Wilayah"].isin(sel_wilayah_k)) & 
    (df_k["Kelompok_Usia"].isin(sel_usia))
]
df_kii_f = df_kii[df_kii["Wilayah"].isin(sel_wilayah_kii)]


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
def metric_row(metrics):
    cols = st.columns(len(metrics))
    for col, (label, val, delta) in zip(cols, metrics):
        col.metric(label, val, delta)

def section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def insight(text, kind="info"):
    css = {
        "info": "insight-box", 
        "warning": "insight-box-warning", 
        "success": "insight-box-success"
    }
    box_class = css.get(kind, "insight-box")
    st.markdown(f'<div class="{box_class}">💡 {text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: RINGKASAN EKSEKUTIF
# ══════════════════════════════════════════════
if page == "🏠 Ringkasan Eksekutif":
    st.title("📊 GECAR — Dashboard Analisa Konteks Kemanusiaan")
    st.markdown(
        "> **Program GECAR** mengumpulkan data dari masyarakat (Kelompok Diskusi) dan pemangku kepentingan "
        "pemerintah/NGO (KII) di wilayah Papua untuk memahami dinamika sosial, kebutuhan, dan ketegangan komunitas."
    )

    # KPI Row
    metric_row([
        ("📝 Total Respons Masyarakat", f"{len(df_k):,}", f"{len(df_k_f)} setelah filter"),
        ("🏛️ Total Respons KII", f"{len(df_kii):,}", f"{len(df_kii_f)} setelah filter"),
        ("🌏 Wilayah Terjangkau", "3", "Jayawijaya · Asmat · Sentani"),
        ("👥 Narasumber KII", str(df_kii["Narsum"].nunique()), "GTY · JPY · ACL · Dinas"),
    ])

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        section_header("📊 Distribusi Responden Masyarakat")

        # Wilayah distribution
        wil_k = df_k.groupby("Wilayah").size().reset_index(name="Jumlah")
        fig = px.pie(
            wil_k, 
            values="Jumlah", 
            names="Wilayah",
            color_discrete_sequence=COLORS["wilayah"],
            hole=0.4,
        )
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(height=300, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Gender + usia stacked
        gu = df_k.groupby(["Jenis_Kelamin", "Kelompok_Usia"]).size().reset_index(name="N")
        fig2 = px.bar(
            gu, 
            x="Jenis_Kelamin", 
            y="N", 
            color="Kelompok_Usia",
            color_discrete_map=COLORS["usia"],
            labels={"N": "Jumlah Respons", "Jenis_Kelamin": "Jenis Kelamin", "Kelompok_Usia": "Usia"},
            barmode="stack",
        )
        fig2.update_layout(height=280, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        section_header("🏛️ Distribusi Responden KII")

        # KII by Narsum
        narsum_dist = df_kii.groupby(["Wilayah", "Narsum"]).size().reset_index(name="N")
        fig3 = px.bar(
            narsum_dist, 
            x="Narsum", 
            y="N", 
            color="Wilayah",
            color_discrete_sequence=COLORS["wilayah"],
            labels={"N": "Jumlah Respons", "Narsum": "Narasumber"},
        )
        fig3.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

        # Dimension coverage KII
        dim_kii = df_kii.groupby("Dimensi").size().reset_index(name="N").sort_values("N", ascending=True)
        fig4 = px.bar(
            dim_kii, 
            x="N", 
            y="Dimensi", 
            orientation="h",
            color="N", 
            color_continuous_scale="Blues",
            labels={"N": "Jumlah Respons", "Dimensi": ""},
        )
        fig4.update_layout(height=280, margin=dict(t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    section_header("🗺️ Dimensi Analisa & Pemetaan Pertanyaan")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Dataset Masyarakat (Kelompok)**")
        dim_k_count = df_k.groupby("Dimensi").size().reset_index(name="Respons")
        for _, row in dim_k_count.iterrows():
            st.markdown(
                f'<span class="dim-badge">{row["Dimensi"]}</span> &nbsp; **{row["Respons"]} respons**', 
                unsafe_allow_html=True
            )
            
    with col_b:
        st.markdown("**Dataset KII (Pemerintah/NGO)**")
        dim_kii_count = df_kii.groupby("Dimensi").size().reset_index(name="Respons")
        for _, row in dim_kii_count.iterrows():
            st.markdown(
                f'<span class="dim-badge">{row["Dimensi"]}</span> &nbsp; **{row["Respons"]} respons**', 
                unsafe_allow_html=True
            )

    st.markdown("---")
    section_header("🔑 Temuan Kunci")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        insight(
            "Ketegangan & Konflik adalah dimensi paling banyak direspons dari kedua dataset — "
            "menunjukkan isu keamanan sebagai prioritas utama.", 
            "warning"
        )
    with col2:
        insight(
            "Jayawijaya mendominasi jumlah respons KII (113 dari 167) dengan 3 narasumber berbeda, "
            "mencerminkan kompleksitas wilayah pegunungan.", 
            "info"
        )
    with col3:
        insight(
            "Terdapat keselarasan antara masyarakat dan pemerintah soal Kebutuhan Utama dan "
            "Kelompok Rentan, namun gap pada Harapan & Kontribusi pemuda.", 
            "success"
        )


# ══════════════════════════════════════════════
# PAGE: DATASET MASYARAKAT
# ══════════════════════════════════════════════
elif page == "📋 Dataset Masyarakat":
    st.title("📋 Analisa Dataset Masyarakat (Kelompok Diskusi)")
    st.markdown(f"**Total Respons:** {len(df_k_f):,} &nbsp;|&nbsp; **Wilayah:** {', '.join(sel_wilayah_k)}")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi", "📐 Dimensi", "🔍 Eksplorasi Data", "📈 Cross-Tabulation"])

    # ── Tab 1: Distribusi ──
    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            # Wilayah
            wil = df_k_f.groupby("Wilayah").size().reset_index(name="N")
            fig = px.pie(
                wil, 
                values="N", 
                names="Wilayah", 
                title="Distribusi per Wilayah",
                color_discrete_sequence=COLORS["wilayah"], 
                hole=0.35
            )
            fig.update_layout(height=320, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gender
            gen = df_k_f.groupby("Jenis_Kelamin").size().reset_index(name="N")
            fig = px.bar(
                gen, 
                x="Jenis_Kelamin", 
                y="N", 
                title="Distribusi Jenis Kelamin",
                color="Jenis_Kelamin",
                color_discrete_map=COLORS["gender"],
                text="N"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=320, margin=dict(t=40, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            # Usia
            usia = df_k_f.groupby("Kelompok_Usia").size().reset_index(name="N")
            fig = px.pie(
                usia, 
                values="N", 
                names="Kelompok_Usia", 
                title="Distribusi Kelompok Usia",
                color_discrete_sequence=["#FF6D00", "#6200EA"], 
                hole=0.35
            )
            fig.update_layout(height=320, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Wilayah x Usia x Gender stacked
        st.markdown("---")
        wug = df_k_f.groupby(["Wilayah", "Kelompok_Usia", "Jenis_Kelamin"]).size().reset_index(name="N")
        fig = px.bar(
            wug, 
            x="Wilayah", 
            y="N", 
            color="Jenis_Kelamin",
            facet_col="Kelompok_Usia",
            barmode="stack",
            color_discrete_map=COLORS["gender"],
            title="Distribusi Wilayah × Usia × Jenis Kelamin",
            labels={"N": "Jumlah Respons"}
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        # Respons per pertanyaan (top 15)
        st.markdown("---")
        q_count = df_k_f.groupby("Pertanyaan").size().reset_index(name="N").nlargest(15, "N")
        q_count["Pertanyaan_Short"] = q_count["Pertanyaan"].apply(
            lambda x: x[:60] + "…" if len(x) > 60 else x
        )
        
        fig = px.bar(
            q_count.sort_values("N"), 
            x="N", 
            y="Pertanyaan_Short", 
            orientation="h",
            title="Top 15 Pertanyaan Berdasarkan Jumlah Respons",
            color="N", 
            color_continuous_scale="Blues",
            labels={"N": "Jumlah Respons", "Pertanyaan_Short": ""}
        )
        fig.update_layout(height=480, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Dimensi ──
    with tab2:
        dim_k = df_k_f.groupby("Dimensi").size().reset_index(name="N").sort_values("N", ascending=False)

        col1, col2 = st.columns([1, 1])
        with col1:
            fig = px.bar(
                dim_k.sort_values("N"), 
                x="N", 
                y="Dimensi", 
                orientation="h",
                title="Jumlah Respons per Dimensi",
                color="N", 
                color_continuous_scale="Blues",
                labels={"N": "Jumlah Respons", "Dimensi": ""}
            )
            fig.update_layout(height=420, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(
                dim_k, 
                values="N", 
                names="Dimensi",
                title="Proporsi Dimensi",
                color_discrete_sequence=px.colors.qualitative.Set3, 
                hole=0.3
            )
            fig.update_layout(height=420, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Dimensi per Wilayah
        dw = df_k_f.groupby(["Wilayah", "Dimensi"]).size().reset_index(name="N")
        fig = px.bar(
            dw, 
            x="Wilayah", 
            y="N", 
            color="Dimensi",
            barmode="stack",
            title="Distribusi Dimensi per Wilayah",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Dimensi x Usia
        du = df_k_f.groupby(["Kelompok_Usia", "Dimensi"]).size().reset_index(name="N")
        fig = px.bar(
            du, 
            x="Dimensi", 
            y="N", 
            color="Kelompok_Usia",
            barmode="group",
            title="Dimensi × Kelompok Usia",
            color_discrete_map=COLORS["usia"]
        )
        fig.update_layout(height=380, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        insight(
            "Dimensi 'Ketegangan & Konflik' dan 'Kebutuhan Utama' mendominasi respons masyarakat, "
            "menunjukkan dua prioritas utama komunitas.", 
            "warning"
        )
        insight(
            "Kelompok Anak cenderung lebih banyak berbicara tentang Harapan & Kontribusi, "
            "sementara Dewasa lebih banyak pada Ketegangan & Aktor.", 
            "info"
        )

    # ── Tab 3: Eksplorasi ──
    with tab3:
        st.markdown("### 🔍 Jelajahi Tanggapan per Dimensi")
        sel_dim = st.selectbox("Pilih Dimensi:", sorted(df_k_f["Dimensi"].unique()))
        sel_wil_exp = st.multiselect(
            "Filter Wilayah:", 
            df_k_f["Wilayah"].unique(), 
            default=list(df_k_f["Wilayah"].unique())
        )

        filtered = df_k_f[
            (df_k_f["Dimensi"] == sel_dim) & 
            (df_k_f["Wilayah"].isin(sel_wil_exp))
        ][["Wilayah", "Kelompok_Usia", "Jenis_Kelamin", "Pertanyaan", "Tanggapan"]].reset_index(drop=True)

        st.info(f"Ditemukan **{len(filtered)}** respons untuk dimensi **{sel_dim}**")
        st.dataframe(filtered, use_container_width=True, height=450)

    # ── Tab 4: Cross-Tabulation ──
    with tab4:
        st.markdown("### 📈 Cross-Tabulation Masyarakat")

        ct = pd.crosstab(df_k_f["Dimensi"], df_k_f["Wilayah"])
        fig = px.imshow(
            ct, 
            text_auto=True, 
            aspect="auto",
            title="Heatmap: Dimensi × Wilayah (Jumlah Respons)",
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

        ct2 = pd.crosstab(df_k_f["Dimensi"], df_k_f["Kelompok_Usia"])
        fig2 = px.imshow(
            ct2, 
            text_auto=True, 
            aspect="auto",
            title="Heatmap: Dimensi × Kelompok Usia",
            color_continuous_scale="Oranges"
        )
        fig2.update_layout(height=480)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: DATASET KII
# ══════════════════════════════════════════════
elif page == "🏛️ Dataset KII (Pemerintah)":
    st.title("🏛️ Analisa Dataset KII (Pemerintah & Pemangku Kepentingan)")
    st.markdown(f"**Total Respons:** {len(df_kii_f):,} &nbsp;|&nbsp; **Wilayah:** {', '.join(sel_wilayah_kii)}")

    tab1, tab2, tab3 = st.tabs(["📊 Distribusi", "📐 Dimensi", "🔍 Eksplorasi Data"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Narsum distribution
            n_dist = df_kii_f.groupby("Narsum").size().reset_index(name="N")
            fig = px.bar(
                n_dist, 
                x="Narsum", 
                y="N", 
                title="Distribusi Narasumber",
                color="Narsum", 
                text="N",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            wil_kii = df_kii_f.groupby("Wilayah").size().reset_index(name="N")
            fig = px.pie(
                wil_kii, 
                values="N", 
                names="Wilayah",
                title="Distribusi per Wilayah",
                color_discrete_sequence=COLORS["wilayah"][:2], 
                hole=0.4
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Narsum x Wilayah x Pertanyaan heatmap
        nw = df_kii_f.groupby(["Narsum", "Wilayah"]).size().reset_index(name="N")
        fig = px.bar(
            nw, 
            x="Narsum", 
            y="N", 
            color="Wilayah",
            barmode="stack",
            title="Distribusi Respons: Narasumber × Wilayah",
            color_discrete_sequence=COLORS["wilayah"]
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Top questions
        q_kii = df_kii_f.groupby("Pertanyaan").size().reset_index(name="N").nlargest(12, "N")
        q_kii["Q_Short"] = q_kii["Pertanyaan"].apply(
            lambda x: x[:65] + "…" if len(x) > 65 else x
        )
        
        fig = px.bar(
            q_kii.sort_values("N"), 
            x="N", 
            y="Q_Short", 
            orientation="h",
            title="Top 12 Pertanyaan KII Berdasarkan Respons",
            color="N", 
            color_continuous_scale="Reds",
            labels={"N": "Jumlah Respons", "Q_Short": ""}
        )
        fig.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dim_kii = df_kii_f.groupby("Dimensi").size().reset_index(name="N").sort_values("N", ascending=False)

        col1, col2 = st.columns([1, 1])
        with col1:
            fig = px.bar(
                dim_kii.sort_values("N"), 
                x="N", 
                y="Dimensi", 
                orientation="h",
                title="Jumlah Respons KII per Dimensi",
                color="N", 
                color_continuous_scale="Reds",
                labels={"N": "Jumlah Respons", "Dimensi": ""}
            )
            fig.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.pie(
                dim_kii, 
                values="N", 
                names="Dimensi",
                title="Proporsi Dimensi KII",
                color_discrete_sequence=px.colors.qualitative.Set2, 
                hole=0.3
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        # Dimensi x Narsum
        dn = df_kii_f.groupby(["Narsum", "Dimensi"]).size().reset_index(name="N")
        
        ct_dn = pd.crosstab(
            dn["Dimensi"] if len(dn) > 0 else pd.Series([], name="Dimensi"),
            dn["Narsum"] if len(dn) > 0 else pd.Series([], name="Narsum"),
            values=dn["N"] if len(dn) > 0 else pd.Series([], name="N"),
            aggfunc="sum"
        ).fillna(0)
        
        fig = px.imshow(
            ct_dn, 
            text_auto=True, 
            aspect="auto",
            title="Heatmap: Dimensi × Narasumber KII",
            color_continuous_scale="Reds"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        insight(
            "Dimensi 'Dampak & Operasional LSM' adalah unik untuk KII — mencerminkan "
            "perspektif institusional yang tidak dimiliki masyarakat umum.", 
            "info"
        )
        insight(
            "Narasumber GTY & JPY (Jayawijaya) paling banyak merespons isu Ketegangan & Konflik, "
            "konsisten dengan dinamika keamanan di Papua Pegunungan.", 
            "warning"
        )

    with tab3:
        st.markdown("### 🔍 Jelajahi Tanggapan KII per Dimensi")
        sel_dim_kii = st.selectbox("Pilih Dimensi KII:", sorted(df_kii_f["Dimensi"].unique()))
        sel_narsum = st.multiselect(
            "Filter Narasumber:", 
            df_kii_f["Narsum"].unique(), 
            default=list(df_kii_f["Narsum"].unique())
        )

        filtered_kii = df_kii_f[
            (df_kii_f["Dimensi"] == sel_dim_kii) & 
            (df_kii_f["Narsum"].isin(sel_narsum))
        ][["Wilayah", "Narsum", "Pertanyaan", "Tanggapan"]].reset_index(drop=True)

        st.info(f"Ditemukan **{len(filtered_kii)}** respons untuk dimensi **{sel_dim_kii}**")
        st.dataframe(filtered_kii, use_container_width=True, height=450)


# ══════════════════════════════════════════════
# PAGE: ANALISA DIMENSI
# ══════════════════════════════════════════════
elif page == "🔍 Analisa Dimensi":
    st.title("🔍 Analisa Mendalam per Dimensi")
    st.markdown("Pilih dimensi untuk melihat distribusi detail dan perbandingan antara kedua dataset.")

    sel_dim_all = st.selectbox(
        "🎯 Pilih Dimensi untuk Dianalisa:",
        SHARED_DIMS,
    )

    dk_dim = df_k_f[df_k_f["Dimensi"] == sel_dim_all]
    dkii_dim = df_kii_f[df_kii_f["Dimensi"] == sel_dim_all]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Respons Masyarakat", len(dk_dim))
    col2.metric("Respons KII", len(dkii_dim))
    col3.metric("Wilayah Masyarakat", dk_dim["Wilayah"].nunique() if len(dk_dim) > 0 else 0)
    col4.metric("Narasumber KII", dkii_dim["Narsum"].nunique() if len(dkii_dim) > 0 else 0)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### 👥 Masyarakat — {sel_dim_all}")
        if len(dk_dim) > 0:
            # Wilayah breakdown
            w_dk = dk_dim.groupby("Wilayah").size().reset_index(name="N")
            fig = px.bar(
                w_dk, 
                x="Wilayah", 
                y="N", 
                color="Wilayah",
                color_discrete_sequence=COLORS["wilayah"],
                text="N",
                labels={"N": "Jumlah Respons"}
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Usia breakdown
            u_dk = dk_dim.groupby(["Kelompok_Usia", "Jenis_Kelamin"]).size().reset_index(name="N")
            fig2 = px.bar(
                u_dk, 
                x="Kelompok_Usia", 
                y="N", 
                color="Jenis_Kelamin",
                color_discrete_map=COLORS["gender"],
                barmode="group",
                text="N",
                labels={"N": "Jumlah Respons", "Kelompok_Usia": "Kelompok Usia"}
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Tidak ada data untuk dimensi dan filter ini.")

    with col2:
        st.markdown(f"#### 🏛️ KII (Pemerintah) — {sel_dim_all}")
        if len(dkii_dim) > 0:
            # Narsum breakdown
            n_dkii = dkii_dim.groupby(["Narsum", "Wilayah"]).size().reset_index(name="N")
            fig = px.bar(
                n_dkii, 
                x="Narsum", 
                y="N", 
                color="Wilayah",
                color_discrete_sequence=COLORS["wilayah"],
                text="N",
                barmode="stack",
                labels={"N": "Jumlah Respons"}
            )
            fig.update_traces(textposition="inside")
            fig.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Pertanyaan breakdown
            q_dkii = dkii_dim.groupby("Pertanyaan").size().reset_index(name="N")
            q_dkii["Q_Short"] = q_dkii["Pertanyaan"].apply(
                lambda x: x[:50] + "…" if len(x) > 50 else x
            )
            
            fig2 = px.pie(
                q_dkii, 
                values="N", 
                names="Q_Short",
                color_discrete_sequence=px.colors.qualitative.Set2, 
                hole=0.3
            )
            fig2.update_layout(height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Tidak ada data KII untuk dimensi ini.")

    st.markdown("---")
    st.markdown("### 📝 Sample Tanggapan")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Suara Masyarakat:**")
        if len(dk_dim) > 0:
            samples = dk_dim[["Wilayah", "Kelompok_Usia", "Tanggapan"]].head(5)
            for _, r in samples.iterrows():
                with st.expander(f"📍 {r['Wilayah']} — {r['Kelompok_Usia']}"):
                    st.write(r["Tanggapan"])
                    
    with col2:
        st.markdown("**Perspektif KII:**")
        if len(dkii_dim) > 0:
            samples_kii = dkii_dim[["Wilayah", "Narsum", "Tanggapan"]].head(5)
            for _, r in samples_kii.iterrows():
                with st.expander(f"🏛️ {r['Wilayah']} — {r['Narsum']}"):
                    st.write(r["Tanggapan"])


# ══════════════════════════════════════════════
# PAGE: PERBANDINGAN
# ══════════════════════════════════════════════
elif page == "⚖️ Perbandingan":
    st.title("⚖️ Perbandingan: Masyarakat vs. Pemerintah/KII")
    st.markdown("Analisa kesenjangan dan keselarasan perspektif antara masyarakat dan pemangku kepentingan pemerintah.")

    # Radar chart — Dimensi coverage comparison
    shared_k = df_k_f[df_k_f["Dimensi"].isin(SHARED_DIMS)].groupby("Dimensi").size()
    shared_kii = df_kii_f[df_kii_f["Dimensi"].isin(SHARED_DIMS)].groupby("Dimensi").size()

    all_dims = SHARED_DIMS
    k_vals = [shared_k.get(d, 0) for d in all_dims]
    kii_vals = [shared_kii.get(d, 0) for d in all_dims]

    # Normalize to percentage
    k_pct = [v / sum(k_vals) * 100 if sum(k_vals) > 0 else 0 for v in k_vals]
    kii_pct = [v / sum(kii_vals) * 100 if sum(kii_vals) > 0 else 0 for v in kii_vals]

    dim_labels = [d.split(" ", 1)[1] for d in all_dims]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=k_pct + [k_pct[0]],
        theta=dim_labels + [dim_labels[0]],
        fill="toself",
        name="Masyarakat",
        line_color=COLORS["kelompok"],
        fillcolor="rgba(26,115,232,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=kii_pct + [kii_pct[0]],
        theta=dim_labels + [dim_labels[0]],
        fill="toself",
        name="KII (Pemerintah)",
        line_color=COLORS["kii"],
        fillcolor="rgba(234,67,53,0.15)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(max(k_pct), max(kii_pct)) + 5])),
        title="Radar: Proporsi Perhatian per Dimensi (% dari respons)",
        legend=dict(orientation="h", y=-0.1),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Side-by-side bar comparison
    comp_df = pd.DataFrame({
        "Dimensi": dim_labels,
        "Masyarakat (%)": k_pct,
        "KII (%)": kii_pct,
    })
    comp_long = comp_df.melt(id_vars="Dimensi", var_name="Kelompok", value_name="Persen")
    
    fig2 = px.bar(
        comp_long, 
        x="Dimensi", 
        y="Persen", 
        color="Kelompok",
        barmode="group",
        color_discrete_map={"Masyarakat (%)": COLORS["kelompok"], "KII (%)": COLORS["kii"]},
        title="Perbandingan Distribusi Dimensi: Masyarakat vs KII (%)",
        labels={"Persen": "Proporsi (%)", "Dimensi": ""},
    )
    fig2.update_layout(height=420, xaxis_tickangle=-25)
    st.plotly_chart(fig2, use_container_width=True)

    # GAP Analysis
    st.markdown("---")
    section_header("📊 Analisa Gap: Perbedaan Prioritas")
    
    comp_df["Gap (pp)"] = comp_df["Masyarakat (%)"] - comp_df["KII (%)"]
    comp_df["Arah"] = comp_df["Gap (pp)"].apply(
        lambda x: "Lebih Tinggi di Masyarakat" if x > 0 else "Lebih Tinggi di KII"
    )
    
    fig3 = px.bar(
        comp_df.sort_values("Gap (pp)"), 
        x="Gap (pp)", 
        y="Dimensi",
        orientation="h",
        color="Arah",
        color_discrete_map={
            "Lebih Tinggi di Masyarakat": COLORS["kelompok"],
            "Lebih Tinggi di KII": COLORS["kii"],
        },
        title="Gap Proporsi: Masyarakat − KII (poin persentase)",
        text="Gap (pp)",
    )
    fig3.update_traces(texttemplate="%{text:.1f}pp", textposition="outside")
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        insight(
            "Masyarakat lebih banyak membahas Keamanan & Harapan — mencerminkan "
            "kebutuhan langsung dan aspirasi hidup sehari-hari.", 
            "info"
        )
    with col2:
        insight(
            "KII lebih fokus pada Dampak Operasional LSM dan Rekomendasi — mencerminkan "
            "perspektif institusional dan program kerja.", 
            "warning"
        )

    # Tabel ringkasan
    st.markdown("---")
    st.markdown("### 📋 Tabel Perbandingan Lengkap")
    st.dataframe(comp_df.set_index("Dimensi").round(1), use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: ANALISA WILAYAH
# ══════════════════════════════════════════════
elif page == "🗺️ Analisa Wilayah":
    st.title("🗺️ Analisa per Wilayah")

    wilayah_list = sorted(df_k["Wilayah"].unique())
    sel_wil_page = st.radio("Pilih Wilayah:", wilayah_list, horizontal=True)

    dk_wil = df_k[df_k["Wilayah"] == sel_wil_page]
    dkii_wil = df_kii[df_kii["Wilayah"] == sel_wil_page] if sel_wil_page in df_kii["Wilayah"].unique() else pd.DataFrame()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Respons Masyarakat", len(dk_wil))
    col2.metric("Respons KII", len(dkii_wil) if len(dkii_wil) > 0 else "—")
    col3.metric("Kelompok Usia", dk_wil["Kelompok_Usia"].nunique())
    col4.metric("Dimensi Tercakup", dk_wil["Dimensi"].nunique())

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### 👥 Profil Masyarakat — {sel_wil_page}")

        # Usia & Gender
        ug = dk_wil.groupby(["Kelompok_Usia", "Jenis_Kelamin"]).size().reset_index(name="N")
        fig = px.bar(
            ug, 
            x="Kelompok_Usia", 
            y="N", 
            color="Jenis_Kelamin",
            barmode="group", 
            color_discrete_map=COLORS["gender"],
            text="N", 
            labels={"N": "Jumlah Respons", "Kelompok_Usia": "Usia"}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Dimensi coverage
        d_wil = dk_wil.groupby("Dimensi").size().reset_index(name="N").sort_values("N", ascending=True)
        fig2 = px.bar(
            d_wil, 
            x="N", 
            y="Dimensi", 
            orientation="h",
            color="N", 
            color_continuous_scale="Blues",
            labels={"N": "Respons", "Dimensi": ""}
        )
        fig2.update_layout(height=380, coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown(f"#### 🏛️ Profil KII — {sel_wil_page}")
        if len(dkii_wil) > 0:
            # Narsum
            n_wil = dkii_wil.groupby("Narsum").size().reset_index(name="N")
            fig = px.pie(
                n_wil, 
                values="N", 
                names="Narsum",
                hole=0.4, 
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Dimensi KII
            dk_wil2 = dkii_wil.groupby("Dimensi").size().reset_index(name="N").sort_values("N", ascending=True)
            fig2 = px.bar(
                dk_wil2, 
                x="N", 
                y="Dimensi", 
                orientation="h",
                color="N", 
                color_continuous_scale="Reds",
                labels={"N": "Respons", "Dimensi": ""}
            )
            fig2.update_layout(height=380, coloraxis_showscale=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(f"Tidak ada data KII untuk wilayah **{sel_wil_page}**.")

    # Temuan khusus per wilayah
    st.markdown("---")
    section_header(f"💡 Temuan Khusus: {sel_wil_page}")
    
    wilayah_insights = {
        "Jayawijaya": [
            (
                "warning", 
                "Jayawijaya memiliki intensitas konflik tertinggi — dengan 3 narasumber KII yang "
                "konsisten menyebut ketegangan keamanan, penambangan rakyat di Yahukimo, dan ancaman OPM."
            ),
            (
                "info", 
                "Masyarakat Jayawijaya lebih banyak membahas Kebutuhan Utama (sandang-pangan-papan) "
                "akibat inflasi tertinggi nasional (5,65% - BPS 2024)."
            ),
            (
                "success", 
                "Terdapat inisiatif perdamaian melalui lembaga adat dan gereja sebagai pemersatu "
                "masyarakat di tengah dinamika konflik."
            ),
        ],
        "Asmat, Papua Selatan": [
            (
                "info", 
                "Asmat hanya terwakili dalam dataset Masyarakat — menunjukkan keterbatasan jangkauan "
                "KII di wilayah selatan Papua."
            ),
            (
                "warning", 
                "Isu dominan: akses layanan dasar (pendidikan, kesehatan) dan isolasi geografis yang "
                "memperparah ketergantungan masyarakat."
            ),
            (
                "success", 
                "Responden dewasa Asmat lebih banyak daripada anak — mencerminkan profil diskusi "
                "kelompok yang lebih matang."
            ),
        ],
        "Sentani": [
            (
                "success", 
                "Sentani relatif lebih stabil — dengan kehadiran Dinas Sosial dan Dinas Pendidikan "
                "sebagai narasumber KII yang mencerminkan aktifnya layanan pemerintah."
            ),
            (
                "info", 
                "Fokus KII Sentani lebih pada program pemberdayaan dan peace-building — berbeda dengan "
                "Jayawijaya yang lebih reaktif terhadap konflik."
            ),
            (
                "warning", 
                "Masyarakat Sentani tetap mencatat ketakutan terkait spillover konflik dari "
                "wilayah pegunungan."
            ),
        ],
    }
    
    for kind, text in wilayah_insights.get(sel_wil_page, [("info", "Tidak ada temuan khusus.")]):
        insight(text, kind)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:12px;'>"
    "📊 GECAR Dashboard &nbsp;|&nbsp; Analisa Konteks Kemanusiaan Papua &nbsp;|&nbsp; "
    "Data: Kelompok Diskusi & KII 2024–2025 &nbsp;|&nbsp; Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True,
)
