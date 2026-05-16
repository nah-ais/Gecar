"""
=============================================================================
GECAR NLP ANALYSIS ENGINE
Senior Data Analyst + Expert NLP Python Pipeline
=============================================================================
Modul utama untuk:
  - Standardisasi & Preprocessing
  - TF-IDF Extraction (per wilayah, per pilar)
  - BERTopic Semantic Clustering (dengan fallback NMF/LDA)
  - Gap Analysis (Masyarakat vs Pemerintah)
=============================================================================
"""

import re
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# SASTRAWI STOPWORDS
# ──────────────────────────────────────────────────────────────────────────────
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    _sastrawi_factory = StopWordRemoverFactory()
    _sastrawi_stopwords = set(_sastrawi_factory.get_stop_words())
    SASTRAWI_AVAILABLE = True
except ImportError:
    _sastrawi_stopwords = set()
    SASTRAWI_AVAILABLE = False
    print("[WARN] PySastrawi tidak tersedia, menggunakan stopwords manual.")

# Custom stopwords tambahan (bahasa informal Papua + kata umum tak bermakna)
CUSTOM_STOPWORDS = {
    "yg", "nya", "bikin", "pas", "udah", "udh", "kayak", "kaya", "aja",
    "juga", "sudah", "sudh", "bisa", "saja", "ini", "itu", "ada", "dan",
    "yang", "ke", "di", "dari", "untuk", "dengan", "pada", "oleh", "atau",
    "dalam", "akan", "dapat", "tidak", "belum", "sudah", "jadi", "saat",
    "lebih", "sangat", "banyak", "kali", "lagi", "sering", "masih",
    "terus", "punya", "perlu", "kami", "kita", "mereka", "saya", "anda",
    "dia", "ia", "kamu", "mu", "ku", "lah", "kah", "pun", "tapi",
    "namun", "karena", "sebab", "jika", "kalau", "ketika", "sehingga",
    "antara", "seperti", "agar", "supaya", "selain", "bahwa", "hanya",
    "sudah", "belum", "pernah", "selalu", "kadang", "sering", "jarang",
    "beberapa", "semua", "setiap", "masing", "masing-masing", "hal", "cara",
    "ada", "adalah", "merupakan", "yaitu", "yakni", "misalnya", "contoh",
    "tersebut", "terkait", "tentang", "mengenai", "berupa", "berupa",
    "wvi", "lsm", "pbb", "ngo",  # acronyms contextual - keep in some analyses
}

ALL_STOPWORDS = _sastrawi_stopwords.union(CUSTOM_STOPWORDS)

# ──────────────────────────────────────────────────────────────────────────────
# LANGKAH 1A: MAPPING PILAR SEKTORAL
# ──────────────────────────────────────────────────────────────────────────────

PILAR_KEYWORDS = {
    "Keamanan & Konflik": [
        "aman", "aman", "takut", "ketegangan", "konflik", "militer", "miras",
        "keamanan", "kekerasan", "kriminal", "kejahatan", "senjata", "bersenjata",
        "ancaman", "bahaya", "perang", "kelompok bersenjata", "penyanderaan",
        "ketakutan", "rasa aman", "merasa aman", "membuatmu aman",
        "membuat anda merasa", "membuatmu merasa", "stabilitas", "situasi keamanan",
        "situasi politik", "politik", "pemilu", "pemilihan", "ketegangan",
        "tegangan", "rawan", "rawan konflik", "egianus", "kogoya", "kpb",
        "kelompok", "bersenjata", "sipil", "milisi",
    ],
    "Ekonomi & Mata Pencaharian": [
        "kebutuhan utama", "kebutuhan", "kerja", "pekerjaan", "bansos",
        "bantuan sosial", "uang", "pangan", "bantuan", "ekonomi", "inflasi",
        "pendapatan", "penghasilan", "mata pencaharian", "usaha", "dagang",
        "pasar", "harga", "sembako", "beras", "logistik", "udara", "jalur",
        "sandang", "papan", "kemiskinan", "miskin", "kaya", "kekurangan",
        "cukup", "tidak cukup", "biaya", "modal", "investasi", "pemberdayaan",
        "ekonomi", "pertanian", "ternak", "kebun", "panen", "hasil",
        "kebutuhan utama", "kampung", "desa",
    ],
    "Sosial & Kesejahteraan": [
        "sehari-hari", "aktivitas", "rentan", "anak", "remaja", "tokoh",
        "masyarakat", "kehidupan", "kesehatan", "pendidikan", "sekolah",
        "belajar", "keluarga", "perempuan", "ibu", "lanjut usia", "lansia",
        "difabel", "disabilitas", "kelompok rentan", "berisiko", "sosial",
        "kesejahteraan", "komunitas", "kampung", "warga", "penduduk",
        "pemimpin", "tokoh adat", "gereja", "agama", "ibadah", "tradisi",
        "budaya", "adat", "memecah belah", "menyatukan", "persatuan",
        "perdamaian", "rukun", "harmonis", "gotong royong", "kebersamaan",
        "aktivitas harian", "harian", "keseharian",
    ],
    "Proyeksi Masa Depan": [
        "6 bulan", "enam bulan", "kedepan", "ke depan", "waktu dekat",
        "skenario", "masa depan", "harapan", "prediksi", "perkiraan",
        "proyeksi", "akan terjadi", "mungkin terjadi", "dalam waktu",
        "mendatang", "antisipasi", "rencana", "prospek", "situasi mendatang",
        "terjadi", "apa yang akan", "apa yang mungkin", "beberapa bulan",
    ],
    "Rekomendasi & Intervensi": [
        "perlu dilakukan", "kontribusi", "pemuda", "lsm", "wvi",
        "masukan", "rekomendasi", "saran", "program", "intervensi",
        "solusi", "penanganan", "penanggulangan", "kegiatan", "pembinaan",
        "perdamaian", "promosi", "advokasi", "pendampingan", "capacity",
        "kapasitas", "pelatihan", "workshop", "mediasi", "dialog",
        "siapa yang", "dilakukan oleh", "bisa dilakukan", "apa yang perlu",
        "berkontribusi", "mengubah", "perubahan", "pesan", "komunikasi",
        "publik", "konsekuensi", "dampak", "operasional",
    ],
}


def map_pilar_sektoral(pertanyaan_text: str) -> str:
    """
    Mapping teks pertanyaan ke salah satu dari 5 Pilar Sektoral.
    Menggunakan keyword matching dengan scoring - pilar dengan skor tertinggi menang.
    Jika tidak ada yang match, dikategorikan ke pilar paling relevan secara konteks.
    """
    if not isinstance(pertanyaan_text, str) or len(pertanyaan_text.strip()) == 0:
        return "Sosial & Kesejahteraan"  # default fallback

    text_lower = pertanyaan_text.lower()
    scores = {}

    for pilar, keywords in PILAR_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw.lower() in text_lower:
                # Keyword multi-kata dapat skor lebih tinggi
                score += len(kw.split())
        scores[pilar] = score

    best_pilar = max(scores, key=scores.get)
    best_score = scores[best_pilar]

    if best_score == 0:
        # Fallback heuristik berdasarkan kata kunci minimal
        if any(w in text_lower for w in ["aman", "takut", "konflik", "politik"]):
            return "Keamanan & Konflik"
        elif any(w in text_lower for w in ["kebutuhan", "bantu", "ekonomi"]):
            return "Ekonomi & Mata Pencaharian"
        elif any(w in text_lower for w in ["6 bulan", "kedepan", "skenario", "harapan"]):
            return "Proyeksi Masa Depan"
        elif any(w in text_lower for w in ["dilakukan", "kontribusi", "wvi", "lsm"]):
            return "Rekomendasi & Intervensi"
        else:
            return "Sosial & Kesejahteraan"

    return best_pilar


# ──────────────────────────────────────────────────────────────────────────────
# LANGKAH 1B: TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Preprocessing teks Bahasa Indonesia:
      1. Lowercase
      2. Hapus karakter non-alfanumerik (kecuali spasi)
      3. Hapus angka standalone
      4. Hapus stopwords (Sastrawi + custom)
      ⚠️ TIDAK melakukan stemming — untuk menjaga struktur kata berimbuhan
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Hapus karakter khusus, tanda baca, simbol
    text = re.sub(r"[^\w\s]", " ", text)       # hapus tanda baca
    text = re.sub(r"[_\*\#\@\!\?\=\+]", " ", text)  # karakter khusus
    text = re.sub(r"\d+", " ", text)            # angka

    # 3. Normalisasi spasi
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Hapus stopwords
    if remove_stopwords and len(ALL_STOPWORDS) > 0:
        words = text.split()
        words = [w for w in words if w not in ALL_STOPWORDS and len(w) > 2]
        text = " ".join(words)

    return text


def preprocess_dataframe(df: pd.DataFrame, source_label: str = "") -> pd.DataFrame:
    """
    Preprocessing menyeluruh pada DataFrame:
      - Tambahkan kolom Pilar_Sektoral
      - Tambahkan kolom Tanggapan_Clean
      - Eksklusikan baris dengan tanggapan kosong
      - Eksklusikan subgrup demografi kosong di wilayah tertentu
    """
    df = df.copy()

    # Mapping Pilar Sektoral
    df["Pilar_Sektoral"] = df["Pertanyaan"].apply(map_pilar_sektoral)

    # Preprocessing teks tanggapan
    df["Tanggapan_Clean"] = df["Tanggapan"].apply(preprocess_text)

    # Eksklusikan baris dengan tanggapan kosong setelah preprocessing
    df = df[df["Tanggapan_Clean"].str.len() > 10].copy()
    df = df[df["Tanggapan"].notna()].copy()
    df = df[df["Tanggapan"].str.strip() != ""].copy()

    # Tambahkan label sumber
    df["Sumber"] = source_label

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# LANGKAH 2A: TF-IDF EXTRACTION PER WILAYAH & PILAR
# ──────────────────────────────────────────────────────────────────────────────

def extract_tfidf_phrases(
    df: pd.DataFrame,
    wilayah: str = None,
    pilar: str = None,
    top_n: int = 15,
    ngram_range: tuple = (2, 3),
    min_df: int = 1,
) -> pd.DataFrame:
    """
    TF-IDF Vectorizer dengan N-Gram (2,3) per kombinasi Wilayah & Pilar.
    Mengembalikan DataFrame berisi frasa teratas beserta skor TF-IDF agregat.
    """
    subset = df.copy()

    if wilayah and wilayah != "Semua":
        subset = subset[subset["Wilayah"] == wilayah]
    if pilar and pilar != "Semua":
        subset = subset[subset["Pilar_Sektoral"] == pilar]

    texts = subset["Tanggapan_Clean"].dropna().tolist()
    texts = [t for t in texts if len(t.strip()) > 10]

    if len(texts) < 2:
        return pd.DataFrame(columns=["Frasa", "Skor_TFIDF", "Frekuensi"])

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=500,
            min_df=min_df,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Agregasi: rata-rata skor TF-IDF per frasa di semua dokumen
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        freq_scores = np.asarray((tfidf_matrix > 0).sum(axis=0)).flatten()

        result_df = pd.DataFrame({
            "Frasa": feature_names,
            "Skor_TFIDF": mean_scores,
            "Frekuensi": freq_scores,
        })

        result_df = result_df.sort_values("Skor_TFIDF", ascending=False)
        return result_df.head(top_n).reset_index(drop=True)

    except Exception as e:
        print(f"[WARN] TF-IDF error: {e}")
        return pd.DataFrame(columns=["Frasa", "Skor_TFIDF", "Frekuensi"])


def run_tfidf_all_segments(
    df_masy: pd.DataFrame,
    df_pem: pd.DataFrame,
    top_n: int = 10,
) -> dict:
    """
    Jalankan TF-IDF untuk semua kombinasi Wilayah x Pilar x Sumber.
    Mengembalikan dict berstruktur: results[wilayah][pilar][sumber] = DataFrame frasa
    """
    results = defaultdict(lambda: defaultdict(dict))
    wilayah_list = sorted(
        set(df_masy["Wilayah"].unique().tolist() + df_pem["Wilayah"].unique().tolist())
    )
    pilar_list = list(PILAR_KEYWORDS.keys())

    for wil in wilayah_list:
        for pil in pilar_list:
            results[wil][pil]["Masyarakat"] = extract_tfidf_phrases(
                df_masy, wilayah=wil, pilar=pil, top_n=top_n
            )
            results[wil][pil]["Pemerintah"] = extract_tfidf_phrases(
                df_pem, wilayah=wil, pilar=pil, top_n=top_n
            )

    return dict(results)


# ──────────────────────────────────────────────────────────────────────────────
# LANGKAH 2B: TOPIC MODELING (NMF sebagai backbone, BERTopic opsional)
# ──────────────────────────────────────────────────────────────────────────────

def run_nmf_topics(
    df: pd.DataFrame,
    wilayah: str = None,
    pilar: str = None,
    n_topics: int = 5,
    top_words: int = 8,
) -> list:
    """
    NMF Topic Modeling — digunakan sebagai backbone yang selalu tersedia.
    Mengembalikan list of dict: [{topic_id, label_otomatis, kata_kunci, dokumen_sampel}]
    """
    subset = df.copy()
    if wilayah and wilayah != "Semua":
        subset = subset[subset["Wilayah"] == wilayah]
    if pilar and pilar != "Semua":
        subset = subset[subset["Pilar_Sektoral"] == pilar]

    texts = subset["Tanggapan_Clean"].dropna().tolist()
    texts = [t for t in texts if len(t.strip()) > 10]

    if len(texts) < n_topics:
        n_topics = max(2, len(texts) // 2)

    if len(texts) < 2:
        return []

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=300,
            min_df=1,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
        W = nmf.fit_transform(tfidf_matrix)  # doc-topic matrix
        H = nmf.components_               # topic-word matrix

        topics = []
        for i, topic_vec in enumerate(H):
            top_idx = topic_vec.argsort()[::-1][:top_words]
            keywords = [feature_names[idx] for idx in top_idx]

            # Ambil dokumen sampel yang paling mewakili topik ini
            doc_scores = W[:, i]
            top_docs_idx = doc_scores.argsort()[::-1][:2]
            sample_docs = []
            for idx in top_docs_idx:
                if idx < len(subset):
                    original = subset.iloc[idx]["Tanggapan"]
                    if isinstance(original, str) and len(original) > 20:
                        sample_docs.append(original[:200] + "..." if len(original) > 200 else original)

            topics.append({
                "topic_id": i + 1,
                "label_otomatis": f"Subtema {i+1}: {' | '.join(keywords[:3])}",
                "kata_kunci": keywords,
                "dokumen_sampel": sample_docs,
                "n_dokumen": int((doc_scores > 0.1).sum()),
            })

        return topics

    except Exception as e:
        print(f"[WARN] NMF error: {e}")
        return []


def run_bertopic_topics(
    df: pd.DataFrame,
    wilayah: str = None,
    pilar: str = None,
) -> tuple:
    """
    BERTopic dengan IndoBERT embeddings.
    Jika BERTopic/IndoBERT tidak tersedia, fallback ke NMF.
    Returns: (topics_list, method_used)
    """
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        subset = df.copy()
        if wilayah and wilayah != "Semua":
            subset = subset[subset["Wilayah"] == wilayah]
        if pilar and pilar != "Semua":
            subset = subset[subset["Pilar_Sektoral"] == pilar]

        # Gunakan teks asli (bukan cleaned) untuk embedding semantik
        texts = subset["Tanggapan"].dropna().tolist()
        texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 20]

        if len(texts) < 5:
            return run_nmf_topics(df, wilayah, pilar), "NMF (data terlalu sedikit)"

        # Load IndoBERT — model terbaik untuk Bahasa Indonesia
        print("[INFO] Loading IndoBERT embedding model...")
        embedding_model = SentenceTransformer("indobenchmark/indobert-base-p1")

        topic_model = BERTopic(
            embedding_model=embedding_model,
            language="multilingual",
            min_topic_size=2,
            nr_topics="auto",
            verbose=False,
        )

        topics_raw, probs = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()

        topics = []
        for _, row in topic_info.iterrows():
            if row["Topic"] == -1:  # outlier cluster
                continue
            topic_id = row["Topic"]
            topic_words_raw = topic_model.get_topic(topic_id)
            keywords = [w for w, _ in topic_words_raw[:8]] if topic_words_raw else []

            # Temukan dokumen sampel untuk topik ini
            doc_indices = [i for i, t in enumerate(topics_raw) if t == topic_id][:2]
            sample_docs = [texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i]
                           for i in doc_indices if i < len(texts)]

            topics.append({
                "topic_id": topic_id,
                "label_otomatis": f"Klaster {topic_id}: {' | '.join(keywords[:3])}",
                "kata_kunci": keywords,
                "dokumen_sampel": sample_docs,
                "n_dokumen": row.get("Count", len(doc_indices)),
            })

        return topics, "BERTopic + IndoBERT"

    except ImportError:
        print("[INFO] BERTopic/sentence-transformers tidak terinstall, menggunakan NMF.")
        return run_nmf_topics(df, wilayah, pilar), "NMF (BERTopic tidak tersedia)"

    except Exception as e:
        print(f"[WARN] BERTopic error: {e} — fallback ke NMF.")
        return run_nmf_topics(df, wilayah, pilar), f"NMF (fallback: {str(e)[:50]})"


# ──────────────────────────────────────────────────────────────────────────────
# LANGKAH 3: GAP ANALYSIS — KOMPARASI HORIZONTAL
# ──────────────────────────────────────────────────────────────────────────────

def compute_gap_score(frasa_masy: list, frasa_pem: list) -> float:
    """
    Hitung Gap Score antara frasa Masyarakat dan Pemerintah.
    Score 0 = identik (tidak ada gap), Score 1 = sama sekali berbeda.
    Menggunakan Jaccard Distance pada set frasa N-Gram.
    """
    set_m = set(" ".join(f.split()[:2]) for f in frasa_masy)  # bigram dari frasa
    set_p = set(" ".join(f.split()[:2]) for f in frasa_pem)

    if not set_m and not set_p:
        return 0.0
    if not set_m or not set_p:
        return 1.0

    intersection = len(set_m & set_p)
    union = len(set_m | set_p)
    jaccard_sim = intersection / union
    return round(1 - jaccard_sim, 3)  # gap = 1 - similarity


def run_gap_analysis(
    tfidf_results: dict,
    wilayah_list: list,
    pilar_list: list,
) -> pd.DataFrame:
    """
    Fungsi komparasi horizontal — menghasilkan matriks Gap Score.
    Kolom: Wilayah, Pilar_Sektoral, Gap_Score, Frasa_Masyarakat, Frasa_Pemerintah,
           Frasa_Overlap, Frasa_Eksklusif_Masy, Frasa_Eksklusif_Pem
    """
    rows = []

    for wil in wilayah_list:
        for pil in pilar_list:
            masy_data = tfidf_results.get(wil, {}).get(pil, {}).get("Masyarakat", pd.DataFrame())
            pem_data = tfidf_results.get(wil, {}).get(pil, {}).get("Pemerintah", pd.DataFrame())

            frasa_masy = masy_data["Frasa"].tolist() if not masy_data.empty else []
            frasa_pem = pem_data["Frasa"].tolist() if not pem_data.empty else []

            gap = compute_gap_score(frasa_masy, frasa_pem)

            # Frasa yang overlap vs eksklusif
            set_m = set(frasa_masy)
            set_p = set(frasa_pem)
            overlap = sorted(set_m & set_p)
            eksklusif_m = sorted(set_m - set_p)
            eksklusif_p = sorted(set_p - set_m)

            rows.append({
                "Wilayah": wil,
                "Pilar_Sektoral": pil,
                "Gap_Score": gap,
                "N_Frasa_Masy": len(frasa_masy),
                "N_Frasa_Pem": len(frasa_pem),
                "N_Overlap": len(overlap),
                "Frasa_Top_Masy": " | ".join(frasa_masy[:5]),
                "Frasa_Top_Pem": " | ".join(frasa_pem[:5]),
                "Frasa_Overlap": " | ".join(overlap[:5]),
                "Frasa_Eksklusif_Masy": " | ".join(eksklusif_m[:5]),
                "Frasa_Eksklusif_Pem": " | ".join(eksklusif_p[:5]),
                "Interpretasi": _interpret_gap(gap, pil),
            })

    return pd.DataFrame(rows)


def _interpret_gap(score: float, pilar: str) -> str:
    """Interpretasi kualitatif dari Gap Score."""
    if score >= 0.90:
        return f"⚠️ KRITIS: Mismatch total pada {pilar} — prioritas berbeda jauh"
    elif score >= 0.70:
        return f"🔴 TINGGI: Kesenjangan signifikan — sedikit titik temu"
    elif score >= 0.50:
        return f"🟡 SEDANG: Ada beberapa kesamaan tapi masih banyak gap"
    elif score >= 0.25:
        return f"🟢 RENDAH: Relatif selaras — ada komunikasi yang baik"
    else:
        return f"✅ MINIMAL: Prioritas sangat selaras antara kedua pihak"


def get_benang_merah(
    df: pd.DataFrame,
    wilayah: str = None,
) -> dict:
    """
    Identifikasi 'Benang Merah' — tema berulang lintas Pilar di suatu wilayah.
    Returns dict: {tema: frekuensi/skor}
    """
    subset = df.copy()
    if wilayah and wilayah != "Semua":
        subset = subset[subset["Wilayah"] == wilayah]

    all_text = " ".join(subset["Tanggapan_Clean"].dropna().tolist())

    if len(all_text) < 50:
        return {}

    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=100, min_df=2)
        matrix = vec.fit_transform(subset["Tanggapan_Clean"].dropna().tolist())
        scores = np.asarray(matrix.mean(axis=0)).flatten()
        names = vec.get_feature_names_out()
        result = dict(zip(names, scores))
        sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:20])
        return sorted_result
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE UTAMA
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    path_kelompok: str,
    path_kii: str,
    top_n_tfidf: int = 12,
) -> dict:
    """
    Pipeline lengkap GECAR NLP Analysis.
    Returns dict berisi semua hasil analisis untuk dikonsumsi Streamlit.
    """
    print("=" * 60)
    print("GECAR NLP ANALYSIS PIPELINE")
    print("=" * 60)

    # ── Load data ──
    print("[1/5] Loading data...")
    df_kelompok_raw = pd.read_csv(path_kelompok)
    df_kii_raw = pd.read_csv(path_kii)

    # ── Preprocessing ──
    print("[2/5] Preprocessing & Standardisasi Dimensi...")
    df_masy = preprocess_dataframe(df_kelompok_raw, source_label="Masyarakat")
    df_pem = preprocess_dataframe(df_kii_raw, source_label="Pemerintah")

    print(f"    ✓ Masyarakat: {len(df_masy)} baris valid")
    print(f"    ✓ Pemerintah: {len(df_pem)} baris valid")
    print(f"    ✓ Distribusi Pilar Masyarakat:\n{df_masy['Pilar_Sektoral'].value_counts().to_string()}")

    # ── TF-IDF ──
    print("[3/5] Ekstraksi TF-IDF (N-Gram 2,3) per Wilayah & Pilar...")
    tfidf_results = run_tfidf_all_segments(df_masy, df_pem, top_n=top_n_tfidf)

    # ── Gap Analysis ──
    print("[4/5] Gap Analysis — Komparasi Horizontal...")
    wilayah_all = sorted(set(df_masy["Wilayah"].unique().tolist() +
                             df_pem["Wilayah"].unique().tolist()))
    pilar_all = list(PILAR_KEYWORDS.keys())

    gap_df = run_gap_analysis(tfidf_results, wilayah_all, pilar_all)
    print(f"    ✓ Gap Matrix: {gap_df.shape[0]} sel Wilayah×Pilar")

    # ── Benang Merah ──
    print("[5/5] Ekstraksi Benang Merah per Wilayah...")
    benang_merah = {}
    benang_merah["Semua Wilayah"] = get_benang_merah(df_masy)
    for wil in wilayah_all:
        benang_merah[wil] = get_benang_merah(df_masy, wil)

    print("\n✅ Pipeline selesai!\n")

    return {
        "df_masy": df_masy,
        "df_pem": df_pem,
        "df_masy_raw": df_kelompok_raw,
        "df_pem_raw": df_kii_raw,
        "tfidf_results": tfidf_results,
        "gap_df": gap_df,
        "benang_merah": benang_merah,
        "wilayah_list": wilayah_all,
        "pilar_list": pilar_all,
    }
