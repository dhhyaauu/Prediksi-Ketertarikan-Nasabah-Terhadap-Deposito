"""
=====================================================================
 APLIKASI PREDIKSI KETERTARIKAN NASABAH TERHADAP DEPOSITO
 Author : Saffa Dhiya Ur Rahma
 Stack  : Streamlit + Scikit-Learn + Plotly + Seaborn
 Run    : streamlit run app_streamlit.py
=====================================================================

Struktur folder yang direkomendasikan:
.
├── app_streamlit.py
├── assets/
│   └── profile.jpg          # Foto profil (upload manual)
├── data/
│   └── bank-full.csv        # Dataset UCI Bank Marketing
├── notebook/
│   └── notebook.html        # Hasil export Jupyter Notebook (HTML)
├── model/
│   └── model.pkl            # Model terlatih (opsional)
└── requirements.txt
"""

# =====================================================================
# 1. IMPORT LIBRARY
# =====================================================================
import os
import base64
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

try:
    from streamlit_option_menu import option_menu
except ImportError:
    st.error("⚠️ Install dulu: pip install streamlit-option-menu")
    st.stop()

warnings.filterwarnings("ignore")

# =====================================================================
# 2. KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="Prediksi Deposito Nasabah | Saffa Dhiya",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# 3. PATH KONSTAN
# =====================================================================
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "bank-full.csv"
NOTEBOOK_PATH = BASE_DIR / "notebook" / "notebook.html"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
PROFILE_PATH = BASE_DIR / "assets" / "profile.jpg"

# =====================================================================
# 4. CUSTOM CSS — TEMA PREMIUM NAVY / WHITE / DARK
# =====================================================================
CUSTOM_CSS = """
<style>
/* === Font === */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* === Background === */
.stApp {
    background:
      radial-gradient(1200px 600px at 10% -10%, rgba(30, 58, 138, 0.08), transparent 60%),
      radial-gradient(900px 500px at 110% 10%, rgba(15, 23, 42, 0.05), transparent 60%),
      linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
}

/* === Hide Streamlit branding === */
#MainMenu, footer, header {visibility: hidden;}

/* === Gradient Header === */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #1e40af 100%);
    padding: 2.5rem 2rem;
    border-radius: 24px;
    color: #ffffff;
    box-shadow: 0 20px 50px -20px rgba(15, 23, 42, 0.45);
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
    animation: fadeInDown 0.7s ease;
}
.hero-header::before {
    content: "";
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.12), transparent 70%);
    border-radius: 50%;
}
.hero-header h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    margin: 0 0 .5rem 0;
    letter-spacing: -0.02em;
}
.hero-header p {
    margin: 0;
    opacity: 0.85;
    font-size: 1.05rem;
}
.hero-badge {
    display: inline-block;
    padding: 6px 14px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    backdrop-filter: blur(10px);
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

/* === Card === */
.premium-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 1.75rem;
    box-shadow: 0 4px 24px -8px rgba(15, 23, 42, 0.08);
    transition: all 0.3s cubic-bezier(.4,0,.2,1);
    margin-bottom: 1rem;
}
.premium-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px -15px rgba(30, 58, 138, 0.25);
    border-color: #1e3a8a;
}

/* === Profile Card === */
.profile-card {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 28px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 25px 60px -25px rgba(15, 23, 42, 0.3);
    transition: all 0.4s ease;
    animation: fadeInUp 0.7s ease;
}
.profile-card:hover { transform: translateY(-6px); }

.profile-img-wrapper {
    width: 160px; height: 160px;
    margin: 0 auto 1.25rem auto;
    border-radius: 50%;
    padding: 4px;
    background: linear-gradient(135deg, #0f172a, #1e40af, #3b82f6);
    box-shadow: 0 15px 40px -10px rgba(30, 58, 138, 0.4);
    animation: pulseRing 3s ease-in-out infinite;
}
.profile-img-wrapper img,
.profile-img-wrapper .placeholder-avatar {
    width: 100%; height: 100%;
    border-radius: 50%;
    object-fit: cover;
    border: 4px solid #ffffff;
}
.placeholder-avatar {
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    color: #fff; font-size: 3.5rem; font-weight: 700;
}

@keyframes pulseRing {
    0%, 100% { box-shadow: 0 15px 40px -10px rgba(30,58,138,0.4); }
    50%      { box-shadow: 0 20px 55px -10px rgba(59,130,246,0.55); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.profile-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.8rem; font-weight: 800;
    color: #0f172a; margin: 0;
}
.profile-role {
    color: #1e40af; font-weight: 600;
    font-size: 1rem; margin: .25rem 0 1rem 0;
}
.profile-desc {
    color: #475569; line-height: 1.7;
    font-size: .95rem; max-width: 520px; margin: 0 auto;
}

/* === Info Item === */
.info-item {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    margin: 8px 0;
    transition: all 0.3s ease;
}
.info-item:hover {
    background: #eff6ff;
    border-color: #3b82f6;
    transform: translateX(4px);
}
.info-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.info-label { font-size: .75rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em;}
.info-value { font-size: .95rem; color: #0f172a; font-weight: 600; }

/* === Social === */
.social-row { display: flex; gap: 12px; justify-content: center; margin-top: 1.25rem; }
.social-btn {
    width: 44px; height: 44px; border-radius: 50%;
    background: #0f172a; color: #fff !important;
    display: flex; align-items: center; justify-content: center;
    text-decoration: none; font-size: 1.05rem;
    transition: all 0.3s ease;
}
.social-btn:hover {
    background: #1e40af;
    transform: translateY(-3px) scale(1.08);
    box-shadow: 0 10px 25px -8px rgba(30,64,175,0.5);
}

/* === Stat Card === */
.stat-card {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 16px; padding: 1.25rem;
    transition: all .3s ease;
}
.stat-card:hover {
    border-color: #1e40af;
    box-shadow: 0 10px 30px -10px rgba(30,64,175,.25);
    transform: translateY(-2px);
}
.stat-label { font-size: .8rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
.stat-value { font-size: 1.75rem; font-weight: 800; color: #0f172a; margin: 6px 0 0; font-family: 'Plus Jakarta Sans';}
.stat-trend { font-size: .8rem; color: #16a34a; font-weight: 600; }

/* === Workflow step === */
.workflow-step {
    background: #fff; border: 1px solid #e2e8f0; border-left: 4px solid #1e40af;
    border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: .75rem;
    transition: all .3s ease;
}
.workflow-step:hover {
    border-left-color: #3b82f6;
    transform: translateX(6px);
    box-shadow: 0 8px 24px -10px rgba(30,64,175,.2);
}
.workflow-num {
    display: inline-block;
    width: 30px; height: 30px; line-height: 30px; text-align: center;
    border-radius: 50%; background: linear-gradient(135deg,#1e3a8a,#3b82f6);
    color: #fff; font-weight: 700; margin-right: 10px;
}

/* === Sidebar === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #fff !important; }
.sb-title {
    font-family: 'Plus Jakarta Sans'; font-weight: 800;
    font-size: 1.15rem; color: #fff; margin-bottom: .5rem;
}
.sb-section {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1rem; margin-bottom: 1rem;
}
.sb-step {
    display:flex; gap:10px; align-items:flex-start;
    padding: 6px 0; font-size: .88rem; color:#cbd5e1;
}
.sb-step b { color: #fff; }
.status-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    margin-right:8px;
}
.status-ok  { background:#22c55e; box-shadow:0 0 8px #22c55e; }
.status-bad { background:#ef4444; box-shadow:0 0 8px #ef4444; }

/* === Result === */
.result-card-success {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #10b981;
    border-left: 6px solid #10b981;
    border-radius: 20px; padding: 2rem; text-align: center;
    animation: fadeInUp 0.6s ease;
}
.result-card-fail {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border: 1px solid #ef4444;
    border-left: 6px solid #ef4444;
    border-radius: 20px; padding: 2rem; text-align: center;
    animation: fadeInUp 0.6s ease;
}
.result-title { font-family:'Plus Jakarta Sans'; font-weight:800; font-size:1.6rem; margin:0; }
.result-prob  { font-size: 3rem; font-weight: 800; margin: .5rem 0; font-family:'Plus Jakarta Sans'; }

/* === Footer === */
.app-footer {
    text-align: center; padding: 2rem 1rem 1rem 1rem;
    color: #64748b; font-size: .85rem; margin-top: 3rem;
    border-top: 1px solid #e2e8f0;
}

/* === Tabs === */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background:#fff; border:1px solid #e2e8f0; border-radius:12px;
    padding: 10px 22px; font-weight: 600; color:#475569;
    transition: all .3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#0f172a,#1e40af) !important;
    color: #fff !important; border-color: transparent !important;
    box-shadow: 0 8px 20px -8px rgba(30,64,175,.5);
}

/* === Buttons === */
.stButton > button {
    background: linear-gradient(135deg,#0f172a,#1e40af) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    transition: all .3s ease !important;
    box-shadow: 0 8px 20px -8px rgba(30,64,175,.5) !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px -10px rgba(30,64,175,.6) !important;
}

/* === Section heading === */
.section-h {
    font-family:'Plus Jakarta Sans'; font-weight:800;
    color:#0f172a; font-size:1.5rem; margin: 1.5rem 0 1rem 0;
    display:flex; align-items:center; gap:10px;
}
.section-h::before {
    content:""; width:5px; height:24px;
    background: linear-gradient(180deg,#1e3a8a,#3b82f6);
    border-radius: 4px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =====================================================================
# 5. HELPER FUNCTIONS
# =====================================================================
@st.cache_data(show_spinner=False)
def load_dataset():
    """Load dataset Bank Marketing (UCI)."""
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH, sep=";")
            return df, None
        except Exception as e:
            return None, str(e)
    return None, "Dataset belum diupload"

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model pickle bila tersedia."""
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f), None
        except Exception as e:
            return None, str(e)
    return None, "Model belum tersedia"

def img_to_base64(path: Path):
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None

def hero(title, subtitle, badge="Premium Dashboard"):
    st.markdown(f"""
    <div class="hero-header">
        <span class="hero-badge">✨ {badge}</span>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# 6. SIDEBAR
# =====================================================================
def render_sidebar(df, model):
    with st.sidebar:
        st.markdown('<div class="sb-title">💎 ML Studio</div>', unsafe_allow_html=True)
        st.caption("Prediksi Deposito Nasabah")
        st.markdown("---")

        st.markdown('<div class="sb-title">📘 Panduan Penggunaan</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sb-section">
            <div class="sb-step">1️⃣ <span>Buka menu <b>Tentang Aplikasi</b> untuk memahami konteks.</span></div>
            <div class="sb-step">2️⃣ <span>Cek <b>Analisis Data</b> untuk eksplorasi dataset & notebook.</span></div>
            <div class="sb-step">3️⃣ <span>Gunakan <b>Prediksi</b> untuk mencoba model.</span></div>
            <div class="sb-step">4️⃣ <span>Pelajari hasil & interpretasi prediksi.</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-title">⚡ Fitur Utama</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sb-section">
            <div class="sb-step">📊 <span>EDA interaktif</span></div>
            <div class="sb-step">📓 <span>Dokumentasi notebook</span></div>
            <div class="sb-step">🤖 <span>Prediksi real-time</span></div>
            <div class="sb-step">📈 <span>Visualisasi premium</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-title">💡 Tips</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sb-section" style="font-size:.85rem; color:#cbd5e1;">
            Isi form prediksi sesuai profil nasabah aktual untuk hasil yang akurat. 
            Cek tab <b>Penjelasan Istilah</b> bila ragu dengan kolom input.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-title">📡 Status Sistem</div>', unsafe_allow_html=True)
        ds_status = "ok" if df is not None else "bad"
        md_status = "ok" if model is not None else "bad"
        st.markdown(f"""
        <div class="sb-section">
            <div class="sb-step">
                <span class="status-dot status-{ds_status}"></span>
                <span>Dataset: <b>{"Tersedia" if df is not None else "Belum ada"}</b></span>
            </div>
            <div class="sb-step">
                <span class="status-dot status-{md_status}"></span>
                <span>Model: <b>{"Siap" if model is not None else "Dummy mode"}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; font-size:.75rem; color:#64748b; margin-top:1rem;">
            v1.0.0 · Built with ❤️<br/>
            © 2026 Saffa Dhiya
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# 7. PAGE: TENTANG SAYA
# =====================================================================
def page_tentang_saya():
    hero("👋 Tentang Saya",
         "Profil singkat pengembang aplikasi ini.",
         badge="Profile")

    c1, c2 = st.columns([1, 1.3], gap="large")

    with c1:
        avatar = img_to_base64(PROFILE_PATH)
        if avatar:
            avatar_html = f'<img src="data:image/jpeg;base64,{avatar}" />'
        else:
            avatar_html = '<div class="placeholder-avatar">SD</div>'

        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-img-wrapper">{avatar_html}</div>
            <p class="profile-name">Saffa Dhiya Ur Rahma</p>
            <p class="profile-role">Rekayasa Perangkat Lunak</p>
            <p class="profile-desc">
                Antusias di bidang <b>Data Science</b> & <b>Machine Learning</b>.
                Senang membangun solusi data-driven yang elegan, interaktif, dan bermanfaat
                untuk pengambilan keputusan bisnis.
            </p>
            <div class="social-row">
                <a class="social-btn" href="mailto:saffadhiyaa1012@gmail.com" title="Email">✉️</a>
                <a class="social-btn" href="https://github.com/dhhyaauu" target="_blank" title="GitHub">🐙</a>
                <a class="social-btn" href="#" title="LinkedIn">💼</a>
                <a class="social-btn" href="#" title="Portfolio">🌐</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-h">📇 Informasi Kontak</div>', unsafe_allow_html=True)
        contacts = [
            ("👤", "Nama Lengkap", "Saffa Dhiya Ur Rahma"),
            ("🎓", "Jurusan", "Rekayasa Perangkat Lunak"),
            ("✉️", "Email", "saffadhiyaa1012@gmail.com"),
            ("🐙", "GitHub", "github.com/dhhyaauu"),
            ("📍", "Fokus", "Data Science & Machine Learning"),
        ]
        for icon, label, value in contacts:
            st.markdown(f"""
            <div class="info-item">
                <div class="info-icon">{icon}</div>
                <div>
                    <div class="info-label">{label}</div>
                    <div class="info-value">{value}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-h">🛠️ Skills</div>', unsafe_allow_html=True)
        skills = ["Python", "Pandas", "Scikit-Learn", "Streamlit",
                  "SQL", "Plotly", "EDA", "Machine Learning"]
        chips = " ".join(
            f'<span style="display:inline-block;padding:6px 14px;margin:4px;'
            f'background:#eff6ff;color:#1e40af;border-radius:999px;'
            f'font-size:.82rem;font-weight:600;border:1px solid #bfdbfe;">{s}</span>'
            for s in skills
        )
        st.markdown(f'<div class="premium-card">{chips}</div>', unsafe_allow_html=True)

# =====================================================================
# 8. PAGE: TENTANG APLIKASI
# =====================================================================
def page_tentang_aplikasi():
    hero("💼 Tentang Aplikasi",
         "Dashboard machine learning untuk prediksi ketertarikan nasabah terhadap produk deposito berjangka.",
         badge="Overview")

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("Algoritma", "ML Klasifikasi", "🤖"),
        ("Dataset", "UCI Bank Mkt.", "🏦"),
        ("Fitur Input", "16 Atribut", "📋"),
        ("Target", "Subscribe (y/n)", "🎯"),
    ]
    for col, (label, value, icon) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size:1.6rem;">{icon}</div>
                <div class="stat-label">{label}</div>
                <div class="stat-value" style="font-size:1.15rem;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-h">📖 Penjelasan & Tujuan</div>', unsafe_allow_html=True)
    cc1, cc2 = st.columns(2, gap="large")
    with cc1:
        st.markdown("""
        <div class="premium-card">
            <h3 style="margin-top:0;color:#1e3a8a;">💡 Apa Aplikasi Ini?</h3>
            <p style="color:#475569;line-height:1.7;">
                Aplikasi ini memanfaatkan model <b>machine learning</b> untuk
                memprediksi apakah seorang nasabah akan tertarik berlangganan
                <b>deposito berjangka</b> berdasarkan data kampanye pemasaran
                sebuah bank di Portugal (UCI Bank Marketing Dataset).
            </p>
        </div>
        """, unsafe_allow_html=True)
    with cc2:
        st.markdown("""
        <div class="premium-card">
            <h3 style="margin-top:0;color:#1e3a8a;">🎯 Tujuan</h3>
            <p style="color:#475569;line-height:1.7;">
                Membantu tim marketing bank menargetkan calon nasabah yang
                berpotensi tinggi untuk berlangganan, sehingga kampanye lebih
                efisien, biaya rendah, dan tingkat konversi meningkat.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-h">⚙️ Teknologi yang Digunakan</div>', unsafe_allow_html=True)
    tech = [
        ("🐍", "Python", "Bahasa pemrograman utama"),
        ("🎈", "Streamlit", "Framework dashboard"),
        ("🧮", "Scikit-Learn", "Library machine learning"),
        ("📊", "Plotly / Seaborn", "Visualisasi interaktif"),
        ("🐼", "Pandas / Numpy", "Manipulasi & analisis data"),
        ("🎨", "Custom CSS", "Desain premium responsif"),
    ]
    tcols = st.columns(3)
    for i, (icon, name, desc) in enumerate(tech):
        with tcols[i % 3]:
            st.markdown(f"""
            <div class="premium-card" style="text-align:center;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-weight:700;color:#0f172a;margin-top:.5rem;">{name}</div>
                <div style="color:#64748b;font-size:.85rem;margin-top:.25rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-h">🔄 Alur Kerja Machine Learning</div>', unsafe_allow_html=True)
    steps = [
        ("Pengumpulan Data", "Dataset diambil dari UCI Bank Marketing Repository."),
        ("Data Cleaning", "Penanganan missing value, duplikasi, dan outlier."),
        ("Preprocessing", "Encoding fitur kategorik, scaling fitur numerik."),
        ("Modeling", "Pelatihan beberapa algoritma klasifikasi."),
        ("Evaluasi", "Akurasi, Precision, Recall, F1-Score, ROC-AUC."),
        ("Deployment", "Integrasi model ke aplikasi Streamlit ini."),
    ]
    for i, (t, d) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="workflow-step">
            <span class="workflow-num">{i}</span>
            <b style="color:#0f172a;">{t}</b>
            <div style="color:#64748b;margin-top:4px;font-size:.92rem;">{d}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-h">🌟 Manfaat Aplikasi</div>', unsafe_allow_html=True)
    benefits = [
        ("⚡", "Efisien", "Hemat waktu kampanye pemasaran."),
        ("🎯", "Tepat Sasaran", "Targeting nasabah potensial."),
        ("📈", "Konversi Naik", "Tingkat subscribe lebih tinggi."),
        ("💰", "Hemat Biaya", "ROI kampanye lebih optimal."),
    ]
    bcols = st.columns(4)
    for col, (i, t, d) in zip(bcols, benefits):
        with col:
            st.markdown(f"""
            <div class="premium-card" style="text-align:center;">
                <div style="font-size:1.8rem;">{i}</div>
                <div style="font-weight:700;color:#1e3a8a;margin-top:.5rem;">{t}</div>
                <div style="color:#64748b;font-size:.85rem;">{d}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="premium-card" style="margin-top:1.5rem;
        background: linear-gradient(135deg,#0f172a,#1e40af); color:#fff; text-align:center;">
        <h3 style="margin:0 0 .5rem 0;">🚀 Siap mencoba?</h3>
        <p style="margin:0;opacity:.85;">Klik menu <b>Prediksi</b> di atas untuk mencoba model prediksi.</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# 9. PAGE: ANALISIS DATA
# =====================================================================
def page_analisis_data(df):
    hero("📊 Analisis Data",
         "Eksplorasi mendalam dataset, dokumentasi notebook, dan kamus istilah fitur.",
         badge="Data Studio")

    tab1, tab2, tab3 = st.tabs(["📁 Dataset", "📓 Notebook", "📚 Penjelasan Istilah"])

    # --------- TAB 1: DATASET ---------
    with tab1:
        if df is None:
            st.warning("⚠️ Dataset belum tersedia. Letakkan file di `data/bank-full.csv`.")
            return

        # Info sumber
        st.markdown("""
        <div class="premium-card">
            <h3 style="margin-top:0;color:#1e3a8a;">📦 Sumber Dataset</h3>
            <p style="color:#475569;line-height:1.7;">
                Dataset <b>Bank Marketing</b> dari <b>UCI Machine Learning Repository</b>,
                berisi data kampanye pemasaran direct marketing (telepon) sebuah bank di Portugal.
            </p>
            <a href="https://archive.ics.uci.edu/dataset/222/bank+marketing" target="_blank"
               style="color:#1e40af;font-weight:600;">🔗 archive.ics.uci.edu/dataset/222</a>
        </div>
        """, unsafe_allow_html=True)

        # Stat cards
        n_rows, n_cols = df.shape
        n_num = df.select_dtypes(include=np.number).shape[1]
        n_cat = df.select_dtypes(exclude=np.number).shape[1]

        c1, c2, c3, c4 = st.columns(4)
        for col, (lbl, val, icon) in zip(
            [c1, c2, c3, c4],
            [("Total Baris", f"{n_rows:,}", "📋"),
             ("Total Kolom", f"{n_cols}", "🗂️"),
             ("Kolom Numerik", f"{n_num}", "🔢"),
             ("Kolom Kategorik", f"{n_cat}", "🔤")]
        ):
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div class="stat-label">{lbl}</div>
                    <div class="stat-value">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-h">👀 Preview Dataset</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True, height=380)

        st.markdown('<div class="section-h">📐 Tipe Data & Info Kolom</div>', unsafe_allow_html=True)
        info_df = pd.DataFrame({
            "Kolom": df.columns,
            "Tipe Data": df.dtypes.astype(str).values,
            "Non-Null": df.notna().sum().values,
            "Unique": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True, height=380)

        st.markdown('<div class="section-h">📈 Statistik Deskriptif</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        st.markdown('<div class="section-h">❓ Missing Value</div>', unsafe_allow_html=True)
        miss = df.isna().sum().to_frame("Jumlah Missing")
        miss["Persentase (%)"] = (miss["Jumlah Missing"] / len(df) * 100).round(2)
        st.dataframe(miss, use_container_width=True)
        if miss["Jumlah Missing"].sum() == 0:
            st.success("✅ Tidak ada missing value dalam dataset.")

        # Distribusi target
        if "y" in df.columns:
            st.markdown('<div class="section-h">🎯 Distribusi Target (y)</div>', unsafe_allow_html=True)
            target_counts = df["y"].value_counts().reset_index()
            target_counts.columns = ["y", "count"]
            fig = px.pie(target_counts, names="y", values="count", hole=0.55,
                         color_discrete_sequence=["#1e40af", "#3b82f6"])
            fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        # Distribusi numerik
        st.markdown('<div class="section-h">📊 Distribusi Kolom Numerik</div>', unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            sel = st.selectbox("Pilih kolom numerik", num_cols)
            fig = px.histogram(df, x=sel, nbins=40, color_discrete_sequence=["#1e40af"])
            fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)

        # Bar chart kategorik
        st.markdown('<div class="section-h">🔤 Distribusi Kolom Kategorik</div>', unsafe_allow_html=True)
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        if cat_cols:
            selc = st.selectbox("Pilih kolom kategorik", cat_cols, key="cat_sel")
            vc = df[selc].value_counts().reset_index()
            vc.columns = [selc, "count"]
            fig = px.bar(vc, x=selc, y="count", color="count",
                         color_continuous_scale="Blues")
            fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        # Heatmap korelasi
        st.markdown('<div class="section-h">🔥 Heatmap Korelasi</div>', unsafe_allow_html=True)
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f",
                        linewidths=.5, ax=ax, cbar_kws={"shrink": .8})
            ax.set_title("Korelasi antar Fitur Numerik", fontsize=13, fontweight="bold", color="#0f172a")
            st.pyplot(fig, clear_figure=True)

        # Insight otomatis
        st.markdown('<div class="section-h">💡 Insight Otomatis</div>', unsafe_allow_html=True)
        insights = []
        if "y" in df.columns:
            pct_yes = (df["y"] == "yes").mean() * 100
            insights.append(f"Tingkat nasabah yang berlangganan deposito hanya **{pct_yes:.2f}%** — dataset sangat *imbalanced*.")
        if "age" in df.columns:
            insights.append(f"Rata-rata umur nasabah: **{df['age'].mean():.1f} tahun**, kisaran {df['age'].min()}–{df['age'].max()}.")
        if "balance" in df.columns:
            insights.append(f"Median saldo: **{df['balance'].median():,.0f}** — distribusi sangat skewed.")
        if "duration" in df.columns:
            insights.append(f"Durasi telepon rata-rata: **{df['duration'].mean():.0f} detik**, fitur ini sangat prediktif.")

        for ins in insights:
            st.markdown(f'<div class="info-item"><div class="info-icon">💡</div><div>{ins}</div></div>',
                        unsafe_allow_html=True)

    # --------- TAB 2: NOTEBOOK ---------
    with tab2:
        st.markdown("""
        <div class="premium-card">
            <h3 style="margin-top:0;color:#1e3a8a;">📓 Dokumentasi Notebook</h3>
            <p style="color:#475569;line-height:1.7;">
                Berikut adalah tampilan lengkap notebook Jupyter yang berisi seluruh proses
                machine learning: <b>preprocessing → cleaning → training → evaluasi → prediksi</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Stage explanation
        stages = [
            ("🧹 Preprocessing & Cleaning",
             "Penanganan nilai 'unknown', encoding kategorik (Label/One-Hot), scaling numerik, dan penanganan outlier."),
            ("⚖️ Handling Imbalanced Data",
             "Penerapan teknik resampling (SMOTE / class_weight) karena target sangat tidak seimbang."),
            ("🤖 Training Model",
             "Eksperimen beberapa algoritma: Logistic Regression, Random Forest, Gradient Boosting, dll."),
            ("📐 Evaluasi Model",
             "Pengukuran Accuracy, Precision, Recall, F1, ROC-AUC, serta Confusion Matrix."),
            ("🔮 Prediksi",
             "Model terbaik disimpan dalam pickle dan digunakan pada halaman Prediksi."),
        ]
        for t, d in stages:
            st.markdown(f"""
            <div class="workflow-step">
                <b style="color:#0f172a;">{t}</b>
                <div style="color:#64748b;margin-top:4px;font-size:.92rem;">{d}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-h">📄 Tampilan Notebook</div>', unsafe_allow_html=True)
        if NOTEBOOK_PATH.exists():
            html_content = NOTEBOOK_PATH.read_text(encoding="utf-8", errors="ignore")
            st.components.v1.html(html_content, height=900, scrolling=True)
        else:
            st.info("📁 Letakkan file notebook hasil export HTML di `notebook/notebook.html`.")

    # --------- TAB 3: PENJELASAN ISTILAH ---------
    with tab3:
        st.markdown("""
        <div class="premium-card">
            <h3 style="margin-top:0;color:#1e3a8a;">📚 Kamus Fitur Dataset</h3>
            <p style="color:#475569;">
                Penjelasan setiap kolom yang digunakan sebagai input prediksi.
            </p>
        </div>
        """, unsafe_allow_html=True)

        features = [
            ("age", "Usia nasabah dalam tahun.", "Numerik"),
            ("job", "Jenis pekerjaan nasabah (admin, blue-collar, dst).", "Kategorik"),
            ("marital", "Status pernikahan (single, married, divorced).", "Kategorik"),
            ("education", "Tingkat pendidikan (primary, secondary, tertiary).", "Kategorik"),
            ("default", "Apakah memiliki kredit macet (yes/no).", "Biner"),
            ("balance", "Saldo rata-rata tahunan dalam euro.", "Numerik"),
            ("housing", "Memiliki pinjaman rumah (yes/no).", "Biner"),
            ("loan", "Memiliki pinjaman pribadi (yes/no).", "Biner"),
            ("contact", "Jenis kontak komunikasi (cellular/telephone).", "Kategorik"),
            ("day", "Hari terakhir dihubungi dalam bulan.", "Numerik"),
            ("month", "Bulan terakhir dihubungi.", "Kategorik"),
            ("duration", "Durasi kontak terakhir dalam detik.", "Numerik"),
            ("campaign", "Jumlah kontak selama kampanye ini.", "Numerik"),
            ("pdays", "Hari sejak kontak terakhir kampanye sebelumnya (-1 = belum pernah).", "Numerik"),
            ("previous", "Jumlah kontak sebelum kampanye ini.", "Numerik"),
            ("poutcome", "Hasil kampanye sebelumnya (success, failure, other, unknown).", "Kategorik"),
            ("y", "Target: apakah berlangganan deposito (yes/no).", "Target"),
        ]

        cols = st.columns(2)
        for i, (name, desc, typ) in enumerate(features):
            with cols[i % 2]:
                badge_color = {"Numerik": "#1e40af", "Kategorik": "#7c3aed",
                               "Biner": "#0891b2", "Target": "#dc2626"}.get(typ, "#475569")
                st.markdown(f"""
                <div class="premium-card" title="{desc}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <b style="color:#0f172a;font-size:1.05rem;">{name}</b>
                        <span style="background:{badge_color}15;color:{badge_color};
                              padding:4px 10px;border-radius:999px;font-size:.72rem;
                              font-weight:700;">{typ}</span>
                    </div>
                    <p style="color:#64748b;margin:.5rem 0 0;font-size:.9rem;line-height:1.5;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

# =====================================================================
# 10. PAGE: PREDIKSI
# =====================================================================
def page_prediksi(df, model):
    hero("🔮 Prediksi Deposito",
         "Isi form di bawah untuk memprediksi apakah nasabah akan berlangganan deposito berjangka.",
         badge="ML Prediction")

    # Opsi default
    job_opts = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                "retired", "self-employed", "services", "student", "technician",
                "unemployed", "unknown"]
    marital_opts = ["married", "single", "divorced"]
    edu_opts = ["primary", "secondary", "tertiary", "unknown"]
    contact_opts = ["cellular", "telephone", "unknown"]
    month_opts = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"]
    poutcome_opts = ["unknown", "failure", "other", "success"]

    if df is not None:
        if "job" in df: job_opts = sorted(df["job"].dropna().unique().tolist())
        if "marital" in df: marital_opts = sorted(df["marital"].dropna().unique().tolist())
        if "education" in df: edu_opts = sorted(df["education"].dropna().unique().tolist())
        if "contact" in df: contact_opts = sorted(df["contact"].dropna().unique().tolist())
        if "month" in df: month_opts = sorted(df["month"].dropna().unique().tolist())
        if "poutcome" in df: poutcome_opts = sorted(df["poutcome"].dropna().unique().tolist())

    st.markdown('<div class="section-h">📝 Formulir Data Nasabah</div>', unsafe_allow_html=True)

    with st.form("form_prediksi"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**👤 Data Pribadi**")
            age = st.slider("Usia", 18, 100, 35, help="Usia nasabah (tahun).")
            job = st.selectbox("Pekerjaan", job_opts, help="Jenis pekerjaan nasabah.")
            marital = st.selectbox("Status Pernikahan", marital_opts)
            education = st.selectbox("Pendidikan", edu_opts)

        with c2:
            st.markdown("**💰 Data Finansial**")
            balance = st.number_input("Saldo Rata-rata (EUR)", -10000, 200000, 1500,
                                       help="Saldo rata-rata tahunan dalam euro.")
            default = st.selectbox("Kredit Macet?", ["no", "yes"])
            housing = st.selectbox("Pinjaman Rumah?", ["no", "yes"])
            loan = st.selectbox("Pinjaman Pribadi?", ["no", "yes"])

        with c3:
            st.markdown("**📞 Data Kampanye**")
            contact = st.selectbox("Jenis Kontak", contact_opts)
            month = st.selectbox("Bulan Kontak Terakhir", month_opts)
            day = st.slider("Hari Kontak", 1, 31, 15)
            duration = st.number_input("Durasi Kontak (detik)", 0, 5000, 200,
                                        help="Durasi panggilan terakhir.")

        c4, c5, c6 = st.columns(3)
        with c4:
            campaign = st.number_input("Jumlah Kontak Kampanye", 1, 100, 2,
                                        help="Kontak selama kampanye ini.")
        with c5:
            pdays = st.number_input("Hari Sejak Kontak Sebelumnya", -1, 999, -1,
                                     help="-1 = belum pernah dikontak.")
        with c6:
            previous = st.number_input("Jumlah Kontak Sebelumnya", 0, 100, 0)

        poutcome = st.selectbox("Hasil Kampanye Sebelumnya", poutcome_opts)

        submitted = st.form_submit_button("🚀 Prediksi Sekarang", use_container_width=True)

    if submitted:
        with st.spinner("⏳ Memproses prediksi..."):
            import time
            time.sleep(1.0)

            input_dict = {
                "age": age, "job": job, "marital": marital, "education": education,
                "default": default, "balance": balance, "housing": housing, "loan": loan,
                "contact": contact, "day": day, "month": month, "duration": duration,
                "campaign": campaign, "pdays": pdays, "previous": previous,
                "poutcome": poutcome,
            }

            # === Inferensi ===
            if model is not None:
                try:
                    input_df = pd.DataFrame([input_dict])
                    pred = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0]
                    prob_yes = float(proba[list(model.classes_).index("yes")]) \
                        if "yes" in list(model.classes_) else float(proba[1])
                    label = "yes" if str(pred).lower() in ("yes", "1", "true") else "no"
                except Exception as e:
                    st.error(f"❌ Error saat prediksi: {e}")
                    return
            else:
                # Heuristik dummy bila model belum tersedia
                score = 0.15
                if duration > 300: score += 0.30
                if poutcome == "success": score += 0.35
                if balance > 2000: score += 0.10
                if housing == "no": score += 0.05
                if loan == "no": score += 0.05
                score = min(score, 0.98)
                prob_yes = score
                label = "yes" if prob_yes >= 0.5 else "no"

        # === Tampil hasil ===
        st.markdown('<div class="section-h">🎯 Hasil Prediksi</div>', unsafe_allow_html=True)

        if label == "yes":
            st.markdown(f"""
            <div class="result-card-success">
                <div style="font-size:3rem;">✅</div>
                <p class="result-title" style="color:#065f46;">Nasabah BERPOTENSI Berlangganan</p>
                <p class="result-prob" style="color:#10b981;">{prob_yes*100:.1f}%</p>
                <p style="color:#047857;">Probabilitas tinggi untuk subscribe deposito berjangka.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-fail">
                <div style="font-size:3rem;">⚠️</div>
                <p class="result-title" style="color:#7f1d1d;">Nasabah TIDAK Berpotensi Berlangganan</p>
                <p class="result-prob" style="color:#ef4444;">{(1-prob_yes)*100:.1f}%</p>
                <p style="color:#991b1b;">Probabilitas rendah untuk subscribe deposito berjangka.</p>
            </div>
            """, unsafe_allow_html=True)

        if model is None:
            st.info("ℹ️ Model belum tersedia — hasil di atas berasal dari simulasi heuristik (dummy).")

        # Grafik probabilitas
        st.markdown('<div class="section-h">📊 Visualisasi Probabilitas</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_yes * 100,
                number={"suffix": "%", "font": {"size": 40, "color": "#0f172a"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                    "bar": {"color": "#1e40af"},
                    "steps": [
                        {"range": [0, 40], "color": "#fee2e2"},
                        {"range": [40, 70], "color": "#fef3c7"},
                        {"range": [70, 100], "color": "#d1fae5"},
                    ],
                    "threshold": {"line": {"color": "#0f172a", "width": 4},
                                  "thickness": 0.75, "value": prob_yes * 100},
                },
                title={"text": "Probabilitas Subscribe"},
            ))
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with cc2:
            bar = pd.DataFrame({
                "Kelas": ["Tidak Berlangganan", "Berlangganan"],
                "Probabilitas": [(1 - prob_yes) * 100, prob_yes * 100],
            })
            fig = px.bar(bar, x="Kelas", y="Probabilitas", color="Kelas",
                         text="Probabilitas",
                         color_discrete_map={"Tidak Berlangganan": "#ef4444",
                                             "Berlangganan": "#10b981"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, yaxis_range=[0, 110])
            st.plotly_chart(fig, use_container_width=True)

        # Interpretasi
        st.markdown('<div class="section-h">🧠 Interpretasi Hasil</div>', unsafe_allow_html=True)
        notes = []
        if duration > 300:
            notes.append("Durasi kontak yang panjang menandakan minat tinggi.")
        if poutcome == "success":
            notes.append("Hasil kampanye sebelumnya sukses → indikator positif kuat.")
        if balance > 2000:
            notes.append("Saldo nasabah cukup tinggi, kemampuan finansial baik.")
        if loan == "yes" or housing == "yes":
            notes.append("Adanya pinjaman aktif dapat menurunkan minat berlangganan.")
        if default == "yes":
            notes.append("Riwayat kredit macet menjadi sinyal negatif.")
        if not notes:
            notes.append("Profil nasabah berada pada kategori standar.")

        for n in notes:
            st.markdown(f'<div class="info-item"><div class="info-icon">📌</div><div>{n}</div></div>',
                        unsafe_allow_html=True)

        # Ringkasan input
        with st.expander("📋 Lihat ringkasan data input"):
            st.json(input_dict)

# =====================================================================
# 11. NAVIGASI UTAMA + ROUTER
# =====================================================================
def main():
    # Load sumber daya
    df, df_err = load_dataset()
    model, model_err = load_model()

    # Sidebar
    render_sidebar(df, model)

    # Top navbar
    selected = option_menu(
        menu_title=None,
        options=["Tentang Saya", "Tentang Aplikasi", "Analisis Data", "Prediksi"],
        icons=["person-circle", "info-circle", "bar-chart-line", "magic"],
        orientation="horizontal",
        default_index=1,
        styles={
            "container": {
                "padding": "8px",
                "background": "linear-gradient(135deg,#ffffff,#f1f5f9)",
                "border-radius": "16px",
                "border": "1px solid #e2e8f0",
                "box-shadow": "0 4px 20px -8px rgba(15,23,42,.08)",
                "margin-bottom": "1.25rem",
            },
            "icon": {"color": "#1e40af", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "font-weight": "600",
                "color": "#475569",
                "text-align": "center",
                "margin": "0 4px",
                "padding": "10px 18px",
                "border-radius": "12px",
                "--hover-color": "#eff6ff",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg,#0f172a,#1e40af)",
                "color": "white",
                "font-weight": "700",
                "box-shadow": "0 8px 20px -8px rgba(30,64,175,.5)",
            },
        },
    )

    # Router
    if selected == "Tentang Saya":
        page_tentang_saya()
    elif selected == "Tentang Aplikasi":
        page_tentang_aplikasi()
    elif selected == "Analisis Data":
        page_analisis_data(df)
    elif selected == "Prediksi":
        page_prediksi(df, model)

    # Footer
    st.markdown("""
    <div class="app-footer">
        © 2026, dibuat oleh <b style="color:#1e3a8a;">Saffa Dhiya Ur Rahma</b> · 
        Dibangun dengan Streamlit, Scikit-Learn, Plotly · 
        <a href="https://github.com/dhhyaauu" style="color:#1e40af;text-decoration:none;">GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
