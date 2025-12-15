"""
Prediksi Harga Rumah Jakarta Selatan
=====================================
Standalone Streamlit App - Deploy langsung ke Streamlit Cloud
Tanpa memerlukan API terpisah

Main file path: streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from collections import deque

# Optional imports
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Harga Rumah Jaksel",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# MODEL & DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the prediction model"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple possible paths
    possible_paths = [
        os.path.join(base_dir, "models", "production_model.pkl"),
        os.path.join(base_dir, "api", "models", "production_model.pkl"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                st.warning(f"Error loading model from {path}: {e}")
    
    return None

@st.cache_data
def load_reference_data():
    """Load reference data for drift detection"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try processed data
    x_train_path = os.path.join(base_dir, "data", "processed", "x_train.pkl")
    if os.path.exists(x_train_path):
        try:
            ref_data = pd.read_pickle(x_train_path)
            return {
                "mean": ref_data.mean().to_dict(),
                "std": ref_data.std().to_dict(),
                "min": ref_data.min().to_dict(),
                "max": ref_data.max().to_dict(),
                "count": len(ref_data)
            }
        except:
            pass
    
    # Fallback default stats
    return {
        "mean": {"LB": 150, "LT": 180, "KT": 4, "KM": 3, "GRS": 2},
        "std": {"LB": 80, "LT": 100, "KT": 2, "KM": 1.5, "GRS": 1},
        "min": {"LB": 30, "LT": 20, "KT": 1, "KM": 1, "GRS": 0},
        "max": {"LB": 2000, "LT": 2000, "KT": 15, "KM": 15, "GRS": 15},
        "count": 100
    }

@st.cache_data
def load_metrics():
    """Load model metrics"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(base_dir, "api", "models", "metrics.json"),
        os.path.join(base_dir, "models", "metrics.json"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except:
                pass
    
    # Fallback
    return {
        "r2": 0.72,
        "mape": 0.28,
        "last_updated": "N/A"
    }

# Load resources
model = load_model()
reference_stats = load_reference_data()
metrics = load_metrics()

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
if 'role' not in st.session_state:
    st.session_state.role = None
if 'prediction_logs' not in st.session_state:
    st.session_state.prediction_logs = deque(maxlen=100)

def login_user():
    st.session_state.role = 'user'
    st.rerun()

def login_admin(username, password):
    if username == "admin" and password == "admin123":
        st.session_state.role = 'admin'
        st.rerun()
    else:
        st.error("Username atau Password salah!")

def logout():
    st.session_state.role = None
    st.rerun()

# -----------------------------------------------------------------------------
# PREDICTION FUNCTION
# -----------------------------------------------------------------------------
def predict_price(lb, lt, kt, km, grs):
    """Make prediction using local model"""
    if model is None:
        return None, "Model tidak tersedia. Pastikan file model ada di repository."
    
    try:
        df = pd.DataFrame({
            "LB": [int(lb)],
            "LT": [int(lt)],
            "KT": [int(kt)],
            "KM": [int(km)],
            "GRS": [int(grs)]
        })
        
        prediction = model.predict(df)[0]
        
        # Log prediction
        st.session_state.prediction_logs.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": {"LB": lb, "LT": lt, "KT": kt, "KM": km, "GRS": grs},
            "prediction": float(prediction),
            "status": "success"
        })
        
        return prediction, None
    except Exception as e:
        return None, str(e)

# -----------------------------------------------------------------------------
# DRIFT DETECTION
# -----------------------------------------------------------------------------
def calculate_drift():
    """Calculate data drift from recent predictions"""
    logs = list(st.session_state.prediction_logs)
    successful = [l for l in logs if l.get("status") == "success"]
    
    if len(successful) < 5:
        return {
            "status": "insufficient_data",
            "current_samples": len(successful),
            "message": "Minimal 5 prediksi diperlukan"
        }
    
    features = ["LB", "LT", "KT", "KM", "GRS"]
    feature_names = {
        "LB": "Luas Bangunan",
        "LT": "Luas Tanah",
        "KT": "Kamar Tidur",
        "KM": "Kamar Mandi",
        "GRS": "Garasi"
    }
    
    recent_data = pd.DataFrame([l["input"] for l in successful])
    
    drift_report = {}
    for feature in features:
        if feature in recent_data.columns:
            recent_mean = recent_data[feature].mean()
            ref_mean = reference_stats["mean"].get(feature, 0)
            ref_std = reference_stats["std"].get(feature, 1)
            
            drift_score = abs(recent_mean - ref_mean) / ref_std if ref_std > 0 else 0
            
            if drift_score < 0.5:
                severity = "low"
            elif drift_score < 1.5:
                severity = "medium"
            else:
                severity = "high"
            
            drift_report[feature] = {
                "name": feature_names.get(feature, feature),
                "severity": severity,
                "ref_mean": round(ref_mean, 1),
                "current_mean": round(recent_mean, 1),
                "drift_score": round(drift_score, 2)
            }
    
    severities = [d["severity"] for d in drift_report.values()]
    if "high" in severities:
        overall = "high"
    elif "medium" in severities:
        overall = "medium"
    else:
        overall = "low"
    
    return {
        "status": overall,
        "features": drift_report,
        "sample_size": len(successful)
    }

# -----------------------------------------------------------------------------
# PAGES
# -----------------------------------------------------------------------------
def show_login_page():
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>🏠 Prediksi Harga Rumah Jakarta Selatan</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Sistem prediksi harga rumah menggunakan Machine Learning</p>", unsafe_allow_html=True)
    st.write("")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["👤 User", "🔒 Admin"])
        
        with tab1:
            st.info("Akses fitur prediksi harga rumah")
            if st.button("🚀 Masuk sebagai User", use_container_width=True, key="user_login"):
                login_user()
                
        with tab2:
            st.warning("Area khusus Administrator")
            username = st.text_input("Username", key="admin_user")
            password = st.text_input("Password", type="password", key="admin_pass")
            if st.button("🔐 Login Admin", use_container_width=True, key="admin_login"):
                login_admin(username, password)

def show_user_page():
    st.markdown("## 🏠 Prediksi Harga Rumah Jakarta Selatan")
    st.markdown("Estimasi harga rumah berdasarkan spesifikasi fisik")
    
    with st.sidebar:
        st.markdown("### 👤 User")
        if st.button("🚪 Logout"):
            logout()
        st.markdown("---")
        if model:
            st.success("✅ Model aktif")
        else:
            st.error("❌ Model tidak tersedia")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📝 Spesifikasi Rumah")
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                lb = st.number_input("Luas Bangunan (m²)", 30, 2000, 100)
            with c2:
                lt = st.number_input("Luas Tanah (m²)", 20, 2000, 120)
            
            c3, c4, c5 = st.columns(3)
            with c3:
                kt = st.number_input("Kamar Tidur", 1, 15, 3)
            with c4:
                km = st.number_input("Kamar Mandi", 1, 15, 2)
            with c5:
                grs = st.number_input("Garasi", 0, 10, 1)
            
            submit = st.form_submit_button("🚀 Hitung Estimasi", use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Hasil Estimasi")
        if submit:
            with st.spinner("Menghitung..."):
                price, error = predict_price(lb, lt, kt, km, grs)
                
                if price is not None:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 30px; border-radius: 15px; text-align: center; color: white;">
                        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Estimasi Harga</p>
                        <h1 style="margin: 10px 0; font-size: 2.2em;">Rp {price:,.0f}</h1>
                        <p style="margin: 0; font-size: 0.8em; opacity: 0.8;">*Berdasarkan model Linear Regression</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"Error: {error}")
        else:
            st.info("👈 Isi form dan klik 'Hitung Estimasi'")

def show_admin_page():
    st.markdown("## 📊 Admin Dashboard")
    
    with st.sidebar:
        st.markdown("### 🔒 Admin")
        if st.button("🚪 Logout"):
            logout()
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "✅ Active" if model else "❌ Inactive"
        st.metric("Model Status", status)
    with col2:
        st.metric("R² Score", f"{metrics.get('r2', 0):.2%}")
    with col3:
        st.metric("MAPE", f"{metrics.get('mape', 0):.2%}")
    
    st.markdown(f"**Last Updated:** {metrics.get('last_updated', 'N/A')}")
    
    # Logs
    st.markdown("---")
    st.markdown("### 📋 Log Prediksi")
    
    logs = list(st.session_state.prediction_logs)
    total = len(logs)
    success = sum(1 for l in logs if l.get("status") == "success")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Request", total)
    c2.metric("Sukses", success)
    c3.metric("Success Rate", f"{(success/total*100) if total > 0 else 0:.0f}%")
    
    if logs:
        log_data = []
        for log in reversed(logs[-15:]):
            inp = log.get("input", {})
            log_data.append({
                "Waktu": log.get("timestamp", "-"),
                "LB": inp.get("LB", "-"),
                "LT": inp.get("LT", "-"),
                "KT": inp.get("KT", "-"),
                "KM": inp.get("KM", "-"),
                "GRS": inp.get("GRS", "-"),
                "Prediksi": f"Rp {log.get('prediction', 0):,.0f}" if log.get("prediction") else "-"
            })
        st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada log prediksi")
    
    # Drift Detection
    st.markdown("---")
    st.markdown("### 📈 Kualitas Data")
    
    drift = calculate_drift()
    
    if drift["status"] == "insufficient_data":
        st.info(f"Mengumpulkan data... ({drift['current_samples']}/5 prediksi)")
    else:
        status_config = {
            "low": ("✅ Data Sesuai", "#d4edda", "#155724"),
            "medium": ("⚠️ Ada Perubahan", "#fff3cd", "#856404"),
            "high": ("🔴 Perlu Perhatian", "#f8d7da", "#721c24")
        }
        title, bg, color = status_config.get(drift["status"], status_config["medium"])
        
        st.markdown(f"""
        <div style="background: {bg}; color: {color}; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
            <strong>{title}</strong> - Berdasarkan {drift['sample_size']} prediksi terakhir
        </div>
        """, unsafe_allow_html=True)
        
        if drift.get("features") and PLOTLY_AVAILABLE:
            features = drift["features"]
            names = [f["name"] for f in features.values()]
            ref_vals = [f["ref_mean"] for f in features.values()]
            cur_vals = [f["current_mean"] for f in features.values()]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Historis', x=names, y=ref_vals, marker_color='#6c757d'))
            fig.add_trace(go.Bar(name='Saat Ini', x=names, y=cur_vals, marker_color='#0d6efd'))
            fig.update_layout(barmode='group', height=350, margin=dict(t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .stButton > button { border-radius: 10px; }
    .stMetric { background: #f8f9fa; padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

if st.session_state.role is None:
    show_login_page()
elif st.session_state.role == 'user':
    show_user_page()
elif st.session_state.role == 'admin':
    show_admin_page()
