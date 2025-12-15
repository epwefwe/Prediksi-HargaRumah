import streamlit as st
import requests
import os
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Harga Rumah Jaksel",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL
API_URL = os.getenv("API_URL", "http://localhost:5000")

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
if 'role' not in st.session_state:
    st.session_state.role = None

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
# API HELPERS
# -----------------------------------------------------------------------------
def get_prediction(data):
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"Status {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_metrics():
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=2)
        if response.status_code == 200:
             data = response.json()
             if data.get("status") == "success":
                 return data.get("data")
        return None
    except:
        return None

def get_logs(limit=50):
    try:
        response = requests.get(f"{API_URL}/logs", params={"limit": limit}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data.get("data")
        return None
    except:
        return None

def get_drift():
    try:
        response = requests.get(f"{API_URL}/drift", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data.get("data")
        return None
    except:
        return None

# -----------------------------------------------------------------------------
# PAGES
# -----------------------------------------------------------------------------
def show_login_page():
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>🏠 House Price Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Silakan pilih akses login Anda</p>", unsafe_allow_html=True)
    st.write("")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["👤 User (Public)", "🔒 Admin (Metrik)"])
        
        with tab1:
            st.info("Akses fitur prediksi harga rumah secara gratis.")
            if st.button("🚀 Masuk sebagai User", use_container_width=True):
                login_user()
                
        with tab2:
            st.warning("Area terbatas khusus Administrator.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("🔐 Login Admin", use_container_width=True):
                login_admin(username, password)

def show_user_page():
    # Header
    st.markdown('<div class="main-header">🏠 Prediksi Harga Rumah Jakarta Selatan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Estimasi harga rumah impian Anda berdasarkan data pasar terbaru.</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/25/25694.png", width=50)
        st.write(f"Logged in as: **User**")
        if st.button("Logout", type="secondary"):
            logout()
    
    # Main Form
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 📝 Masukkan Spesifikasi Rumah")
        with st.form(key='house_form'):
            c1, c2 = st.columns(2)
            with c1:
                lb = st.number_input("Luas Bangunan (m²)", 30, 2000, 100)
            with c2:
                lt = st.number_input("Luas Tanah (m²)", 20, 2000, 120)
                
            c3, c4, c5 = st.columns(3)
            with c3:
                kt = st.number_input("Kamar Tidur", 1, 20, 3)
            with c4:
                km = st.number_input("Kamar Mandi", 1, 15, 2)
            with c5:
                grs = st.number_input("Garasi/Carport", 0, 10, 1)
                
            submit = st.form_submit_button("🚀 Hitung Estimasi Harga", use_container_width=True)

    with col_result:
        if submit:
            st.markdown("### 💰 Hasil Estimasi")
            with st.spinner("Mengkalkulasi harga pasar..."):
                payload = {"LB": lb, "LT": lt, "KT": kt, "KM": km, "GRS": grs}
                result = get_prediction(payload)
                
                if result.get("status") == "success":
                    price = result['prediction']
                    st.markdown(f"""
                    <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #1E88E5;">
                        <h4 style="margin:0; color: #1565C0;">Estimasi Harga Pasar</h4>
                        <h1 style="color: #0D47A1; margin: 10px 0;">Rp {price:,.0f}</h1>
                        <p style="margin:0; color: #555;">*Prediksi berdasarkan spesifikasi fisik rumah.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"Gagal memprediksi: {result.get('message')}")
        else:
             st.info("👈 Isi formulir di kiri untuk melihat estimasi.")

def show_admin_page():
    st.markdown('<div class="main-header">📊 Admin Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Monitoring Performa Model Regresi.</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/97/97895.png", width=50)
        st.write(f"Logged in as: **Admin**")
        if st.button("Logout", type="secondary"):
            logout()
            
    metrics = get_metrics()
    
    if metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
             st.metric("Model Status", "Active", "Online")
        with col2:
             st.metric("R2 Score (Akurasi)", f"{metrics['r2']:.4f}")
        with col3:
             st.metric("MAPE (Error Rate)", f"{metrics['mape']:.2%}", delta_color="inverse")
             
        st.markdown("---")
        st.markdown(f"### 🕒 Last Updated: {metrics['last_updated']}")
        
        st.markdown("### 📈 Detail Insights")
        st.info("Saat ini model menggunakan algoritma **Linear Regression**. Error rate 28% menunjukkan perlunya penambahan fitur lokasi untuk meningkatkan akurasi.")
        
        # -------------------------------------------------------------------------
        # MONITORING LOG SECTION
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📋 Monitoring Log Prediksi")
        
        logs_data = get_logs(limit=50)
        
        if logs_data:
            summary = logs_data.get("summary", {})
            logs = logs_data.get("logs", [])
            
            # Summary metrics
            log_col1, log_col2, log_col3, log_col4 = st.columns(4)
            with log_col1:
                st.metric("Total Request", summary.get("total_requests", 0))
            with log_col2:
                st.metric("Sukses", summary.get("success_count", 0), delta="✓", delta_color="normal")
            with log_col3:
                st.metric("Error", summary.get("error_count", 0), delta="✗" if summary.get("error_count", 0) > 0 else None, delta_color="inverse")
            with log_col4:
                st.metric("Success Rate", f"{summary.get('success_rate', 0):.1f}%")
            
            # Log table
            if logs:
                st.markdown("#### 📜 Riwayat Prediksi Terbaru")
                
                # Convert logs to display format
                log_display = []
                for log in logs[:20]:  # Show last 20
                    input_data = log.get("input", {})
                    log_display.append({
                        "Waktu": log.get("timestamp", "-"),
                        "Status": "✅ Sukses" if log.get("status") == "success" else "❌ Error",
                        "LB": input_data.get("LB", "-"),
                        "LT": input_data.get("LT", "-"),
                        "KT": input_data.get("KT", "-"),
                        "KM": input_data.get("KM", "-"),
                        "GRS": input_data.get("GRS", "-"),
                        "Prediksi (Rp)": f"{log.get('prediction', 0):,.0f}" if log.get("prediction") else "-",
                    })
                
                import pandas as pd
                df_logs = pd.DataFrame(log_display)
                st.dataframe(df_logs, use_container_width=True, hide_index=True)
            else:
                st.info("Belum ada log prediksi. Lakukan prediksi dari halaman User terlebih dahulu.")
        else:
            st.warning("⚠️ Tidak dapat mengambil data log. Pastikan API berjalan dengan benar.")
        
        # -------------------------------------------------------------------------
        # KUALITAS DATA - PEMANTAUAN KESEHATAN MODEL
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Pemantauan Kualitas Data")
        st.markdown("*Memastikan data yang masuk sesuai dengan pola yang dipelajari sistem*")
        
        drift_data = get_drift()
        
        if drift_data:
            overall_status = drift_data.get("overall_status", "unknown")
            sample_count = drift_data.get('sample_size', 0)
            reference_count = drift_data.get('reference_size', 0)
            detection_method = drift_data.get('method', 'unknown')
            dataset_drift = drift_data.get('dataset_drift', False)
            drift_share = drift_data.get('drift_share', 0)
            drifted_count = drift_data.get('drifted_features_count', 0)
            total_features = drift_data.get('total_features', 5)
            
            if overall_status == "insufficient_data":
                current_samples = drift_data.get('current_samples', 0)
                progress_pct = min((current_samples / 5) * 100, 100)
                
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 30px; border-radius: 12px; border: 1px solid #e9ecef;">
                    <div style="text-align: center;">
                        <div style="font-size: 3em; margin-bottom: 15px;">📊</div>
                        <h3 style="margin: 0; color: #495057; font-weight: 500;">Mengumpulkan Data Awal</h3>
                        <p style="color: #6c757d; margin: 10px 0 20px 0;">
                            Sistem membutuhkan minimal 5 data prediksi untuk mulai menganalisis kualitas data
                        </p>
                        <div style="background: #e9ecef; border-radius: 10px; height: 12px; max-width: 300px; margin: 0 auto;">
                            <div style="background: #0d6efd; width: {progress_pct}%; height: 100%; border-radius: 10px; transition: width 0.3s;"></div>
                        </div>
                        <p style="color: #495057; margin-top: 12px; font-weight: 500;">{current_samples} dari 5 data terkumpul</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Evidently Info Badge
                method_label = "Evidently" if detection_method == "evidently" else "Statistical Analysis"
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                    <span style="background: #e7f1ff; color: #0d6efd; padding: 4px 12px; border-radius: 20px; font-size: 0.75em; font-weight: 500;">
                        Powered by {method_label}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Konfigurasi status
                status_config = {
                    "low": {
                        "title": "Data Sesuai Standar",
                        "desc": "Karakteristik rumah yang diprediksi masih sesuai dengan pola data historis. Hasil prediksi dapat diandalkan.",
                        "color": "#198754",
                        "bg": "#d1e7dd",
                        "border": "#badbcc",
                        "indicator": "●"
                    },
                    "medium": {
                        "title": "Ada Perubahan Pola",
                        "desc": "Beberapa karakteristik rumah yang diprediksi mulai berbeda dari pola biasanya. Disarankan untuk memantau lebih lanjut.",
                        "color": "#fd7e14",
                        "bg": "#fff3cd",
                        "border": "#ffecb5",
                        "indicator": "●"
                    },
                    "high": {
                        "title": "Perlu Perhatian",
                        "desc": "Karakteristik rumah yang diprediksi cukup berbeda dari data yang dipelajari sistem. Akurasi prediksi mungkin terpengaruh.",
                        "color": "#dc3545",
                        "bg": "#f8d7da",
                        "border": "#f5c2c7",
                        "indicator": "●"
                    }
                }
                
                config = status_config.get(overall_status, status_config["medium"])
                
                # Status Card Utama
                st.markdown(f"""
                <div style="background: {config['bg']}; padding: 24px; border-radius: 12px; border: 1px solid {config['border']}; margin-bottom: 24px;">
                    <div style="display: flex; align-items: flex-start; gap: 16px;">
                        <div style="color: {config['color']}; font-size: 2.5em; line-height: 1;">{config['indicator']}</div>
                        <div style="flex: 1;">
                            <h3 style="margin: 0 0 8px 0; color: #212529; font-weight: 600; font-size: 1.25em;">{config['title']}</h3>
                            <p style="margin: 0; color: #495057; line-height: 1.5;">{config['desc']}</p>
                        </div>
                        <div style="text-align: right; color: #6c757d; font-size: 0.875em;">
                            Berdasarkan<br><strong style="color: #212529;">{sample_count} prediksi</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Evidently Statistics Summary
                st.markdown("#### Ringkasan Analisis Statistik")
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    drift_status_text = "Ya" if dataset_drift else "Tidak"
                    drift_color = "#dc3545" if dataset_drift else "#198754"
                    st.markdown(f"""
                    <div style="background: white; padding: 16px; border-radius: 10px; border: 1px solid #e9ecef; text-align: center;">
                        <div style="color: #6c757d; font-size: 0.8em; text-transform: uppercase;">Dataset Drift</div>
                        <div style="color: {drift_color}; font-size: 1.5em; font-weight: 700;">{drift_status_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stat_col2:
                    st.markdown(f"""
                    <div style="background: white; padding: 16px; border-radius: 10px; border: 1px solid #e9ecef; text-align: center;">
                        <div style="color: #6c757d; font-size: 0.8em; text-transform: uppercase;">Fitur Bermasalah</div>
                        <div style="color: #212529; font-size: 1.5em; font-weight: 700;">{drifted_count} / {total_features}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stat_col3:
                    share_pct = drift_share * 100 if drift_share <= 1 else drift_share
                    st.markdown(f"""
                    <div style="background: white; padding: 16px; border-radius: 10px; border: 1px solid #e9ecef; text-align: center;">
                        <div style="color: #6c757d; font-size: 0.8em; text-transform: uppercase;">Tingkat Drift</div>
                        <div style="color: #212529; font-size: 1.5em; font-weight: 700;">{share_pct:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stat_col4:
                    st.markdown(f"""
                    <div style="background: white; padding: 16px; border-radius: 10px; border: 1px solid #e9ecef; text-align: center;">
                        <div style="color: #6c757d; font-size: 0.8em; text-transform: uppercase;">Data Referensi</div>
                        <div style="color: #212529; font-size: 1.5em; font-weight: 700;">{reference_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("")
                
                # Detail per Karakteristik
                features_drift = drift_data.get("features", {})
                if features_drift:
                    
                    feature_info = {
                        "LB": {"name": "Luas Bangunan", "unit": "m²", "desc": "Ukuran bangunan rumah"},
                        "LT": {"name": "Luas Tanah", "unit": "m²", "desc": "Ukuran total tanah"},
                        "KT": {"name": "Kamar Tidur", "unit": "kamar", "desc": "Jumlah kamar tidur"},
                        "KM": {"name": "Kamar Mandi", "unit": "kamar", "desc": "Jumlah kamar mandi"},
                        "GRS": {"name": "Garasi", "unit": "mobil", "desc": "Kapasitas garasi"}
                    }
                    
                    # =====================================================
                    # SECTION: GRAFIK VISUALISASI
                    # =====================================================
                    st.markdown("#### Visualisasi Perbandingan Data")
                    st.markdown("*Grafik perbandingan antara data historis dengan data saat ini*")
                    
                    # Prepare data for charts
                    chart_features = []
                    chart_historical = []
                    chart_current = []
                    chart_changes = []
                    chart_colors = []
                    chart_p_values = []
                    chart_drift_detected = []
                    
                    for feature, data in features_drift.items():
                        info = feature_info.get(feature, {"name": feature, "unit": ""})
                        ref_val = data.get("reference_mean", 0)
                        # Support both 'current_mean' (Evidently) and 'recent_mean' (fallback)
                        cur_val = data.get("current_mean", data.get("recent_mean", 0))
                        severity = data.get("severity", "low")
                        p_value = data.get("p_value", None)
                        is_drifted = data.get("drift_detected", False)
                        
                        chart_features.append(info["name"])
                        chart_historical.append(ref_val)
                        chart_current.append(cur_val)
                        chart_p_values.append(p_value)
                        chart_drift_detected.append(is_drifted)
                        
                        if ref_val > 0:
                            change = ((cur_val - ref_val) / ref_val) * 100
                        else:
                            change = 0
                        chart_changes.append(change)
                        
                        # Color based on severity
                        if severity == "low":
                            chart_colors.append("#198754")
                        elif severity == "medium":
                            chart_colors.append("#fd7e14")
                        else:
                            chart_colors.append("#dc3545")
                    
                    # Tab untuk berbagai grafik
                    tab_bar, tab_radar, tab_change, tab_stats = st.tabs(["Perbandingan Nilai", "Pola Karakteristik", "Tingkat Perubahan", "Detail Statistik"])
                    
                    with tab_bar:
                        # Grouped Bar Chart
                        fig_bar = go.Figure()
                        
                        fig_bar.add_trace(go.Bar(
                            name='Data Historis',
                            x=chart_features,
                            y=chart_historical,
                            marker_color='#6c757d',
                            text=[f'{v:.0f}' for v in chart_historical],
                            textposition='outside'
                        ))
                        
                        fig_bar.add_trace(go.Bar(
                            name='Data Saat Ini',
                            x=chart_features,
                            y=chart_current,
                            marker_color='#0d6efd',
                            text=[f'{v:.0f}' for v in chart_current],
                            textposition='outside'
                        ))
                        
                        fig_bar.update_layout(
                            barmode='group',
                            title=dict(
                                text='Perbandingan Rata-rata Nilai per Karakteristik',
                                font=dict(size=16, color='#212529'),
                                x=0
                            ),
                            xaxis_title='Karakteristik Rumah',
                            yaxis_title='Nilai Rata-rata',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(family="Segoe UI, sans-serif", size=12, color="#495057"),
                            height=400,
                            margin=dict(t=80, b=60)
                        )
                        
                        fig_bar.update_xaxes(showgrid=False, showline=True, linecolor='#dee2e6')
                        fig_bar.update_yaxes(showgrid=True, gridcolor='#f8f9fa', showline=True, linecolor='#dee2e6')
                        
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        st.caption("Grafik ini menunjukkan perbandingan nilai rata-rata antara data yang dipelajari sistem (historis) dengan data yang masuk saat ini. Semakin mirip kedua batang, semakin konsisten datanya.")
                    
                    with tab_radar:
                        # Normalize data for radar chart (0-100 scale)
                        max_vals = [max(h, c) if max(h, c) > 0 else 1 for h, c in zip(chart_historical, chart_current)]
                        normalized_historical = [(h / m) * 100 for h, m in zip(chart_historical, max_vals)]
                        normalized_current = [(c / m) * 100 for c, m in zip(chart_current, max_vals)]
                        
                        fig_radar = go.Figure()
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=normalized_historical + [normalized_historical[0]],
                            theta=chart_features + [chart_features[0]],
                            fill='toself',
                            fillcolor='rgba(108, 117, 125, 0.2)',
                            line=dict(color='#6c757d', width=2),
                            name='Data Historis'
                        ))
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=normalized_current + [normalized_current[0]],
                            theta=chart_features + [chart_features[0]],
                            fill='toself',
                            fillcolor='rgba(13, 110, 253, 0.2)',
                            line=dict(color='#0d6efd', width=2),
                            name='Data Saat Ini'
                        ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100],
                                    showticklabels=False,
                                    gridcolor='#e9ecef'
                                ),
                                angularaxis=dict(
                                    gridcolor='#e9ecef'
                                ),
                                bgcolor='white'
                            ),
                            title=dict(
                                text='Pola Karakteristik Rumah',
                                font=dict(size=16, color='#212529'),
                                x=0
                            ),
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.15,
                                xanchor="center",
                                x=0.5
                            ),
                            paper_bgcolor='white',
                            font=dict(family="Segoe UI, sans-serif", size=12, color="#495057"),
                            height=450,
                            margin=dict(t=80, b=80)
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)
                        
                        st.caption("Grafik radar menunjukkan pola keseluruhan karakteristik rumah. Area biru (saat ini) yang mendekati area abu-abu (historis) menandakan data yang konsisten.")
                    
                    with tab_change:
                        # Horizontal Bar Chart for Change Percentage
                        df_change = pd.DataFrame({
                            'Karakteristik': chart_features,
                            'Perubahan (%)': chart_changes,
                            'Warna': chart_colors
                        })
                        df_change = df_change.sort_values('Perubahan (%)', ascending=True)
                        
                        fig_change = go.Figure()
                        
                        fig_change.add_trace(go.Bar(
                            y=df_change['Karakteristik'],
                            x=df_change['Perubahan (%)'],
                            orientation='h',
                            marker=dict(
                                color=df_change['Warna'],
                                line=dict(color='white', width=1)
                            ),
                            text=[f'{v:+.1f}%' for v in df_change['Perubahan (%)']],
                            textposition='outside',
                            textfont=dict(size=12, color='#495057')
                        ))
                        
                        # Add reference line at 0
                        fig_change.add_vline(x=0, line_width=2, line_color="#dee2e6")
                        
                        fig_change.update_layout(
                            title=dict(
                                text='Tingkat Perubahan dari Data Historis',
                                font=dict(size=16, color='#212529'),
                                x=0
                            ),
                            xaxis_title='Perubahan (%)',
                            yaxis_title='',
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(family="Segoe UI, sans-serif", size=12, color="#495057"),
                            height=350,
                            margin=dict(l=20, r=80, t=60, b=40),
                            showlegend=False
                        )
                        
                        fig_change.update_xaxes(showgrid=True, gridcolor='#f8f9fa', showline=True, linecolor='#dee2e6', zeroline=True, zerolinecolor='#adb5bd')
                        fig_change.update_yaxes(showgrid=False, showline=False)
                        
                        st.plotly_chart(fig_change, use_container_width=True)
                        
                        # Color legend
                        col_leg1, col_leg2, col_leg3 = st.columns(3)
                        with col_leg1:
                            st.markdown('<div style="display:flex;align-items:center;gap:8px;"><span style="width:12px;height:12px;background:#198754;border-radius:50%;display:inline-block;"></span><span style="color:#495057;font-size:0.9em;">Sesuai (perubahan kecil)</span></div>', unsafe_allow_html=True)
                        with col_leg2:
                            st.markdown('<div style="display:flex;align-items:center;gap:8px;"><span style="width:12px;height:12px;background:#fd7e14;border-radius:50%;display:inline-block;"></span><span style="color:#495057;font-size:0.9em;">Berubah (perlu perhatian)</span></div>', unsafe_allow_html=True)
                        with col_leg3:
                            st.markdown('<div style="display:flex;align-items:center;gap:8px;"><span style="width:12px;height:12px;background:#dc3545;border-radius:50%;display:inline-block;"></span><span style="color:#495057;font-size:0.9em;">Berbeda (perubahan besar)</span></div>', unsafe_allow_html=True)
                        
                        st.caption("Grafik ini menunjukkan seberapa besar perubahan nilai dari data historis. Nilai positif (+) berarti lebih tinggi, negatif (-) berarti lebih rendah.")
                    
                    with tab_stats:
                        # Evidently Statistical Details
                        st.markdown("##### Hasil Uji Statistik per Fitur")
                        st.markdown("*Analisis menggunakan metode statistik untuk mendeteksi perubahan distribusi data*")
                        
                        # Create detailed statistics table
                        stats_data = []
                        for feature, data in features_drift.items():
                            info = feature_info.get(feature, {"name": feature})
                            p_val = data.get("p_value", None)
                            stattest = data.get("stattest", "N/A")
                            drift_detected = data.get("drift_detected", False)
                            drift_score = data.get("drift_score", 0)
                            ref_mean = data.get("reference_mean", 0)
                            cur_mean = data.get("current_mean", data.get("recent_mean", 0))
                            ref_std = data.get("reference_std", 0)
                            cur_std = data.get("current_std", 0)
                            
                            stats_data.append({
                                "Karakteristik": info["name"],
                                "Metode Uji": stattest.replace("_", " ").title() if stattest else "N/A",
                                "P-Value": f"{p_val:.4f}" if p_val is not None else "N/A",
                                "Drift Score": f"{drift_score:.4f}",
                                "Drift Terdeteksi": "✓ Ya" if drift_detected else "✗ Tidak",
                                "Mean Referensi": f"{ref_mean:.1f}",
                                "Mean Saat Ini": f"{cur_mean:.1f}",
                                "Std Referensi": f"{ref_std:.1f}",
                                "Std Saat Ini": f"{cur_std:.1f}"
                            })
                        
                        df_stats = pd.DataFrame(stats_data)
                        st.dataframe(df_stats, use_container_width=True, hide_index=True)
                        
                        # Explanation
                        with st.expander("ℹ️ Cara Membaca Tabel Ini"):
                            st.markdown("""
                            **Kolom-kolom pada tabel:**
                            
                            | Kolom | Penjelasan |
                            |-------|------------|
                            | **Metode Uji** | Teknik statistik yang digunakan untuk mendeteksi perubahan (contoh: Kolmogorov-Smirnov, Chi-Square) |
                            | **P-Value** | Nilai probabilitas. Semakin kecil (< 0.05), semakin signifikan perubahannya |
                            | **Drift Score** | Skor numerik yang menunjukkan tingkat perubahan |
                            | **Drift Terdeteksi** | Apakah perubahan signifikan ditemukan |
                            | **Mean** | Nilai rata-rata (Referensi = data training, Saat Ini = data prediksi terbaru) |
                            | **Std** | Standar deviasi (tingkat sebaran data) |
                            
                            **Interpretasi:**
                            - P-Value < 0.05 → Perubahan signifikan secara statistik
                            - P-Value ≥ 0.05 → Tidak ada bukti perubahan signifikan
                            - Drift Score tinggi → Perbedaan distribusi yang besar
                            """)
                    
                    # =====================================================
                    # SECTION: DETAIL CARDS
                    # =====================================================
                    st.markdown("---")
                    st.markdown("#### Detail Karakteristik Rumah")
                    st.markdown("*Informasi lengkap per karakteristik*")
                    
                    # Buat 2 kolom
                    col1, col2 = st.columns(2)
                    items = list(features_drift.items())
                    
                    for idx, (feature, data) in enumerate(items):
                        info = feature_info.get(feature, {"name": feature, "unit": "", "desc": ""})
                        severity = data.get("severity", "low")
                        ref_val = data.get("reference_mean", 0)
                        cur_val = data.get("current_mean", data.get("recent_mean", 0))
                        drift_detected = data.get("drift_detected", False)
                        
                        # Hitung persentase perubahan
                        if ref_val > 0:
                            change_pct = ((cur_val - ref_val) / ref_val) * 100
                        else:
                            change_pct = 0
                        
                        # Warna berdasarkan severity
                        if severity == "low":
                            status_color = "#198754"
                            status_bg = "#d1e7dd"
                            status_text = "Sesuai"
                        elif severity == "medium":
                            status_color = "#fd7e14"
                            status_bg = "#fff3cd"
                            status_text = "Berubah"
                        else:
                            status_color = "#dc3545"
                            status_bg = "#f8d7da"
                            status_text = "Berbeda"
                        
                        # Arrow untuk perubahan
                        if change_pct > 5:
                            arrow = "↑"
                            arrow_color = "#198754"
                        elif change_pct < -5:
                            arrow = "↓"
                            arrow_color = "#dc3545"
                        else:
                            arrow = "→"
                            arrow_color = "#6c757d"
                        
                        card_html = f"""
                        <div style="background: white; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <span style="font-weight: 600; color: #212529;">{info['name']}</span>
                                <span style="background: {status_bg}; color: {status_color}; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 500;">{status_text}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                                <div>
                                    <div style="color: #6c757d; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px;">Rata-rata saat ini</div>
                                    <div style="font-size: 1.5em; font-weight: 600; color: #212529;">{cur_val:.0f} <span style="font-size: 0.5em; font-weight: 400; color: #6c757d;">{info['unit']}</span></div>
                                </div>
                                <div style="text-align: center; padding: 0 15px;">
                                    <span style="color: {arrow_color}; font-size: 1.5em;">{arrow}</span>
                                    <div style="color: {arrow_color}; font-size: 0.8em; font-weight: 500;">{change_pct:+.0f}%</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="color: #6c757d; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px;">Data historis</div>
                                    <div style="font-size: 1.1em; color: #6c757d;">{ref_val:.0f} {info['unit']}</div>
                                </div>
                            </div>
                        </div>
                        """
                        
                        if idx % 2 == 0:
                            with col1:
                                st.markdown(card_html, unsafe_allow_html=True)
                        else:
                            with col2:
                                st.markdown(card_html, unsafe_allow_html=True)
                    
                    # Penjelasan untuk pengguna awam
                    st.markdown("---")
                    
                    with st.expander("ℹ️ Apa maksudnya ini?", expanded=False):
                        st.markdown("""
                        **Mengapa pemantauan ini penting?**
                        
                        Sistem prediksi harga rumah ini belajar dari data rumah-rumah yang sudah terjual sebelumnya. 
                        Ketika karakteristik rumah yang diprediksi (seperti luas, jumlah kamar) sangat berbeda dari 
                        data yang dipelajari, hasil prediksi mungkin kurang akurat.
                        
                        **Cara membaca informasi di atas:**
                        
                        | Status | Artinya |
                        |--------|---------|
                        | **Sesuai** | Karakteristik rumah masih dalam rentang normal |
                        | **Berubah** | Ada sedikit perbedaan, tapi masih bisa diterima |
                        | **Berbeda** | Perbedaan cukup besar, perlu diperhatikan |
                        
                        **Contoh:**
                        Jika rata-rata luas bangunan yang diprediksi adalah 200 m², tapi data historis menunjukkan 
                        rata-rata 100 m², artinya pengguna sekarang lebih banyak mencari rumah yang lebih besar 
                        dari biasanya.
                        """)
                    
                    # Rekomendasi (hanya jika ada masalah)
                    if overall_status in ["medium", "high"]:
                        st.markdown("#### Yang Perlu Diketahui")
                        
                        if overall_status == "medium":
                            st.markdown("""
                            <div style="background: #fff3cd; padding: 20px; border-radius: 10px; border-left: 4px solid #ffc107;">
                                <p style="margin: 0; color: #664d03; line-height: 1.6;">
                                    <strong>Pola penggunaan mulai bergeser.</strong><br>
                                    Pengguna saat ini mencari rumah dengan karakteristik yang sedikit berbeda dari biasanya. 
                                    Ini bisa disebabkan oleh perubahan tren pasar atau musim. Hasil prediksi masih dapat 
                                    digunakan sebagai referensi, namun disarankan untuk membandingkan dengan harga pasar aktual.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background: #f8d7da; padding: 20px; border-radius: 10px; border-left: 4px solid #dc3545;">
                                <p style="margin: 0; color: #842029; line-height: 1.6;">
                                    <strong>Karakteristik rumah cukup berbeda dari data yang dipelajari.</strong><br>
                                    Pengguna saat ini banyak mencari rumah dengan spesifikasi di luar rentang data historis. 
                                    Hasil prediksi sebaiknya digunakan sebagai gambaran awal saja. Untuk estimasi yang lebih 
                                    akurat, disarankan berkonsultasi dengan agen properti atau melihat harga pasaran terkini.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Footer info
                    st.markdown("")
                    st.caption(f"Data dianalisis dari {sample_count} prediksi terakhir • Diperbarui secara otomatis")
                    
        else:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 30px; border-radius: 12px; text-align: center; border: 1px solid #e9ecef;">
                <div style="font-size: 2em; margin-bottom: 10px;">🔌</div>
                <p style="color: #6c757d; margin: 0;">Tidak dapat terhubung ke sistem pemantauan.<br>Pastikan layanan berjalan dengan baik.</p>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.error("Gagal mengambil data metrik dari API. Pastikan API berjalan.")

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

if st.session_state.role is None:
    show_login_page()
elif st.session_state.role == 'user':
    show_user_page()
elif st.session_state.role == 'admin':
    show_admin_page()
