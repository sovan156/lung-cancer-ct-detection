# app.py
# Main web application - Run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import io
import time
import joblib
from datetime import datetime
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from preprocessing   import preprocess_single_image
from predict         import load_model, load_model_info, predict_ct_scan
from gradcam         import run_gradcam
from report_generator import generate_pdf_report, check_reportlab

# ─────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
.main { background: #0d1117; }

.hero {
    background: linear-gradient(135deg,#0d1b2a,#1a237e,#0d47a1);
    border-radius: 20px;
    padding: 35px;
    text-align: center;
    margin-bottom: 25px;
    border: 1px solid rgba(33,150,243,0.3);
    box-shadow: 0 8px 32px rgba(13,71,161,0.4);
}
.hero h1 { color:white; font-size:2.5em; margin:0; }
.hero p  { color:#90caf9; margin-top:8px; font-size:1.05em; }
.hero .badge {
    background: rgba(33,150,243,0.2);
    border: 1px solid rgba(33,150,243,0.4);
    color: #64b5f6;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 0.85em;
    display: inline-block;
    margin-top: 10px;
}

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 20px;
    margin: 10px 0;
}

.result-malignant {
    background: linear-gradient(135deg,#1a0000,#2d0000);
    border: 2px solid #f44336;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 30px rgba(244,67,54,0.3);
    margin: 15px 0;
}
.result-benign {
    background: linear-gradient(135deg,#1a1000,#2d1a00);
    border: 2px solid #ff9800;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 30px rgba(255,152,0,0.3);
    margin: 15px 0;
}
.result-normal {
    background: linear-gradient(135deg,#001a00,#002d00);
    border: 2px solid #4caf50;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 30px rgba(76,175,80,0.3);
    margin: 15px 0;
}

.info-box {
    background: rgba(33,150,243,0.1);
    border-left: 4px solid #2196f3;
    border-radius: 8px;
    padding: 14px;
    margin: 8px 0;
    color: #e3f2fd;
}
.warn-box {
    background: rgba(255,152,0,0.1);
    border-left: 4px solid #ff9800;
    border-radius: 8px;
    padding: 14px;
    margin: 8px 0;
    color: #fff3e0;
}

.stButton > button {
    background: linear-gradient(135deg,#1565c0,#0288d1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-size: 1em !important;
    font-weight: 600 !important;
    width: 100% !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a0a1a,#0d1b2a);
    border-right: 1px solid rgba(33,150,243,0.2);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL (once, cached)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_everything():
    model      = load_model('model/best_model.h5')
    model_info = load_model_info('model/model_info.pkl')
    return model, model_info


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────

if 'history' not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────────────────────────
# CHART FUNCTIONS
# ─────────────────────────────────────────────────────────────

def gauge_chart(confidence, predicted_class):
    """Speedometer gauge showing confidence"""
    colors = {
        'malignant': '#f44336',
        'benign':    '#ff9800',
        'normal':    '#4caf50'
    }
    color = colors.get(predicted_class, '#2196f3')

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "AI Confidence", 'font': {'color': 'white', 'size': 16}},
        number={'suffix': "%", 'font': {'color': color, 'size': 30}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'gray'},
            'bar':  {'color': color},
            'steps': [
                {'range': [0,  33], 'color': 'rgba(76,175,80,0.1)'},
                {'range': [33, 66], 'color': 'rgba(255,152,0,0.1)'},
                {'range': [66,100], 'color': 'rgba(244,67,54,0.1)'}
            ],
            'threshold': {
                'line':      {'color': color, 'width': 4},
                'thickness': 0.8,
                'value':     confidence
            }
        }
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        margin=dict(l=30, r=30, t=60, b=20)
    )
    return fig


def probability_chart(probabilities):
    """Bar chart showing probability for each class"""
    class_colors = {
        'normal':    '#4caf50',
        'benign':    '#ff9800',
        'malignant': '#f44336'
    }

    classes = list(probabilities.keys())
    values  = list(probabilities.values())
    cols    = [class_colors.get(c, '#2196f3') for c in classes]

    fig = go.Figure(data=[go.Bar(
        x=[c.upper() for c in classes],
        y=values,
        marker_color=cols,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        textfont={'color': 'white', 'size': 13}
    )])

    fig.update_layout(
        title={'text': 'Class Probabilities', 'x': 0.5,
               'font': {'color': 'white', 'size': 14}},
        yaxis=dict(range=[0, 120], gridcolor='rgba(255,255,255,0.1)',
                   tickfont={'color': 'gray'}),
        xaxis=dict(tickfont={'color': 'white', 'size': 12}),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=280,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def donut_chart(probabilities):
    """Donut/pie chart for risk distribution"""
    class_colors = {
        'normal':    '#4caf50',
        'benign':    '#ff9800',
        'malignant': '#f44336'
    }

    labels = [c.upper() for c in probabilities.keys()]
    values = list(probabilities.values())
    cols   = [class_colors.get(c, '#2196f3')
              for c in probabilities.keys()]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.6,
        marker=dict(colors=cols,
                    line=dict(color='#0d1b2a', width=2)),
        textfont={'color': 'white', 'size': 12}
    )])

    fig.update_layout(
        title={'text': 'Risk Distribution', 'x': 0.5,
               'font': {'color': 'white', 'size': 14}},
        paper_bgcolor='rgba(0,0,0,0)',
        legend={'font': {'color': 'white'},
                'bgcolor': 'rgba(0,0,0,0)'},
        height=280,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():

    # ── Hero Banner ──────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🫁 AI Lung Cancer Detection</h1>
        <p>Deep Learning CT Scan Analysis | Transfer Learning | Grad-CAM</p>
        <div class="badge">
            🎓 B.Tech Final Year AI/ML Project | MobileNetV2 + Grad-CAM
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Model ────────────────────────────────────────────
    with st.spinner("Loading AI model..."):
        model, model_info = load_everything()

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:15px 0; color:white;">
            <div style="font-size:3em;">🫁</div>
            <div style="font-weight:700; font-size:1.1em;">LungAI System</div>
            <div style="color:#90caf9; font-size:0.85em;">CT Scan Analysis v1.0</div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.15);">
        """, unsafe_allow_html=True)

        page = st.radio("", [
            "🔬 CT Scan Analysis",
            "📊 Model Dashboard",
            "📋 Prediction History",
            "ℹ️ About"
        ], label_visibility='collapsed')

        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);'>",
                    unsafe_allow_html=True)

        # Model status
        if model is not None:
            acc = model_info.get('accuracy', 0) * 100 if model_info else 0
            st.markdown(f"""
            <div style="background:rgba(76,175,80,0.1);
                        border:1px solid rgba(76,175,80,0.3);
                        border-radius:10px; padding:12px; text-align:center;">
                <div style="color:#4caf50; font-weight:700;">✅ Model Active</div>
                <div style="color:#90caf9; font-size:0.85em; margin-top:5px;">
                    MobileNetV2
                </div>
                <div style="color:white; font-size:0.9em; margin-top:4px;">
                    Accuracy: <b style="color:#4caf50;">{acc:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Model not found!\nRun: python train_model.py")

        st.markdown("""
        <hr style="border-color:rgba(255,255,255,0.15);">
        <div style="text-align:center; color:#546e7a; font-size:0.75em;">
            🎓 Final Year Project<br>AI & Machine Learning
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # PAGE 1: CT SCAN ANALYSIS
    # ══════════════════════════════════════════════════════════

    if page == "🔬 CT Scan Analysis":

        if model is None:
            st.error("""
            ❌ Model not found!

            Please train the model first:
            ```
            python train_model.py
            ```
            """)
            return

        st.markdown("## 🔬 Upload CT Scan for Analysis")

        st.markdown("""
        <div class="warn-box">
            ⚕️ <b>Disclaimer:</b> This AI system is for educational
            purposes only. NOT a substitute for professional medical
            diagnosis. Always consult a qualified doctor.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("### 📤 Upload CT Scan")

            # Optional patient info
            with st.expander("👤 Patient Details (Optional)"):
                p_name   = st.text_input("Patient Name",
                                         placeholder="Enter name")
                col_a, col_b = st.columns(2)
                with col_a:
                    p_age = st.number_input("Age", 0, 120, 0)
                with col_b:
                    p_gender = st.selectbox("Gender",
                                            ["Not Specified", "Male", "Female"])

            # Upload button
            uploaded = st.file_uploader(
                "Choose CT Scan Image",
                type=['jpg', 'jpeg', 'png'],
                label_visibility='collapsed'
            )

            if uploaded:
                img = Image.open(uploaded)
                st.image(img, caption=f"📄 {uploaded.name}",
                         use_column_width=True)
                st.markdown(f"""
                <div class="info-box" style="font-size:0.85em;">
                    📐 Size: {img.size[0]}×{img.size[1]}px &nbsp;|&nbsp;
                    🎨 Mode: {img.mode} &nbsp;|&nbsp;
                    📄 {uploaded.name}
                </div>
                """, unsafe_allow_html=True)

                predict_btn = st.button("🔬 ANALYZE WITH AI")
            else:
                st.markdown("""
                <div style="border:2px dashed rgba(33,150,243,0.5);
                            border-radius:14px; padding:50px;
                            text-align:center;
                            background:rgba(33,150,243,0.05);">
                    <div style="font-size:3em;">🫁</div>
                    <div style="color:#90caf9; margin-top:10px;
                                font-size:1.05em;">
                        Upload CT Scan Image Here
                    </div>
                    <div style="color:#546e7a; font-size:0.85em;
                                margin-top:5px;">
                        JPG, JPEG or PNG format
                    </div>
                </div>
                """, unsafe_allow_html=True)
                predict_btn = False

        with right:
            st.markdown("### 📋 Detection Classes")
            st.markdown("""
            <div style="display:flex; flex-direction:column; gap:10px;">
                <div style="background:rgba(76,175,80,0.1);
                            border:1px solid #4caf50;
                            border-radius:10px; padding:14px;">
                    <b style="color:#4caf50; font-size:1.05em;">
                        ✅ NORMAL
                    </b><br>
                    <small style="color:#a5d6a7;">
                        No cancer detected. Healthy lung tissue.
                        Continue regular checkups.
                    </small>
                </div>
                <div style="background:rgba(255,152,0,0.1);
                            border:1px solid #ff9800;
                            border-radius:10px; padding:14px;">
                    <b style="color:#ff9800; font-size:1.05em;">
                        ⚠️ BENIGN TUMOR
                    </b><br>
                    <small style="color:#ffcc80;">
                        Non-cancerous growth detected.
                        Medical monitoring required.
                    </small>
                </div>
                <div style="background:rgba(244,67,54,0.1);
                            border:1px solid #f44336;
                            border-radius:10px; padding:14px;">
                    <b style="color:#f44336; font-size:1.05em;">
                        🔴 MALIGNANT
                    </b><br>
                    <small style="color:#ef9a9a;">
                        Cancerous tissue detected.
                        Immediate medical attention needed.
                    </small>
                </div>
            </div>

            <div class="card" style="margin-top:15px;">
                <div style="color:white; font-weight:600;
                            margin-bottom:10px;">
                    ⚙️ How AI Analyzes CT Scans:
                </div>
                <div style="color:#90caf9; font-size:0.88em;
                            line-height:1.9;">
                    1️⃣ Image resized to 224×224 pixels<br>
                    2️⃣ Pixels normalized to 0-1 range<br>
                    3️⃣ MobileNetV2 extracts 1280 features<br>
                    4️⃣ Dense layers classify the features<br>
                    5️⃣ Softmax gives probability per class<br>
                    6️⃣ Grad-CAM shows attention regions
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── RUN PREDICTION ───────────────────────────────────
        if uploaded and predict_btn:

            st.markdown("---")
            st.markdown("## 🧠 AI Analysis Results")

            # Progress animation
            prog  = st.progress(0)
            status = st.empty()

            steps = [
                (25, "📥 Loading CT scan..."),
                (50, "🔧 Preprocessing image..."),
                (75, "🧠 Running deep learning model..."),
                (90, "🔥 Generating Grad-CAM heatmap..."),
                (100, "✅ Analysis complete!")
            ]

            for p, msg in steps:
                status.markdown(
                    f"<div style='color:#90caf9; text-align:center;'>"
                    f"{msg}</div>",
                    unsafe_allow_html=True
                )
                prog.progress(p)
                time.sleep(0.5)

            prog.empty()
            status.empty()

            try:
                # Run prediction
                uploaded.seek(0)
                result, img_array, original_img = predict_ct_scan(
                    model, model_info, uploaded
                )

                predicted_class = result['predicted_class']
                confidence      = result['confidence']
                probabilities   = result['probabilities']
                color           = result['color']

                # ── Result Box ────────────────────────────
                result_class = f"result-{predicted_class}"

                st.markdown(f"""
                <div class="{result_class}">
                    <div style="font-size:3.5em;">
                        {result['icon']}
                    </div>
                    <div style="font-size:1.8em; font-weight:700;
                                color:{color}; margin-top:10px;
                                letter-spacing:1px;">
                        {result['display_name']}
                    </div>
                    <div style="color:#ccc; margin-top:8px;
                                font-size:1.05em;">
                        {result['action']}
                    </div>
                    <div style="margin-top:15px;">
                        <span style="background:rgba(255,255,255,0.1);
                                     border:1px solid {color};
                                     border-radius:20px;
                                     padding:6px 18px;
                                     color:{color}; font-weight:700;
                                     font-size:1.05em;">
                            Confidence: {confidence:.1f}%
                        </span>
                        &nbsp;&nbsp;
                        <span style="background:rgba(255,255,255,0.1);
                                     border:1px solid {color};
                                     border-radius:20px;
                                     padding:6px 18px;
                                     color:{color}; font-weight:700;">
                            {result['risk']}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Charts ────────────────────────────────
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.plotly_chart(
                        gauge_chart(confidence, predicted_class),
                        use_container_width=True
                    )
                with c2:
                    st.plotly_chart(
                        probability_chart(probabilities),
                        use_container_width=True
                    )
                with c3:
                    st.plotly_chart(
                        donut_chart(probabilities),
                        use_container_width=True
                    )

                # ── Grad-CAM ──────────────────────────────
                st.markdown("---")
                st.markdown("### 🔥 Grad-CAM Heatmap Analysis")
                st.markdown("""
                <div class="info-box">
                    🔥 <b>Grad-CAM</b> reveals <b>which regions of the
                    CT scan</b> the AI focused on.
                    <b>Red/Yellow = High attention</b> (critical regions).
                    Blue = Low attention. This makes the AI decision
                    transparent and explainable.
                </div>
                """, unsafe_allow_html=True)

                with st.spinner("🔥 Generating heatmap..."):
                    uploaded.seek(0)
                    gradcam_buf, _ = run_gradcam(
                        model, img_array, original_img,
                        {
                            'class_idx':  result['class_idx'],
                            'class':      predicted_class,
                            'confidence': confidence
                        }
                    )

                st.image(gradcam_buf, use_column_width=True,
                         caption="Grad-CAM: Red areas = where AI looked most")

                # ── Metrics ───────────────────────────────
                st.markdown("---")
                st.markdown("### 📊 Detailed Scores")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Prediction",    predicted_class.upper())
                m2.metric("Confidence",    f"{confidence:.1f}%")
                m3.metric("Cancer Risk",
                          f"{probabilities.get('malignant', 0):.1f}%")
                m4.metric("Normal Prob.",
                          f"{probabilities.get('normal', 0):.1f}%")

                # ── Description ───────────────────────────
                st.markdown("### 💊 Medical Description")
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.05);
                            border:1px solid {color};
                            border-radius:12px; padding:20px;
                            color:#e0e0e0; line-height:1.8;
                            font-size:1.0em;">
                    {result['description']}
                </div>
                """, unsafe_allow_html=True)

                # ── Download Section ──────────────────────
                st.markdown("---")
                st.markdown("### 💾 Download Results")

                dl1, dl2, dl3 = st.columns(3)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Save CT scan
                with dl1:
                    os.makedirs('uploads', exist_ok=True)
                    uploaded.seek(0)
                    save_path = f"uploads/scan_{timestamp}.png"
                    with open(save_path, 'wb') as f:
                        f.write(uploaded.read())
                    st.success(f"✅ Scan saved!\n{save_path}")

                # Download heatmap
                with dl2:
                    st.download_button(
                        label="⬇️ Download Heatmap",
                        data=gradcam_buf,
                        file_name=f"heatmap_{timestamp}.png",
                        mime="image/png",
                        use_container_width=True
                    )

                # Download PDF report
                with dl3:
                    if check_reportlab():
                        pdf = generate_pdf_report(
                            patient_name=p_name or "Anonymous",
                            patient_age=p_age if p_age > 0 else None,
                            patient_gender=p_gender,
                            prediction_result=result
                        )
                        if pdf:
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf,
                                file_name=f"report_{timestamp}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                    else:
                        if st.button("📄 Install PDF support"):
                            st.code("pip install reportlab")

                # Save to history
                st.session_state.history.append({
                    'Time':       datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'Patient':    p_name or 'Anonymous',
                    'Age':        p_age if p_age > 0 else 'N/A',
                    'File':       uploaded.name,
                    'Result':     predicted_class.upper(),
                    'Confidence': f"{confidence:.1f}%",
                    'Risk':       result['risk'],
                    'Cancer%':    f"{probabilities.get('malignant', 0):.1f}%"
                })

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("""
                **If you see an error:**
                1. Make sure you ran: python train_model.py
                2. Make sure dataset images are in correct folders
                3. Try a different image
                4. Restart PyCharm and try again
                """)

    # ══════════════════════════════════════════════════════════
    # PAGE 2: MODEL DASHBOARD
    # ══════════════════════════════════════════════════════════

    elif page == "📊 Model Dashboard":

        st.markdown("## 📊 Model Performance Dashboard")

        if model_info is None:
            st.error("Train the model first: python train_model.py")
            return

        # Metrics
        acc = model_info.get('accuracy', 0) * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Accuracy",    f"{acc:.2f}%")
        c2.metric("🤖 Model",       "MobileNetV2")
        c3.metric("📦 Classes",     "3")
        c4.metric("🖼️ Input",       "224×224")

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 Training History")
            if os.path.exists('model/training_history.png'):
                st.image('model/training_history.png',
                         use_column_width=True)
            else:
                st.info("Run training to see charts")
        with col2:
            st.markdown("### 🔲 Confusion Matrix")
            if os.path.exists('model/confusion_matrix.png'):
                st.image('model/confusion_matrix.png',
                         use_column_width=True)
            else:
                st.info("Run training to see matrix")

        # Classification report
        st.markdown("### 📋 Classification Report")
        report = model_info.get('classification_report', 'Not available')
        st.markdown(f"""
        <div class="card">
            <pre style="color:#e0e0e0; font-family:monospace;
                        font-size:0.9em;">{report}</pre>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # PAGE 3: HISTORY
    # ══════════════════════════════════════════════════════════

    elif page == "📋 Prediction History":

        st.markdown("## 📋 Prediction History")

        if not st.session_state.history:
            st.info("📭 No predictions yet. Upload a CT scan first!")
            return

        df = pd.DataFrame(st.session_state.history)

        total  = len(df)
        mal    = len(df[df['Result'] == 'MALIGNANT'])
        ben    = len(df[df['Result'] == 'BENIGN'])
        norm   = len(df[df['Result'] == 'NORMAL'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scans",    total)
        c2.metric("🔴 Malignant",   mal)
        c3.metric("⚠️ Benign",      ben)
        c4.metric("✅ Normal",      norm)

        st.markdown("---")
        st.dataframe(df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download as CSV",
                df.to_csv(index=False),
                f"history_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

    # ══════════════════════════════════════════════════════════
    # PAGE 4: ABOUT
    # ══════════════════════════════════════════════════════════

    elif page == "ℹ️ About":

        st.markdown("## ℹ️ About This Project")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("""
            <div class="card">
                <h3 style="color:white;">🎯 Project Details</h3>
                <div style="color:#90caf9; line-height:1.9;">
                    <b style="color:white;">Title:</b>
                    AI-Based Lung Cancer Detection<br>
                    <b style="color:white;">Model:</b>
                    MobileNetV2 (Transfer Learning)<br>
                    <b style="color:white;">Framework:</b>
                    TensorFlow + Keras<br>
                    <b style="color:white;">Frontend:</b>
                    Streamlit + Plotly<br>
                    <b style="color:white;">Visualization:</b>
                    Grad-CAM Heatmaps<br>
                    <b style="color:white;">Classes:</b>
                    Normal | Benign | Malignant<br>
                    <b style="color:white;">Input Size:</b>
                    224 × 224 pixels
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="card">
                <h3 style="color:white;">🚀 Quick Commands</h3>
                <div style="color:#90caf9; line-height:2.2;
                            font-family:monospace; font-size:0.9em;">
                    Install libraries:<br>
                    <span style="color:#4fc3f7;">
                    pip install -r requirements.txt</span><br><br>
                    Train model:<br>
                    <span style="color:#4fc3f7;">
                    python train_model.py</span><br><br>
                    Run app:<br>
                    <span style="color:#4fc3f7;">
                    streamlit run app.py</span>
                </div>
                <div style="color:#ef9a9a; margin-top:15px;
                            font-size:0.88em;">
                    ⚕️ Educational purposes only.<br>
                    Not for clinical medical use.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#546e7a; font-size:0.8em;">
        🫁 AI Lung Cancer Detection |
        🎓 B.Tech Final Year AI/ML Project |
        MobileNetV2 + Grad-CAM + Streamlit
        <br>⚕️ For Educational Purposes Only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
