import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run CPU-only mode

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Diabetic Retinopathy AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI elements
st.markdown("""
<style>
    /* Card containers */
    .metric-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #374151;
        margin-bottom: 15px;
    }
    .badge-positive {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-negative {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. MODEL & CONSTANTS ---
MODEL_PATH = 'best_hybrid_phase2_final.keras'
HEALTHY_INDEX = 2

CLASS_LABELS = {
    0: "Mild DR",
    1: "Moderate DR",
    2: "No_DR (Healthy)",
    3: "Proliferative DR",
    4: "Severe DR"
}

CLASS_DESCRIPTIONS = {
    "No_DR (Healthy)": "No signs of retinopathy detected.",
    "Mild DR": "Microaneurysms present; early vascular changes.",
    "Moderate DR": "Vessels showing blockage; hemorrhages visible.",
    "Severe DR": "Widespread vessel blockage; high risk of progression.",
    "Proliferative DR": "Advanced stage; growth of abnormal new blood vessels."
}

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        loss = -y_true * (tf.pow((1 - y_pred), gamma)) * tf.math.log(y_pred)
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fixed

@st.cache_resource
def load_dr_model():
    return load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()})

# Load Model
try:
    model = load_dr_model()
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/ophthalmology.png", width=70)
    st.title("Settings & Info")
    
    st.markdown("---")
    st.subheader("Model Configuration")
    optimal_threshold = st.slider(
        "Decision Threshold (Sick Probability)", 
        min_value=0.10, 
        max_value=0.80, 
        value=0.39, 
        step=0.01,
        help="Custom sensitivity threshold tuned for high clinical recall."
    )
    
    st.markdown("---")
    st.subheader("Clinical Guidance")
    st.caption("This tool uses CLAHE contrast adjustment combined with deep convolutional feature extraction to detect early microaneurysms and lesions.")

# --- 4. MAIN DASHBOARD UI ---
st.title("👁️ Diabetic Retinopathy AI Diagnostician")
st.markdown("Upload a fundus retinal scan to perform automated multi-stage evaluation.")

# File Uploader
uploaded_file = st.file_uploader("Choose a retinal fundus image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process Image
    raw_img = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(raw_img)
    img_resized = cv2.resize(img_np, (300, 300))
    
    # CLAHE Preprocessing
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    processed_img = preprocess_input(clahe_img.astype(np.float32))
    
    # Display Images side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Scan")
        st.image(raw_img, use_container_width=True)
        
    with col2:
        st.subheader("🔍 CLAHE Enhanced Scan")
        st.image(clahe_img, use_container_width=True, caption="Contrast adjusted for micro-vascular inspection")
        
    st.markdown("---")
    
    # Run Inference
    with st.spinner("Analyzing retinal structures..."):
        input_batch = np.expand_dims(processed_img, axis=0)
        preds = model.predict(input_batch, verbose=0)[0]
        
    healthy_prob = float(preds[HEALTHY_INDEX])
    sick_prob = 1.0 - healthy_prob
    
    # Diagnostic Decision Logic
    if sick_prob > optimal_threshold:
        sick_classes = preds.copy()
        sick_classes[HEALTHY_INDEX] = -1
        worst_stage_index = np.argmax(sick_classes)
        diagnosis_title = CLASS_LABELS[worst_stage_index]
        confidence = float(preds[worst_stage_index]) * 100
        is_positive = True
    else:
        diagnosis_title = "No_DR (Healthy)"
        confidence = healthy_prob * 100
        is_positive = False

    # Diagnostic Summary Box
    st.subheader("Diagnostic Report")
    res_col1, res_col2, res_col3 = st.columns([2, 1, 1])
    
    with res_col1:
        if is_positive:
            st.error(f"### ⚠️ POSITIVE: {diagnosis_title}")
        else:
            st.success(f"### ✅ NEGATIVE: Healthy")
        st.write(f"**Clinical Summary:** {CLASS_DESCRIPTIONS[diagnosis_title]}")

    with res_col2:
        st.metric(label="Primary Stage Confidence", value=f"{confidence:.2f}%")
        
    with res_col3:
        st.metric(label="Overall Risk Index", value=f"{(sick_prob * 100):.1f}%")

    # Visual Breakdown Tabs
    st.markdown("### Detailed Probability Analysis")
    tab1, tab2 = st.tabs(["📊 Confidence Distribution", "📋 Raw Class Probabilities"])
    
    chart_data = {CLASS_LABELS[i]: float(preds[i]) for i in range(5)}
    
    with tab1:
        st.bar_chart(chart_data)
        
    with tab2:
        formatted_probs = [{"Stage": CLASS_LABELS[i], "Probability Score": f"{float(preds[i]):.4f}"} for i in range(5)]
        st.table(formatted_probs)

else:
    # Empty State Info Container
    st.info("👆 Please upload a retinal fundus image above to generate a diagnostic report.")
    
    with st.expander("ℹ️ How to use this diagnostician"):
        st.write("""
        1. Select a fundus image (JPEG/PNG format).
        2. The pipeline automatically resizes the image to 300x300 and applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        3. The EfficientNet/ResNet model evaluates structural features and calculates classification probabilities across 5 stages of Diabetic Retinopathy.
        """)
