import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run cleanly on CPU

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- CONFIGURATION ---
MODEL_PATH = 'best_hybrid_phase2_final.keras'
OPTIMAL_THRESHOLD = 0.39
HEALTHY_INDEX = 2

CLASS_LABELS = {
    0: "Mild DR",
    1: "Moderate DR",
    2: "No_DR (Healthy)",
    3: "Proliferate DR",
    4: "Severe DR"
}

# --- CUSTOM LOSS FUNCTION ---
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        loss = -y_true * (tf.pow((1 - y_pred), gamma)) * tf.math.log(y_pred)
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fixed

# Cache model loading so it doesn't reload on every button click
@st.cache_resource
def load_dr_model():
    return load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()})

# --- UI HEADER ---
st.title("Diabetic Retinopathy AI Diagnostician")
st.write(f"Model evaluates 5 stages of Diabetic Retinopathy with a classification threshold of {OPTIMAL_THRESHOLD}.")

# --- MODEL LOADING ---
with st.spinner("Loading model..."):
    model = load_dr_model()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload Retinal Scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Retinal Scan', use_container_width=True)
    
    # Preprocessing
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, (300, 300))
    
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    processed_img = preprocess_input(final_img.astype(np.float32))
    
    # Inference
    input_batch = np.expand_dims(processed_img, axis=0)
    preds = model.predict(input_batch, verbose=0)[0]
    
    healthy_prob = float(preds[HEALTHY_INDEX])
    sick_prob = 1.0 - healthy_prob
    
    # Render Output
    st.subheader("Diagnosis Results")
    if sick_prob > OPTIMAL_THRESHOLD:
        sick_classes = preds.copy()
        sick_classes[HEALTHY_INDEX] = -1
        worst_stage_index = np.argmax(sick_classes)
        stage_name = CLASS_LABELS[worst_stage_index]
        st.error(f"⚠️ POSITIVE: {stage_name}")
    else:
        st.success("✅ NEGATIVE: Healthy (No DR)")
        
    st.subheader("Confidence Scores")
    chart_data = {CLASS_LABELS[i]: float(preds[i]) for i in range(5)}
    st.bar_chart(chart_data)
