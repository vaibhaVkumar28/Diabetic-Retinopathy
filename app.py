import os
# Force TensorFlow to run cleanly on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gradio as gr
import tensorflow as tf
import cv2
import numpy as np
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

# --- CUSTOM LOSS & PREPROCESSING ---
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        loss = -y_true * (tf.pow((1 - y_pred), gamma)) * tf.math.log(y_pred)
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fixed

def apply_clahe(img):
    img_uint8 = img.astype('uint8')
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return preprocess_input(final_img.astype(np.float32))

# --- LOAD MODEL ---
print("🔄 Loading Model...")
model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()})
print("✅ Model Loaded Successfully!")

# --- PREDICTION LOGIC ---
def predict_eye(image):
    if image is None: 
        return "No Image Uploaded", {}

    img_resized = cv2.resize(image, (300, 300))
    processed_img = apply_clahe(img_resized)
    input_batch = np.expand_dims(processed_img, axis=0)

    preds = model.predict(input_batch, verbose=0)[0]

    healthy_prob = float(preds[HEALTHY_INDEX])
    sick_prob = 1.0 - healthy_prob

    confidences = {CLASS_LABELS[i]: float(preds[i]) for i in range(5)}

    if sick_prob > OPTIMAL_THRESHOLD:
        sick_classes = preds.copy()
        sick_classes[HEALTHY_INDEX] = -1
        worst_stage_index = np.argmax(sick_classes)
        stage_name = CLASS_LABELS[worst_stage_index]
        diagnosis = f"⚠️ POSITIVE: {stage_name}"
    else:
        diagnosis = "✅ NEGATIVE: Healthy (No DR)"

    return diagnosis, confidences

# --- LAUNCH APP FOR RENDER ---
interface = gr.Interface(
    fn=predict_eye,
    inputs=gr.Image(label="Upload Retinal Scan"),
    outputs=[
        gr.Textbox(label="Final Diagnosis"),
        gr.Label(num_top_classes=5, label="Detailed Confidence Scores")
    ],
    title="Diabetic Retinopathy AI Diagnostician",
    description=f"Model checks for 5 stages of Diabetic Retinopathy.\nSick Threshold: {OPTIMAL_THRESHOLD}"
)

# Render dynamically assigns PORT via environment variable
port = int(os.environ.get("PORT", 10000))
interface.launch(server_name="0.0.0.0", server_port=port)
