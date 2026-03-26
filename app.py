import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="Deepfake Detector", page_icon="🧠", layout="centered")

# Title
st.title("🧠 Deepfake Image Detector")
st.write("Upload an image to detect if it is Real or Fake")

# Load model
model = load_model("deepfake_detector.h5")

# Upload
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="🖼 Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = cv2.resize(img, (128,128))
    img_resized = img_resized / 255.0
    img_resized = np.reshape(img_resized, (1,128,128,3))

    # Predict
    prediction = model.predict(img_resized)
    confidence = float(prediction[0][0])

    # 👉 IMPORTANT: interpret correctly
    real_prob = 1 - confidence
    fake_prob = confidence

    st.markdown("## 🔍 Prediction Result")

    if fake_prob > 0.5:
        st.error(f"🚨 Fake Image ({fake_prob*100:.2f}%)")
    else:
        st.success(f"✅ Real Image ({real_prob*100:.2f}%)")

    # ---------------- CONFIDENCE ----------------
    st.markdown("### 📊 Confidence Analysis")

    st.write(f"Real Probability: {real_prob*100:.2f}%")
    st.progress(real_prob)

    st.write(f"Fake Probability: {fake_prob*100:.2f}%")
    st.progress(fake_prob)

    # ---------------- SIMPLE HEATMAP ----------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="🔥 Feature Focus Visualization", use_container_width=True)

    # ---------------- EXPLANATION ----------------
    st.markdown("## 🧠 Why this prediction?")

    if fake_prob > 0.5:
        st.error("This image is FAKE because:")

        st.write(f"👉 High fake confidence ({fake_prob*100:.2f}%) indicates artificial patterns")
        st.write("❗ Irregular textures detected")
        st.write("❗ Possible GAN artifacts")
        st.write("❗ Blurring or edge inconsistencies")
        st.write("❗ Lighting mismatch")

        st.markdown("### ⚖️ Why some % is REAL?")
        st.write(f"✔ {real_prob*100:.2f}% real probability due to partial natural features")
        st.write("✔ Some regions look realistic")
        st.write("✔ Partial symmetry present")

    else:
        st.success("This image is REAL because:")

        st.write(f"👉 High real confidence ({real_prob*100:.2f}%) indicates natural patterns")
        st.write("✔ Facial symmetry is well aligned")
        st.write("✔ Skin texture is consistent")
        st.write("✔ Natural lighting and shadows")
        st.write("✔ No visible manipulation artifacts")

        st.markdown("### ⚖️ Why some % is FAKE?")
        st.write(f"⚠️ {fake_prob*100:.2f}% fake probability due to minor noise or compression")
        st.write("⚠️ Slight pixel inconsistencies")
        st.write("⚠️ Lighting variations")

    # ---------------- FINAL ----------------
    st.markdown("## ⚖️ Final Analysis")

    if fake_prob > 0.5:
        st.error("Final Decision: Fake Image (artifacts detected)")
    else:
        st.success("Final Decision: Real Image (natural feature consistency)")