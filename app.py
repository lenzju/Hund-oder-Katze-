import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
from tensorflow.keras.models import load_model

# -----------------------------
# Seiten-Einstellungen
# -----------------------------
st.set_page_config(
    page_title="Hund oder Katze",
    page_icon="🐾",
    layout="centered"
)

# -----------------------------
# modell laden (Cloud-sicher)
# -----------------------------
@st.cache_resource
def load_my_model():
    base_path = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_path, "keras_model.h5")
    labels_path = os.path.join(base_path, "labels.txt")

    if not os.path.exists(model_path):
        st.error("❌ keras_model.h5 nicht gefunden!")
        st.stop()

    if not os.path.exists(labels_path):
        st.error("❌ labels.txt nicht gefunden!")
        st.stop()

    model = load_model(model_path, compile=False)

    with open(labels_path, "r") as f:
        class_names = f.readlines()

    return model, class_names


model, class_names = load_my_model()

# -----------------------------
# UI
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🐶🐱 Hund oder Katze?</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Lade ein Bild hoch und finde es heraus!</p>", unsafe_allow_html=True)

st.divider()

uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

# -----------------------------
# Bild verarbeitet
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    # Bild vorbereiten
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    confidence_score = float(prediction[0][index])

    class_name = class_names[index].strip()

    if " " in class_name:
        class_name = class_name.split(" ", 1)[1]

    st.divider()
    st.markdown("## 🔎 Ergebnis")

    if "Hund" in class_name:
        st.success(f"🐶 {class_name}")
    else:
        st.info(f"🐱 {class_name}")

    st.progress(confidence_score)
    st.markdown(f"### 🎯 Sicherheit: {confidence_score * 100:.2f}%")

