import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# -----------------------------
# Seiten-Config (modern)
# -----------------------------
st.set_page_config(
    page_title="🐶🐱 Hunde oder Katze?",
    page_icon="🐾",
    layout="centered"
)

# -----------------------------
# Modell laden (einmalig)
# -----------------------------
@st.cache_resource
def load_my_model():
    model = load_model("keras_Model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
    return model, class_names

model, class_names = load_my_model()

# -----------------------------
# Titel
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🐾 Hunde oder Katze?</h1>
    <p style='text-align: center; font-size:18px;'>
    Lade ein Bild hoch und finde es heraus!
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Datei-Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📷 Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Wenn Bild hochgeladen wurde
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

    class_name = class_names[index].strip().split(" ", 1)[1]

    st.divider()

    # Ergebnis modern anzeigen
    st.markdown("## 🔎 Ergebnis")

    if "Hund" in class_name:
        st.success(f"🐶 **{class_name}** erkannt!")
    else:
        st.info(f"🐱 **{class_name}** erkannt!")

    st.progress(confidence_score)

    st.markdown(
        f"### 🎯 Sicherheit: **{confidence_score*100:.2f}%**"
    )
