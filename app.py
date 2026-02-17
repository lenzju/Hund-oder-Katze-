import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Einstellungen
st.set_page_config(page_title="Hund vs. Katze", layout="centered")
st.title("🐶 Hund oder 🐱 Katze?")
st.write("Lade ein Bild hoch und das Modell sagt dir, was es ist.")

# Numpy Format
np.set_printoptions(suppress=True)

# Modell laden (Caching für Performance)
@st.cache_resource
def load_my_model():
    return load_model("keras_model.h5", compile=False)

model = load_my_model()

# Labels laden
class_names = open("labels.txt", "r").readlines()

# Bild Upload
uploaded_file = st.file_uploader(
    "📸 Bild hochladen (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

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
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    st.subheader("🔍 Ergebnis")
    st.write(f"**Klasse:** {class_name}")
    st.write(f"**Confidence:** {confidence_score:.2%}")
