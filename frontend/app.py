import streamlit as st, requests, base64
from PIL import Image
import io

st.set_page_config(page_title="Pothole Detection App", layout="wide")
st.title("🕳️ Pothole Detection")

uploaded = st.file_uploader("Upload road image", type=["jpg","jpeg","png"])

if uploaded:
    files = {"file": uploaded.getvalue()}
    res = requests.post("http://localhost:8000/detect/", files={"file": ("img.jpg", uploaded.getvalue(), "image/jpeg")})
    data = res.json()

    st.subheader("Detections")
    st.write(data["detections"])

    img_bytes = base64.b64decode(data["image_b64"])
    img = Image.open(io.BytesIO(img_bytes))
    st.image(img, caption="Detected Potholes")
