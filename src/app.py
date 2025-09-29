import os
import streamlit as st
from PIL import Image
from inference import load_csrnet_model, get_count_and_heatmap

@st.cache_resource
def get_model():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../csrnet_model/csrnet_train.pth")
    return load_csrnet_model(MODEL_PATH)

model = get_model()

st.title("👥 CSRNet Crowd Counting")
st.write("Upload an image to estimate crowd count and visualize heatmap.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    overlay, count = get_count_and_heatmap(model, img)
    st.image(overlay, caption=f"Estimated Count: {count}", use_column_width=True)
    st.success(f"Estimated Crowd Count: {count}")

