import os
import cv2
import streamlit as st
from PIL import Image
from inference import load_csrnet_model, get_count_and_heatmap

def send_alert_email(to_email, subject, body):
    print("[DUMMY EMAIL] Alert triggered!")
    print(f"To: {to_email}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")

@st.cache_resource
def get_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "csrnet_model", "csrnet_train.pth")
    model = load_csrnet_model(MODEL_PATH)
    return model

model = get_model()

st.title("👥 CSRNet Crowd Counting with Alerts (Dummy Email)")
st.write("Upload an image to estimate crowd count and visualize heatmap.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

CROWD_THRESHOLD = 15  

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    overlay, count = get_count_and_heatmap(model, img)

    st.image(overlay, caption=f"Estimated Count: {count}", use_column_width=True)
    st.success(f"Estimated Crowd Count: {count}")

    if count > CROWD_THRESHOLD:
        st.warning(f"Crowd exceeds threshold ({CROWD_THRESHOLD})! Triggering alert...")
        send_alert_email(
            to_email="dummyreceiver@example.com",
            subject="Crowd Alert",
            body=f"Estimated crowd count: {count} exceeds threshold of {CROWD_THRESHOLD}."
        )

