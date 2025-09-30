import os
import cv2
import streamlit as st
from PIL import Image
from inference import load_csrnet_model, get_count_and_heatmap
import smtplib
from email.message import EmailMessage

def send_alert_email(subject, body, to_email):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["To"] = to_email

    user = "sender@gmail.com"      
    password = "your_app_password_here"    
    msg["From"] = user
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
        server.quit()
        print(f"[INFO] Alert sent to {to_email}")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")

try:
    get_cache = st.cache_resource  
except AttributeError:
    get_cache = lambda func: st.cache(allow_output_mutation=True)  

@get_cache
def get_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "csrnet_model", "csrnet_train.pth")
    model = load_csrnet_model(MODEL_PATH)
    return model

model = get_model()

st.title("👥 Crowd Monitoring System")
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
            subject="Crowd Alert",
            body=f"Estimated crowd count: {count} exceeds threshold of {CROWD_THRESHOLD}.",
            to_email="receiver@gmail.com"  
        )
