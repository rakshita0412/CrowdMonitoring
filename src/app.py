# import os
# import cv2
# import streamlit as st
# from PIL import Image
# from inference import load_csrnet_model, get_count_and_heatmap
# import smtplib
# from email.message import EmailMessage

# def send_alert_email(subject, body, to_email):
#     msg = EmailMessage()
#     msg.set_content(body)
#     msg["Subject"] = subject
#     msg["To"] = to_email

#     user = "monitoringcrowd@gmail.com"      
#     password = "nbis rmjo ocgb agsp"    
#     msg["From"] = user
    
#     try:
#         server = smtplib.SMTP("smtp.gmail.com", 587)
#         server.starttls()
#         server.login(user, password)
#         server.send_message(msg)
#         server.quit()
#         print(f"[INFO] Alert sent to {to_email}")
#     except Exception as e:
#         print(f"[ERROR] Failed to send email: {e}")

# try:
#     get_cache = st.cache_resource  
# except AttributeError:
#     get_cache = lambda func: st.cache(allow_output_mutation=True)  

# @get_cache
# def get_model():
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     MODEL_PATH = os.path.join(BASE_DIR, "csrnet_model", "csrnet_train.pth")
#     model = load_csrnet_model(MODEL_PATH)
#     return model

# model = get_model()

# st.title("👥 Crowd Monitoring System")
# st.write("Upload an image to estimate crowd count and visualize heatmap.")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
# CROWD_THRESHOLD = 15  

# if uploaded_file is not None:
#     img = Image.open(uploaded_file).convert("RGB")
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     overlay, count = get_count_and_heatmap(model, img)

#     st.image(overlay, caption=f"Estimated Count: {count}", use_column_width=True)
#     st.success(f"Estimated Crowd Count: {count}")

#     if count > CROWD_THRESHOLD:
#         st.warning(f"Crowd exceeds threshold ({CROWD_THRESHOLD})! Triggering alert...")
#         send_alert_email(
#             subject="Crowd Alert",
#             body=f"Estimated crowd count: {count} exceeds threshold of {CROWD_THRESHOLD}.",
#             to_email="rakshitavipperla@gmail.com"  
#         )


import os
import io
import cv2
import streamlit as st
from PIL import Image
from inference import load_csrnet_model, get_count_and_heatmap
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
import matplotlib.pyplot as plt

def send_alert_email(subject, body_html, to_email, image_pil=None):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["To"] = to_email
    msg["From"] = "monitoringcrowd@gmail.com"
    msg.set_content("This is a HTML email. Please view in an HTML capable client.")
    
    if image_pil:
        image_cid = make_msgid(domain='xyz.com')
        msg.add_alternative(body_html.format(image_cid=image_cid[1:-1]), subtype='html')
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format='PNG')
        msg.get_payload()[1].add_related(img_byte_arr.getvalue(),
                                         maintype='image', subtype='png',
                                         cid=image_cid)
    else:
        msg.add_alternative(body_html, subtype='html')

    user = "monitoringcrowd@gmail.com"
    password = "nbis rmjo ocgb agsp"  
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
        server.quit()
        print(f"[INFO] Alert sent to {to_email}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False

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

st.title("👥 CSRNet Crowd Counting with Alerts")
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
        st.warning(f"Crowd exceeds threshold ({CROWD_THRESHOLD})!")

        fig, ax = plt.subplots()
        ax.bar(["Threshold", "Estimated"], [CROWD_THRESHOLD, count], color=["red", "blue"])
        ax.set_ylabel("Crowd Count")
        ax.set_title("Crowd Alert Summary")
        plot_img = Image.fromarray(cv2.cvtColor(cv2.imread(uploaded_file.name), cv2.COLOR_BGR2RGB))


        html_content = f"""
        <html>
        <body>
            <h2 style="color:red;">🚨 Crowd Alert Notification</h2>
            <p>Estimated crowd count: <b>{count}</b></p>
            <p>Threshold: <b>{CROWD_THRESHOLD}</b></p>
            <p>Status: <b style='color:red;'>Crowd exceeds threshold!</b></p>
            <h3>Uploaded Image & Heatmap:</h3>
            <img src="cid:{{image_cid}}" width="600">
            <h3>Comparison Plot:</h3>
            <img src="cid:{{image_cid}}" width="600">
            <p>Please take necessary action.</p>
        </body>
        </html>
        """

        success = send_alert_email(
            subject="🚨 Crowd Alert Notification",
            body_html=html_content,
            to_email="recipient@gmail.com",
            image_pil=overlay
        )

        if success:
            st.success("✅ Alert email sent with visual summary!")
        else:
            st.error("❌ Failed to send alert email. Check logs.")

