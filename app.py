import os
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import CSRNet

MODEL_DIR = "csrnet_model"
MODEL_PATH = os.path.join(MODEL_DIR, "csrnet_small.pth")

# Make sure folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSRNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

st.title("Crowd Counting App")
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_t = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        density_map = model(img_t)
    count = density_map.sum().item()
    st.success(f"Estimated Crowd Count: {round(count)}")

    if st.checkbox("Show density map heatmap"):
        density_np = density_map.squeeze().cpu().numpy()
        fig, ax = plt.subplots()
        ax.imshow(np.array(img))
        ax.imshow(density_np, cmap='jet', alpha=0.5)
        ax.axis('off')
        st.pyplot(fig)
