import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from training import CSRNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(pth_path):
    model = CSRNet(load_weights=False).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def run_inference(model, img_path, output_path="output_heatmap.png"):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)

    density_map = pred.squeeze().cpu().numpy()
    est_count = density_map.sum()

    heatmap = density_map / (density_map.max() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, img.size)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(output_path, overlay)

    print(f"[INFO] Image: {img_path}")
    print(f"[INFO] Estimated Count: {est_count:.2f}")
    print(f"[INFO] Heatmap saved at: {output_path}")
    return est_count, output_path

if __name__ == "__main__":
    model_path = "./csrnet_train.pth"
    test_image = "./test.jpg"  
    output_file = "./output_heatmap.png"

    model = load_model(model_path)
    run_inference(model, test_image, output_file)
