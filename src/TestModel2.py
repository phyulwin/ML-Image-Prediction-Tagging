#=========================================================
# Model - Hugging Face Furniture-Dataset 
#=========================================================
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO
import json
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class mapping
JSON_PATH = r"src/model/class_to_idx.json"
MODEL_PATH = r"src\model\furniture_resnet18.pth"

with open(JSON_PATH, 'r') as f:
    class_to_idx = json.load(f)

# Create inverse mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Model initialization and loading
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing for inference
img_size = 128
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def display_image(img):
    plt.figure(figsize=(3,3))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def load_image(img_path_or_url):
    """Load image from URL or local path."""
    if img_path_or_url.startswith(('http://', 'https://')):
        response = requests.get(img_path_or_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(img_path_or_url).convert("RGB")
    return img

def _HF_Funiture_model(image_path_or_url, topk=3):
    try:
        img = load_image(image_path_or_url)
        # display_image(img)
        img_tensor = transform(img).unsqueeze(0).to(device)
    
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_idxs = probabilities.topk(topk)
    
        top_probs = top_probs.cpu().numpy().flatten()
        top_idxs = top_idxs.cpu().numpy().flatten()
        top_classes = [idx_to_class[idx] for idx in top_idxs]
    
        print("Top predictions:")
        for cls, prob in zip(top_classes, top_probs):
            print(f"{cls}: {prob:.4f}")
        
        top_predictions = {cls: prob for cls, prob in zip(top_classes, top_probs)}
        return top_predictions
    except Exception as e:
        print(e)

# try:
#     _HF_Funiture_model("https://t3.ftcdn.net/jpg/05/28/57/64/360_F_528576447_j08koWfGyhXMweJzDz6qzx6yYBBKJSnM.jpg")
#     _HF_Funiture_model("https://st4.depositphotos.com/1023934/37752/i/450/depositphotos_377527168-stock-photo-interior-design-modern-living-room.jpg")
#     _HF_Funiture_model("https://t3.ftcdn.net/jpg/05/28/57/64/360_F_528576447_j08koWfGyhXMweJzDz6qzx6yYBBKJSnM.jpg")
# except Exception as e:
#     print(e)

# Top predictions:
# living room: 0.7981
# sectional: 0.0642
# chair: 0.0436

# Top predictions:
# living room: 0.9567
# storage: 0.0175
# chair: 0.0117

# Top predictions:
# living room: 0.7981
# sectional: 0.0642
# chair: 0.0436