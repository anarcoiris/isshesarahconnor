# app_features.py
import io
import json
from typing import Dict, List
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify

# deteccion de colores 
import cv2
import numpy as np
from sklearn.cluster import KMeans

# face detector
from facenet_pytorch import MTCNN

app = Flask(__name__)

# --- configuraciones ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=DEVICE)

# Etiquetas (ejemplo)
HAIR_COLORS = ["black", "brown", "blonde", "red", "gray", "other"]  # multiclass
EYE_COLORS  = ["brown", "blue", "green", "other"]  # multiclass
ACCESSORIES = ["glasses", "hat", "earrings"]  # multi-label

# Construir modelo: backbone + heads
class MultiAttrModel(nn.Module):
    def __init__(self, n_hair=len(HAIR_COLORS), n_eye=len(EYE_COLORS), n_acc=len(ACCESSORIES)):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        # quitar fc
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # heads
        self.hair_head = nn.Linear(in_features, n_hair)        # logits para hair (Softmax/CrossEntropy)
        self.eye_head  = nn.Linear(in_features, n_eye)         # logits para eye
        self.acc_head  = nn.Linear(in_features, n_acc)         # logits para accesorios (sigmoid)
    def forward(self, x):
        feat = self.backbone(x)
        return {
            "hair": self.hair_head(feat),
            "eye":  self.eye_head(feat),
            "acc":  self.acc_head(feat)
        }

model = MultiAttrModel().to(DEVICE)
# Carga de pesos (debes entrenarlo previamente)
model.load_state_dict(torch.load("model_attrs.pth", map_location=DEVICE))
model.eval()

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_attributes(cropped_face: Image.Image) -> Dict:
    x = transform(cropped_face).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
    # hair, eye: apply softmax; acc: sigmoid
    hair_probs = torch.softmax(out["hair"], dim=1).cpu().numpy()[0]
    eye_probs  = torch.softmax(out["eye"], dim=1).cpu().numpy()[0]
    acc_probs  = torch.sigmoid(out["acc"]).cpu().numpy()[0]

    hair_pred = [{"label": HAIR_COLORS[i], "prob": float(hair_probs[i])} for i in range(len(HAIR_COLORS))]
    eye_pred  = [{"label": EYE_COLORS[i], "prob": float(eye_probs[i])} for i in range(len(EYE_COLORS))]
    acc_pred  = [{"label": ACCESSORIES[i], "prob": float(acc_probs[i])} for i in range(len(ACCESSORIES))]

    return {"hair": hair_pred, "eye": eye_pred, "accessories": acc_pred}

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    img_bytes = request.files["image"].read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Detect faces
    boxes, probs = mtcnn.detect(img)

    results = []
    if boxes is None:
        return jsonify({"faces": []})

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(b) for b in box]
        face_crop = img.crop((x1, y1, x2, y2))
        attrs = predict_attributes(face_crop)
        results.append({
            "box": [x1, y1, x2, y2],
            "confidence_face": float(probs[i]) if probs is not None else None,
            "attributes": attrs
        })

    return jsonify({"faces": results})

def dominant_color_pil(pil_img, k=3):
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    counts = np.bincount(kmeans.labels_)
    center = kmeans.cluster_centers_[np.argmax(counts)]
    return center  # HSV center; map to nearest color label

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
