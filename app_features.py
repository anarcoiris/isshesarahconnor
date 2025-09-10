# app_features.py (actualizado)
import io
import os
import json
from typing import Dict, List
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify

# color detection
import cv2
from sklearn.cluster import KMeans

# face detector
from facenet_pytorch import MTCNN

# --- etiquetas (coincidir con train) ---
HAIR_COLORS = ["black", "brown", "blonde", "red", "gray", "other"]
EYE_COLORS  = ["brown", "blue", "green", "other"]
ACCESSORIES = ["glasses", "hat", "earrings"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=DEVICE)

# ---------------- Model definition (same arquitectura que en train) ----------------
class MultiAttrModel(nn.Module):
    def __init__(self, n_hair=len(HAIR_COLORS), n_eye=len(EYE_COLORS), n_acc=len(ACCESSORIES)):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.hair_head = nn.Linear(in_features, n_hair)
        self.eye_head  = nn.Linear(in_features, n_eye)
        self.acc_head  = nn.Linear(in_features, n_acc)

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "hair": self.hair_head(feat),
            "eye":  self.eye_head(feat),
            "acc":  self.acc_head(feat)
        }

MODEL_PATH = "model_attrs.pth"
model = MultiAttrModel().to(DEVICE)
model_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        model_loaded = True
        print(f"[INFO] Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load model_attrs.pth: {e}. App will respond but model predictions disabled.")
else:
    print(f"[WARN] {MODEL_PATH} not found. App will run but model predictions disabled; fallback color heuristics available.")

# transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- utility: dominant color (HSV) ----------------
def dominant_color_pil(pil_img, k=3):
    img = np.array(pil_img)
    if img.size == 0:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    pixels = img.reshape(-1, 3)
    # remove near-black/near-white pixels to focus on color (optional)
    vals = pixels.copy()
    # KMeans on HSV
    kmeans = KMeans(n_clusters=min(k, len(vals)), random_state=0).fit(vals)
    counts = np.bincount(kmeans.labels_)
    center = kmeans.cluster_centers_[np.argmax(counts)]
    return center  # HSV

def hsv_to_hair_label(hsv):
    # hsv: [H, S, V], H in [0,180] (OpenCV) or [0,360] if scaled; scikit-learn KMeans returns same scale as input
    h, s, v = hsv
    # heuristics (approx)
    if s < 30 and v > 160:
        return "gray"  # washed-out -> gray/white
    if v < 50:
        return "black"
    # Hue ranges (OpenCV H range 0-180): convert if necessary
    # assume H in 0-180
    if h < 10 or h > 160:
        # red-ish or very low hue -> map to red/black
        if v > 80 and s > 50:
            return "red"
        return "brown"
    if 10 <= h < 30:
        return "brown"
    if 30 <= h < 50:
        return "blonde"
    if 50 <= h < 100:
        return "blonde"
    return "other"

def hsv_to_eye_label(hsv):
    h, s, v = hsv
    # simple heuristics:
    if v < 40:
        return "brown"
    if 80 <= h <= 140 and s > 30:
        return "blue"  # approximate
    if 40 <= h < 80 and s > 30:
        return "green"
    return "brown"

# ---------------- prediction wrapper ----------------
def predict_attributes(cropped_face: Image.Image):
    """
    Returns dict with model probs (if available) and heuristics fallback.
    """
    result = {
        "hair": [],
        "eye": [],
        "accessories": []
    }

    # 1) model prediction if loaded
    if model_loaded:
        x = transform(cropped_face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)
        hair_probs = torch.softmax(out["hair"], dim=1).cpu().numpy()[0]
        eye_probs  = torch.softmax(out["eye"], dim=1).cpu().numpy()[0]
        acc_probs  = torch.sigmoid(out["acc"]).cpu().numpy()[0]

        result["hair"] = [{"label": HAIR_COLORS[i], "prob": float(hair_probs[i])} for i in range(len(HAIR_COLORS))]
        result["eye"]  = [{"label": EYE_COLORS[i], "prob": float(eye_probs[i])} for i in range(len(EYE_COLORS))]
        result["accessories"] = [{"label": ACCESSORIES[i], "prob": float(acc_probs[i])} for i in range(len(ACCESSORIES))]

        # If top model confidence low, we'll compute heuristic and append as fallback
        top_hair_conf = max([p["prob"] for p in result["hair"]]) if result["hair"] else 0.0
        top_eye_conf  = max([p["prob"] for p in result["eye"]]) if result["eye"] else 0.0
    else:
        top_hair_conf = 0.0
        top_eye_conf = 0.0

    # 2) heuristic: dominant color of upper region (hair) and eye region approximate
    # We'll use entire face crop as a coarse fallback — it's simple and not perfect.
    hsv = dominant_color_pil(cropped_face, k=3)
    if hsv is not None:
        hair_label = hsv_to_hair_label(hsv)
        eye_label  = hsv_to_eye_label(hsv)
        # Append heuristic only if model low confidence
        if top_hair_conf < 0.6:
            # include as a fallback with a lower prob
            result["hair"].append({"label": hair_label, "prob": float(0.5)})
        if top_eye_conf < 0.6:
            result["eye"].append({"label": eye_label, "prob": float(0.5)})

    # Normalize result lists: sort by prob desc
    for k in ("hair", "eye", "accessories"):
        if result[k]:
            result[k] = sorted(result[k], key=lambda x: x["prob"], reverse=True)
    return result

# ---------------- Flask app ----------------
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    img_bytes = request.files["image"].read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

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

if __name__ == "__main__":
    # enable simple debug logs
    print("[INFO] Starting app on 0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080)
