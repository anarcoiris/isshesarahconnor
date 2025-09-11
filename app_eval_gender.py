#!/usr/bin/env python3
"""
Flask app robusta para clasificación de imágenes con PyTorch.

Características:
 - carga de checkpoint flexible (state_dict o checkpoint con 'model_state', 'state_dict', etc.)
 - corrige prefijos 'module.' si el checkpoint fue guardado con DataParallel
 - mueve modelo y tensores al device correcto (cpu/cuda)
 - convierte cualquier imagen a RGB y maneja formatos raros (L, LA, RGBA, CMYK)
 - devuelve probabilidades y etiqueta en HTML y JSON
 - endpoint / (form) y /predict (POST form o multipart) y /predict_json (POST -> JSON)
 - endpoint /debug devuelve logits y probs (solo si debug=1 en query)
"""

from __future__ import annotations

import io
import os
import json
import logging
from typing import List, Optional, Dict

from PIL import Image, ImageFile
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template_string

# permitir cargar imágenes truncadas sin lanzar excepción PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serv")

# -------- Configuración --------
CHECKPOINT_PATH = "models/male_female.pth"   # cambia por tu checkpoint
LABELS_PATH = "labels.json"          # opcional: archivo JSON con lista de etiquetas en orden de índice
HOST = "0.0.0.0"
PORT = 8123
DEBUG_MODE = True

# -------- HTML simple --------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Classifier</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    .result { margin-top: 20px; padding: 10px; border: 1px solid #ddd; display:inline-block; }
    img.preview { max-width: 320px; margin-top: 10px; display:block; }
    table { border-collapse: collapse; margin-top:10px }
    td, th { padding:6px 10px; border:1px solid #ddd; }
  </style>
</head>
<body>
  <h2>Subir imagen</h2>
  <form action="/predict" method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <input type="submit" value="Subir y clasificar">
  </form>

  {% if error %}
    <div class="result"><b>Error:</b> {{ error }}</div>
  {% endif %}

  {% if prediction %}
    <div class="result">
      <b>Predicción:</b> {{ prediction }} <br/>
      <b>Probabilidad:</b> {{ probstr }} <br/>
      <img class="preview" src="data:image/png;base64,{{ preview_b64 }}" alt="preview" />
      <h4>Probabilidades por clase:</h4>
      <table>
        <tr><th>Clase</th><th>Probabilidad</th></tr>
        {% for lbl, p in probs.items() %}
          <tr><td>{{ lbl }}</td><td>{{ '{:.4f}'.format(p) }}</td></tr>
        {% endfor %}
      </table>
    </div>
  {% endif %}
</body>
</html>
"""

# -------- utilidades de carga --------
def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Si las keys empiezan por 'module.', las limpia (guardado por DataParallel)."""
    new_state = {}
    changed = False
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
            changed = True
        else:
            new_state[k] = v
    if changed:
        logger.info("Se ha detectado prefijo 'module.' en checkpoint: lo he eliminado para cargar.")
    return new_state

def load_checkpoint(path: str) -> Dict:
    """Carga de forma flexible: puede devolver dict con state_dict o un state_dict directamente."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint no encontrado en {path}")
    raw = torch.load(path, map_location="cpu")
    # si es checkpoint con claves conocidas:
    for key in ("model_state", "state_dict", "model_state_dict"):
        if isinstance(raw, dict) and key in raw:
            sd = raw[key]
            if isinstance(sd, dict):
                return strip_module_prefix(sd)
    # si raw ya parece un state_dict
    if isinstance(raw, dict) and any(k.startswith("layer") or k.startswith("conv") or k.startswith("fc") or k.startswith("module.") for k in raw.keys()):
        return strip_module_prefix(raw)
    raise RuntimeError("Formato de checkpoint no reconocido. Esperado state_dict o checkpoint con 'model_state'/'state_dict'.")

def load_labels(path: str, default: Optional[List[str]] = None) -> List[str]:
    if default is None:
        default = ["class0", "class1"]
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
            # si viene dict, tomar values/keys heurísticamente
            if isinstance(data, dict):
                # intentar ordenar por índice si es mapping idx->label
                try:
                    items = sorted(data.items(), key=lambda kv: int(kv[0]))
                    return [v for k,v in items]
                except Exception:
                    return list(data.values())
        except Exception as e:
            logger.warning("No se pudo leer labels.json: %s", e)
    return default

# -------- transform y modelo --------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def build_base_model(num_classes: int = 2, base: str = "resnet50", pretrained: bool = False) -> torch.nn.Module:
    """Construye arquitectura coincidente con la que entrenaste (ajusta si usas otra)."""
    if base == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif base == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError("base model no soportado")
    in_features = m.fc.in_features
    m.fc = torch.nn.Linear(in_features, num_classes)
    return m

# -------- App Flask --------
app = Flask(__name__)

# Cargar labels (si tienes un archivo labels.json en el mismo directorio, lo usará)
labels = load_labels(LABELS_PATH, default=["Mujer", "Hombre"])
logger.info("Etiquetas cargadas: %s", labels)

# Determinar device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device seleccionado: %s", device)

# Build model y cargar checkpoint
num_classes = len(labels)
model = build_base_model(num_classes=num_classes, base="resnet50", pretrained=False)
try:
    state = load_checkpoint(CHECKPOINT_PATH)
    model.load_state_dict(state)
    logger.info("Checkpoint cargado correctamente desde %s", CHECKPOINT_PATH)
except Exception as e:
    logger.warning("No se pudo cargar checkpoint como state_dict directamente: %s", e)
    # Intentar cargar como whole-file dict con key 'model_state' anidada:
    raw = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if isinstance(raw, dict) and 'model_state' in raw:
        sd = strip_module_prefix(raw['model_state'])
        model.load_state_dict(sd)
        logger.info("Checkpoint cargado usando raw['model_state']")
    else:
        # si todo falla, levantar excepción clara
        raise

model = model.to(device)
model.eval()

# -------- Helpers de inferencia --------
def prepare_image(pil_img: Image.Image) -> torch.Tensor:
    # Forzar RGB (maneja L, LA, RGBA, CMYK, etc.)
    if pil_img.mode != "RGB":
        try:
            pil_img = pil_img.convert("RGB")
        except Exception:
            # si no puede convertir, convertir a 'RGB' vía 'RGBA' fallback
            pil_img = pil_img.convert("RGBA").convert("RGB")
    t = transform(pil_img)
    return t.unsqueeze(0)  # batch dim

def softmax_probs(logits: torch.Tensor) -> List[float]:
    probs = F.softmax(logits, dim=-1).cpu().numpy().ravel().tolist()
    return probs

# -------- Rutas --------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict_html():
    """Recibe form multipart desde navegador y muestra HTML con resultado."""
    try:
        if "image" not in request.files:
            return render_template_string(HTML_PAGE, error="No se subió ninguna imagen")
        file = request.files["image"]
        content = file.read()
        if not content:
            return render_template_string(HTML_PAGE, error="Archivo vacío")
        pil = Image.open(io.BytesIO(content))
        tensor = prepare_image(pil)
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = softmax_probs(logits)
            top_idx = int(torch.argmax(logits, dim=1).item())
        pred_label = labels[top_idx] if top_idx < len(labels) else str(top_idx)
        probstr = f"{probs[top_idx]:.4f}"
        # preview base64 for embedding
        import base64
        buf = io.BytesIO()
        # generar preview reducido
        pil.thumbnail((320, 320))
        pil.save(buf, format="PNG")
        preview_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        probs_map = {labels[i] if i < len(labels) else str(i): probs[i] for i in range(len(probs))}
        return render_template_string(HTML_PAGE, prediction=pred_label, probstr=probstr, preview_b64=preview_b64, probs=probs_map)
    except Exception as e:
        logger.exception("Error en /predict:")
        return render_template_string(HTML_PAGE, error=str(e))

@app.route("/predict_json", methods=["POST"])
def predict_json():
    """Endpoint para automatización: POST multipart/form-data con campo 'image' o POST JSON con base64."""
    debug = request.args.get("debug", "0") == "1"
    try:
        # 1) soporte multipart upload
        if "image" in request.files:
            file = request.files["image"]
            content = file.read()
            pil = Image.open(io.BytesIO(content))
        else:
            # 2) soporte JSON { "image_b64": "..." }
            data = request.get_json(silent=True) or {}
            b64 = data.get("image_b64")
            if not b64:
                return jsonify({"error": "no image provided"}), 400
            import base64
            content = base64.b64decode(b64)
            pil = Image.open(io.BytesIO(content))
        tensor = prepare_image(pil).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = softmax_probs(logits)
            top_idx = int(torch.argmax(logits, dim=1).item())
        result = {
            "pred_index": top_idx,
            "pred_label": labels[top_idx] if top_idx < len(labels) else str(top_idx),
            "probabilities": { (labels[i] if i < len(labels) else str(i)) : float(probs[i]) for i in range(len(probs)) }
        }
        if debug:
            # incluir logits crudos para diagnóstico
            result["logits"] = logits.cpu().numpy().tolist()
        return jsonify(result)
    except Exception as e:
        logger.exception("Error en /predict_json")
        return jsonify({"error": str(e)}), 500

# -------- Run --------
if __name__ == "__main__":
    logger.info("Iniciando servidor en %s:%s", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=DEBUG_MODE)
