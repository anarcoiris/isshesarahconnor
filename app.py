import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
import io

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo entrenado
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 clases
model.load_state_dict(torch.load('hugh_laurie.pth', map_location=torch.device('cpu')))
model.eval()

# Transformación de la imagen
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Página HTML sencilla
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sarah Connor Classifier</title>
</head>
<body>
    <h2>Subir una imagen para clasificar</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <input type="submit" value="Subir y clasificar">
    </form>
    {% if prediction %}
        <h3>Resultado: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

# Ruta principal (formulario)
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)

# Ruta de predicción
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se subió ninguna imagen"}), 400

    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    # 🔑 Forzar siempre a RGB (aunque venga en L, LA, CMYK, RGBA...)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Transformar la imagen
    img_tensor = transform(img).unsqueeze(0)

    # Realizar la predicción
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    # Resultado
    if predicted.item() == 0:
        prediction = "Hugh Laurie"
    else:
        prediction = "not Hugh Laurie"

    return render_template_string(HTML_PAGE, prediction=prediction)

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8123, debug=True)
