import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
import io

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo entrenado
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 clases
model.load_state_dict(torch.load('sarah_connor_model.pth'))
model.eval()

# Transformación de la imagen
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Definir la ruta para la clasificación de imágenes
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen desde la solicitud
    img = request.files['image'].read()
    img = Image.open(io.BytesIO(img))
    
    # Transformar la imagen
    img_tensor = transform(img).unsqueeze(0)
    
    # Realizar la predicción
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    
    # Determinar si es Sarah Connor
    if predicted.item() == 0:
        return jsonify({'prediction': 'Sarah Connor'})
    else:
        return jsonify({'prediction': 'Not Sarah Connor'})

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
