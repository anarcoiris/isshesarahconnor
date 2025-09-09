# Usar una imagen base con PyTorch y CUDA
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el código de la aplicación y el modelo entrenado
COPY . /app

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Puerto que usaremos para la API
EXPOSE 8080

# Comando para ejecutar la API
CMD ["python", "app.py"]
