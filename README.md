# Integración de Azure Face API con `isshesarahconnor.py`
**Documentación técnica y how-to**

---

## Resumen ejecutivo
Azure Face API es un servicio gestionado para detección y reconocimiento facial.  
Esta documentación explica cómo usar Face API (`detect`, `verify`, `identify`, `find_similar`, `liveness`) y cómo integrarlo en el repositorio **isshesarahconnor** que unifica fetch, preprocess, train, export y servicio de inferencia.  

Incluye ejemplos de código listos para pegar y recomendaciones de seguridad y despliegue.

---

## Qué hace Azure Face API
- **Detección de rostros:** bounding boxes, landmarks, atributos faciales.
- **Verificación:** comparaciones one-to-one entre dos caras.
- **Identificación:** búsqueda one-to-many contra PersonGroup / LargePersonGroup o FaceList.
- **Find_similar:** buscar caras similares (usa representaciones internas).
- **Liveness (anti-spoofing):** detección de ataques de suplantación.

### Nota sobre embeddings
Face API gestiona internamente representaciones para verify/identify.  
Si tu objetivo es almacenamiento y uso directo de vectores (embeddings) para búsqueda no biométrica, considera usar las APIs de embeddings multimodales de Azure o extraer vectores localmente (ResNet, Facenet) y almacenarlos en una vector DB (Milvus, FAISS, Azure Cognitive Search con vector store).

---

## QuickStart: recurso Azure y autenticación
1. Crear recurso **Face** o un recurso multi-service en el portal de Azure.  
2. Obtener `endpoint` y `API key` desde el portal.  
3. Instalar SDK:
```bash
pip install azure-ai-vision-face azure-core azure-identity
```
En pruebas: usar `AzureKeyCredential`;  
en producción: `DefaultAzureCredential` o **Managed Identity**.

---

## Comandos REST de la Face API de Azure

### Resumen rápido
La Face API se expone típicamente en la forma:

```
https://<tu-recurso>.cognitiveservices.azure.com/face/v1.0
```

Las operaciones principales:
- `detect`
- `verify`
- `identify`
- `findsimilars`

Recursos de gestión: `personGroup`, `largePersonGroup`, `faceList`, `largeFaceList`.

**Autenticación:** `Ocp-Apim-Subscription-Key: <KEY>` o Azure AD.

### Operaciones principales
- **Detect** → `POST /face/v1.0/detect`  
- **Verify** → `POST /face/v1.0/verify`  
- **Identify** → `POST /face/v1.0/identify`  
- **Find Similar** → `POST /face/v1.0/findsimilars`

### Colecciones y gestión (persistencia)
- **LargePersonGroup** → crear/administrar grupos a gran escala.  
- **Person** dentro de un grupo → crear persona y añadir caras (`persistedFaceId`).  
- **FaceList / LargeFaceList** → listas de caras persistidas.  
- Operaciones típicas: `train`, `get person`, `delete persistedFace`, etc.

---

## Ejemplos con curl

```bash
# Detectar
curl -X POST "https://<endpoint>/face/v1.0/detect?returnFaceId=true" \
  -H "Ocp-Apim-Subscription-Key: <KEY>" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@imagen.jpg"
```

```bash
# Verify
curl -X POST "https://<endpoint>/face/v1.0/verify" \
  -H "Ocp-Apim-Subscription-Key: <KEY>" \
  -H "Content-Type: application/json" \
  -d '{"faceId1":"<uuid1>", "faceId2":"<uuid2>"}'
```

```bash
# Identify
curl -X POST "https://<endpoint>/face/v1.0/identify" \
  -H "Ocp-Apim-Subscription-Key: <KEY>" \
  -H "Content-Type: application/json" \
  -d '{"faceIds":["<faceId>"], "largePersonGroupId":"mi_grupo"}'
```

---

## Snippets how-to (Python)

### Detectar caras
```python
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel

endpoint = "https://<tu-subdominio>.cognitiveservices.azure.com/"
key = "<TU_API_KEY>"
face_client = FaceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

with open("img.jpg", "rb") as f:
    img_bytes = f.read()

results = face_client.detect(
    img_bytes,
    detection_model=FaceDetectionModel.DETECTION03,
    recognition_model=FaceRecognitionModel.RECOGNITION04,
    return_face_id=True,
    return_face_landmarks=True,
    face_id_time_to_live=120
)

for face in results:
    print(face.face_id, face.face_rectangle)
```

### Verificar dos caras
```python
res = face_client.verify_face_to_face(face_id1, face_id2)
print(res.is_identical, res.confidence)
```

### Registrar personas y entrenar
```python
from azure.ai.vision.face import FaceAdministrationClient
admin = FaceAdministrationClient(endpoint=endpoint, credential=AzureKeyCredential(key))
admin.create_large_person_group(large_person_group_id="my_group", name="Mi grupo")
admin.begin_train(large_person_group_id="my_group").result()
```

---

## Integración con `pipeline_all_in_one.py`

El script soporta múltiples backends locales para detección.  
A continuación se muestra un helper para Face API:

```python
def azure_detect_faces_bytes(img_bytes: bytes, endpoint: str, key: str, face_ttl:int=120):
    face_client = FaceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    result = face_client.detect(
        img_bytes,
        detection_model=FaceDetectionModel.DETECTION03,
        recognition_model=FaceRecognitionModel.RECOGNITION04,
        return_face_id=True,
        return_face_landmarks=False,
        face_id_time_to_live=face_ttl
    )
    boxes, face_ids = [], []
    for face in result:
        r = face.face_rectangle
        boxes.append((r.left, r.top, r.left + r.width, r.top + r.height))
        face_ids.append(face.face_id)
    return boxes, face_ids
```

---

## Comandos de uso

```bash
# Descargar imágenes
python pipeline_all_in_one.py --fetch --query "Ada Lovelace" --out raw/adalovelace
```

```bash
# Preparar dataset
python -c "from pipeline_all_in_one import internal_prepare_dataset; internal_prepare_dataset('crops','dataset_out',0.7,0.15,0.15)"
```

```bash
# Levantar servidor Flask
python -c "from pipeline_all_in_one import create_flask_app; app=create_flask_app('models/best_model.pth', labels=['a','b']); app.run(host='0.0.0.0', port=8080)"
```

---

## Recomendaciones
- No publiques claves en repositorios → usa variables de entorno o Managed Identity.  
- **Privacidad:** cumplir GDPR y normativas locales.  
- **Costes:** Face API se factura por transacción → optimiza tus llamadas.  
- **Producción:** containerizar, usar HTTPS y monitorizar errores/latencia.  

---

## Referencias
- [Azure Face API docs](https://learn.microsoft.com/azure/cognitive-services/face/)  
- [Azure ML docs](https://learn.microsoft.com/azure/machine-learning/)  
- [Azure Custom Vision docs](https://learn.microsoft.com/azure/cognitive-services/custom-vision-service/)
