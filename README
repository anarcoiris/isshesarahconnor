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
