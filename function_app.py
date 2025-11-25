import os
import json
import pickle
from typing import Optional

import azure.functions as func
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector


# ========= Configuración global (se ejecuta al cold start) =========

# URL de Qdrant (configúrala en Application Settings en Azure: QDRANT_URL)
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "code_knowledge")

# Ruta del vectorizer entrenado (mismo que usaste en el notebook)
# Asegúrate de que el archivo exista en el root de la Function App
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "tfidf_vectorizer.pkl")

vectorizer: Optional[object] = None

try:
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print(f"[startup] Vectorizer cargado desde: {VECTORIZER_PATH}")
except Exception as e:
    print(f"[startup] Error al cargar el vectorizer desde '{VECTORIZER_PATH}': {e}")
    vectorizer = None

# Cliente de Qdrant global (reutilizable entre invocaciones)
client = QdrantClient(
    url=QDRANT_URL,
    prefer_grpc=False,
    timeout=30,
)


def search_tfidf(
    client: QdrantClient,
    collection_name: str,
    query: str,
    vectorizer,
    top_k: int = 5,
):
    """
    Hace una búsqueda TF-IDF usando Qdrant con sparse vectors.
    """
    # 1) TF-IDF del query
    q_vec = vectorizer.transform([query])  # scipy.sparse matrix
    q_coo = q_vec.tocoo()

    sparse = SparseVector(
        indices=q_coo.col.tolist(),
        values=q_coo.data.tolist(),
    )

    # 2) Buscar en Qdrant usando sparse vector
    results = client.search(
        collection_name=collection_name,
        query_vector=None,                # no dense vector
        query_sparse_vector=sparse,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return results


# ========= Definición de la Function App =========

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# =============================
# HEALTH ENDPOINTS
# =============================

@app.function_name(name="root_health")
@app.route(route="", methods=["GET"])
def root_health(req: func.HttpRequest) -> func.HttpResponse:
    """
    Endpoint raíz para verificar que la Function está viva.
    URL: /
    """
    return func.HttpResponse(
        json.dumps({
            "status": "ok",
            "service": "sauco-tfidf-function",
            "message": "Azure Function is running",
        }),
        mimetype="application/json",
        status_code=200
    )


@app.function_name(name="health_status")
@app.route(route="health", methods=["GET"])
def detailed_health(req: func.HttpRequest) -> func.HttpResponse:
    """
    Endpoint /health con verificación de dependencias.
    URL: /api/health
    """
    qdrant_status = "unknown"

    try:
        # Llamada simple para verificar conexión
        client.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"

    return func.HttpResponse(
        json.dumps({
            "status": "ok",
            "qdrant": qdrant_status,
            "qdrant_url": QDRANT_URL,
            "collection": COLLECTION,
            "vectorizer_loaded": vectorizer is not None
        }),
        mimetype="application/json",
        status_code=200
    )


# =============================
# SEARCH TF-IDF ENDPOINT
# =============================

@app.function_name(name="tfidf_search")
@app.route(route="search", methods=["POST", "GET"])
def tfidf_search(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP endpoint:
    - GET  /api/search?query=...&top_k=5
    - POST /api/search  { "query": "...", "top_k": 5 }
    """
    try:
        if vectorizer is None:
            return func.HttpResponse(
                json.dumps({"error": "Vectorizer not loaded. Check VECTORIZER_PATH and deployment files."}),
                status_code=500,
                mimetype="application/json",
            )

        if req.method == "GET":
            query = req.params.get("query")
            top_k = req.params.get("top_k")
        else:
            try:
                body = req.get_json()
            except ValueError:
                body = {}
            query = body.get("query")
            top_k = body.get("top_k")

        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'query' parameter"}),
                status_code=400,
                mimetype="application/json",
            )

        try:
            top_k = int(top_k) if top_k is not None else 5
        except (ValueError, TypeError):
            top_k = 5

        results = search_tfidf(
            client=client,
            collection_name=COLLECTION,
            query=query,
            vectorizer=vectorizer,
            top_k=top_k,
        )

        # Convertir los resultados a algo JSON-friendly
        out = []
        for r in results:
            out.append(
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload,  # aquí tendrás page, text, source, etc.
                }
            )

        return func.HttpResponse(
            json.dumps(
                {
                    "query": query,
                    "top_k": top_k,
                    "results": out,
                },
                ensure_ascii=False,
            ),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as e:
        # log con print simple; en Azure aparecerá en Application Insights / Log stream
        print(f"Error in tfidf_search: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json",
        )
