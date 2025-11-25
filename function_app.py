import os
import json
import pickle
from typing import List

import azure.functions as func
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector


# ========= Configuración global (se ejecuta en cold start) =========

# URL de Qdrant (ponlo en Application Settings en Azure: QDRANT_URL)
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "code_knowledge")

# Cargar el vectorizer entrenado (mismo que usaste en el notebook)
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "vectorizer.pkl")

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

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


# ========= Definición de la Function HTTP =========

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.function_name(name="tfidf_search")
@app.route(route="search", methods=["POST", "GET"])
def tfidf_search(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP endpoint:
    - GET  /api/search?query=...&top_k=5
    - POST /api/search  { "query": "...", "top_k": 5 }
    """
    try:
        if req.method == "GET":
            query = req.params.get("query")
            top_k = req.params.get("top_k")
        else:
            body = req.get_json()
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
        except ValueError:
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
        # log con print simple; en Azure aparecerá en Application Insights
        print(f"Error in tfidf_search: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json",
        )
