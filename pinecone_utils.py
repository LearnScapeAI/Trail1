import logging
logger = logging.getLogger(__name__)

def generate_embedding(text, model):
    return model.encode(text, show_progress_bar=False).tolist()

def query_knowledge_base(index, query, model, top_k=5):
    embedding = generate_embedding(query, model)
    response = index.query(vector=embedding, top_k=top_k, include_metadata=True, namespace="default")
    matches = response.get("matches", [])
    if not matches:
        logger.warning("No matching documents in Pinecone for query: '%s'", query)
    return matches