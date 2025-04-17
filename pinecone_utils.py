import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_embedding(text, model):
    """Generate an embedding vector for the given text."""
    return model.encode(text, show_progress_bar=False).tolist()


def make_id(text):
    """Create a deterministic SHA256-based ID from the text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks for finer-grained embedding.
    Yields text chunks of up to `chunk_size` tokens with `overlap` tokens overlap.
    """
    tokens = text.split()
    step = chunk_size - overlap
    for i in range(0, len(tokens), step):
        yield " ".join(tokens[i : i + chunk_size])


def upsert_if_unique(index, text, model,
                     metadata=None, threshold=0.9,
                     namespace="default"):  # pragma: no cover
    """
    Upsert a text chunk only if it's not too similar to existing vectors.

    Returns True if upserted, False if skipped as duplicate.
    """
    # 1) generate embedding
    emb = generate_embedding(text, model)
    # 2) query for nearest neighbor
    resp = index.query(vector=emb, top_k=1,
                       include_values=False,
                       namespace=namespace)
    matches = resp.get("matches", [])
    if matches and matches[0].score > threshold:
        logger.info(f"Duplicate detected (score={matches[0].score:.2f}), skipping upsert.")
        return False

    # 3) deterministic ID & metadata
    doc_id = make_id(text)
    meta = metadata.copy() if metadata else {}
    meta.setdefault("date_added", datetime.utcnow().isoformat())

    # 4) upsert vector
    index.upsert(vectors=[(doc_id, emb, meta)], namespace=namespace)
    logger.info(f"Upserted chunk id={doc_id}")
    return True


def query_knowledge_base(index, query, model,
                         top_k=5, namespace="default",
                         metadata_filter=None):  # pragma: no cover
    """
    Query the Pinecone index with an embedding of the query string.
    Supports metadata filtering and returns the top_k matches.
    """
    emb = generate_embedding(query, model)
    resp = index.query(vector=emb,
                       top_k=top_k,
                       include_metadata=True,
                       filter=metadata_filter or {},
                       namespace=namespace)
    return resp.get("matches", [])
