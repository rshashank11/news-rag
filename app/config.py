import os
from dotenv import load_dotenv

load_dotenv()

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL")

OPENSEARCH_INDEX_NAME = os.environ.get("OPENSEARCH_INDEX_NAME", "news_index")

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
NORMALIZE_EMBEDDINGS = os.environ.get("NORMALIZE_EMBEDDINGS", "true").lower() == "true"

DEFAULT_SEARCH_K = int(os.environ.get("DEFAULT_SEARCH_K", "5"))
MIN_SEARCH_K = int(os.environ.get("MIN_SEARCH_K", "3"))
MAX_SEARCH_K = int(os.environ.get("MAX_SEARCH_K", "20"))
