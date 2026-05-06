import os  # Gives Python access to environment variables like OPENSEARCH_URL.
from dotenv import load_dotenv  # Loads local values from a .env file while developing.

load_dotenv()  # Reads the .env file once so os.environ can use those values.

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL")  # The OpenSearch server URL where vector chunks are stored.

OPENSEARCH_INDEX_NAME = os.environ.get(  # The OpenSearch index name; think of it like a table for searchable chunks.
    "OPENSEARCH_INDEX_NAME",  # Environment variable name we can override during deployment.
    "news_index",  # Default index name if no environment value is provided.
)

EMBEDDING_MODEL_NAME = os.environ.get(  # The model used to convert text into vectors.
    "EMBEDDING_MODEL_NAME",  # Environment variable name for changing the embedding model.
    "sentence-transformers/all-MiniLM-L6-v2",  # Free, small CPU-friendly embedding model.
)
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")  # Uses CPU by default because the app runs on cheap/free infra.
NORMALIZE_EMBEDDINGS = (  # Normalization makes vector similarity comparisons more stable.
    os.environ.get("NORMALIZE_EMBEDDINGS", "true").lower() == "true"  # Converts "true"/"false" text into a Python boolean.
)

DEFAULT_SEARCH_K = int(os.environ.get("DEFAULT_SEARCH_K", "5"))  # Default number of final sources to return.
MIN_SEARCH_K = int(os.environ.get("MIN_SEARCH_K", "3"))  # Smallest k the planner is allowed to request.
MAX_SEARCH_K = int(os.environ.get("MAX_SEARCH_K", "20"))  # Largest k allowed, so retrieval does not become too expensive.
