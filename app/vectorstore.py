from langchain_community.vectorstores import OpenSearchVectorSearch  # LangChain class that stores/searches vectors in OpenSearch.

from app.config import OPENSEARCH_INDEX_NAME, OPENSEARCH_URL  # Central OpenSearch settings.
from app.embeddings import get_embeddings  # Function that creates the embedding model.

def get_vectorstore():  # Creates the object used for semantic/vector search.
    return OpenSearchVectorSearch(  # LangChain vector-store wrapper around OpenSearch.
        index_name=OPENSEARCH_INDEX_NAME,  # Index where our article chunks are stored.
        embedding_function=get_embeddings(),  # Converts query text into the same vector type as stored chunks.
        opensearch_url=OPENSEARCH_URL,  # URL of the OpenSearch service.
        use_ssl=True,  # Uses HTTPS because hosted OpenSearch usually requires it.
        verify_certs=True,  # Verifies the server certificate for safer connections.
        ssl_assert_hostname=False,  # Keeps compatibility with some managed OpenSearch certificate setups.
        ssl_show_warn=False,  # Avoids noisy SSL warnings in logs.
        engine="lucene",  # Uses Lucene engine because it works well for this OpenSearch setup.
    )
