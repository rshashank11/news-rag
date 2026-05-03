from langchain_community.vectorstores import OpenSearchVectorSearch

from app.config import OPENSEARCH_INDEX_NAME, OPENSEARCH_URL
from app.embeddings import get_embeddings

def get_vectorstore():
    return OpenSearchVectorSearch(
        index_name=OPENSEARCH_INDEX_NAME,
        embedding_function=get_embeddings(),
        opensearch_url=OPENSEARCH_URL,
        use_ssl=True,
        verify_certs=True,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        engine="lucene",
    )
