from langchain_huggingface import HuggingFaceEmbeddings
from app.config import EMBEDDING_DEVICE, EMBEDDING_MODEL_NAME, NORMALIZE_EMBEDDINGS

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": EMBEDDING_DEVICE
        },
        encode_kwargs={
            "normalize_embeddings": NORMALIZE_EMBEDDINGS
        }
    )
