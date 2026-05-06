from langchain_huggingface import HuggingFaceEmbeddings  # LangChain wrapper for local Hugging Face embedding models.
from app.config import (  # Imports embedding settings from one central config file.
    EMBEDDING_DEVICE,  # "cpu" or another device if available.
    EMBEDDING_MODEL_NAME,  # The exact embedding model name.
    NORMALIZE_EMBEDDINGS,  # Whether vector lengths should be normalized.
)

def get_embeddings():  # Factory function so other files can create the same embedding setup.
    return HuggingFaceEmbeddings(  # Returns an embedding object LangChain can use.
        model_name=EMBEDDING_MODEL_NAME,  # Tells Hugging Face which model to load.
        model_kwargs={  # Extra model-loading options.
            "device": EMBEDDING_DEVICE  # Runs on CPU by default for Docker/free deployment.
        },
        encode_kwargs={  # Extra options used when text is converted into vectors.
            "normalize_embeddings": NORMALIZE_EMBEDDINGS  # Helps similarity search behave consistently.
        }
    )
