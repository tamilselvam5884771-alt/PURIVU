import os

# Centralized Gemini Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-2").strip()
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

def get_embedding_dimension(model_name: str = EMBEDDING_MODEL_NAME) -> int:
    return EMBEDDING_DIMENSION

