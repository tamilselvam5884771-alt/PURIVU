import os

# Centralized Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base").strip()

def get_embedding_dimension(model_name: str) -> int:
    if "multilingual-e5" in model_name.lower() or "e5-base" in model_name.lower():
        return 768
    elif "minilm" in model_name.lower():
        return 384
    else:
        return 768
