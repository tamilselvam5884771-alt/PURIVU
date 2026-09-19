import os

# Centralized Embedding Model Configuration
# Using intfloat/multilingual-e5-small for high quality multilingual RAG with low RAM footprint (~470MB)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small").strip()

def get_embedding_dimension(model_name: str) -> int:
    model_lower = model_name.lower()
    if "e5-small" in model_lower or "minilm" in model_lower:
        return 384
    elif "e5-base" in model_lower:
        return 768
    elif "multilingual-e5" in model_lower:
        return 384
    else:
        return 384
