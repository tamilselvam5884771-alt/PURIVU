import sys
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import ChatRequest, ChatResponse, HealthResponse, VisionResponse, ProductInfo
from backend.rag_service import rag_service

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize RAG Service & load FAISS index
    try:
        rag_service.initialize()
        print("[INFO] PURIVU RAG Service initialized successfully.")
    except Exception as e:
        print(f"[WARN] RAG Service initialization warning: {e}")
    yield
    # Shutdown logic if needed
    print("[INFO] Shutting down PURIVU Backend.")

app = FastAPI(
    title="PURIVU API - BIS Saathi Assistant",
    description="REST API powering PURIVU, an AI-powered Intelligent Assistant for Indian Standards and BIS Services.",
    version="1.0.0",
    lifespan=lifespan
)

# -----------------------------
# CORS Configuration
# -----------------------------
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Root Welcome Endpoint
# -----------------------------
@app.get(
    "/",
    summary="Root API Welcome",
    tags=["System"]
)
async def root():
    """
    Root endpoint returning service identity and helpful quick links.
    """
    return {
        "service": "PURIVU API - BIS Saathi Assistant",
        "status": "online",
        "health": "/api/health",
        "docs": "/docs"
    }

# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.get(
    "/api/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="RAG Health Check",
    tags=["System"]
)
async def get_health():
    """
    Verifies that the PURIVU service is online and the BIS FAISS vector index is ready.
    """
    if not rag_service.is_ready():
        err_detail = rag_service._init_error or "FAISS vector database index is missing or failed to initialize."
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "service": "PURIVU",
                "rag": "not_ready",
                "error": err_detail
            }
        )
    
    return HealthResponse(
        status="ok",
        service="PURIVU",
        rag="ready",
        total_chunks=rag_service.get_chunk_count()
    )

# -----------------------------
# Chat API Endpoint
# -----------------------------
@app.post(
    "/api/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Query BIS Knowledge Base",
    tags=["RAG Chat"]
)
async def chat_endpoint(request: ChatRequest):
    """
    Processes user questions against the local BIS knowledge base using multi-query expansion,
    FAISS similarity search, and Gemini grounded generation.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or contain only whitespace."
        )

    try:
        result = rag_service.process_query(request.question, response_language=request.response_language)

        return ChatResponse(
            answer=result["answer"],
            evidence_status=result["evidence_status"],
            sources=result["sources"],
            response_language=result.get("response_language", "English")
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BIS Knowledge Base index missing. Please initialize the database."
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        import traceback
        print(f"[ERROR] Chat Endpoint Exception: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request through the BIS knowledge engine: {str(e)}"
        )

# -----------------------------
# Vision API Endpoint
# -----------------------------
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit

@app.post(
    "/api/vision",
    response_model=VisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Product Image & Query BIS Knowledge Base",
    tags=["Vision RAG"]
)
async def vision_endpoint(
    image: UploadFile = File(..., description="Uploaded product image file (JPEG, PNG, WEBP, max 10MB)"),
    question: Optional[str] = Form(None, description="Optional question about the product")
):
    """
    Identifies physical product details from an uploaded image using Gemini Vision,
    then automatically queries the local BIS RAG knowledge base for applicable evidence.
    """
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a valid product image file."
        )

    ext = image.filename.split(".")[-1].lower() if "." in image.filename else ""
    if image.content_type not in ALLOWED_IMAGE_TYPES and ext not in {"jpg", "jpeg", "png", "webp"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a JPEG, PNG, or WEBP image."
        )

    contents = await image.read()
    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image size exceeds the maximum limit of 10 MB."
        )

    try:
        result = rag_service.process_vision_image(contents, question)
        return VisionResponse(
            product=ProductInfo(**result["product"]),
            answer=result["answer"],
            evidence_status=result["evidence_status"],
            sources=result["sources"]
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing the product image."
        )
