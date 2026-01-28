from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict

from src.models import TicketRequest, TicketResponse, ErrorResponse
from src.rag import RAGService
from src.llm import LLMService
from src.config import get_settings

settings = get_settings()

# Global services (initialized at startup)
rag_service: RAGService = None
llm_service: LLMService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI application.
    Initializes services once at startup, keeps them in memory.
    """
    global rag_service, llm_service
    
    
    print("KNOWLEDGE ASSISTANT - STARTING UP...........")
    
    
    # Initialize RAG service
    print("Initializing RAG Service...")
    rag_service = RAGService()
    
    # Check if vector store is ready
    if not rag_service.index or rag_service.index.ntotal == 0:
        print(" WARNING: Vector store is empty!")
        print("Run: python src/ingest.py")
        print("Then restart the API server.")
    else:
        print(f"Vector store ready with {rag_service.index.ntotal} chunks")
    
    # Initialize LLM service
    print("Initializing LLM Service...")
    llm_service = LLMService()
    
    
    print("KNOWLEDGE ASSISTANT READY")
    print(f"  Model: {llm_service.model}")
    print(f"  Indexed chunks: {rag_service.index.ntotal if rag_service.index else 0}")
    
    
    yield
    
    print("Shutting down Knowledge Assistant...")


# Create FastAPI app
app = FastAPI(
    title="Knowledge Assistant API",
    description="RAG-powered support ticket resolution system ",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/resolve-ticket",
    response_model=TicketResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"}
    },
    summary="Resolve Support Ticket",
    description="""
    Process a support ticket using RAG pipeline:
    1. Retrieve relevant documentation chunks
    2. Re-rank for accuracy using cross-encoder
    3. Generate structured response using LLM
    4. Return answer with source references and action recommendation
    """
)
async def resolve_ticket(request: TicketRequest) -> TicketResponse:
    """
    Main endpoint for support ticket resolution.
    
    Flow:
    1. Validate input (Pydantic handles this)
    2. Check if vector store is available
    3. Retrieve and re-rank relevant chunks
    4. Format context for LLM
    5. Generate structured response
    6. Return JSON with answer, references, and action
    """
    try:
        # Check if services are initialized
        if not rag_service or not llm_service:
            raise HTTPException(
                status_code=503,
                detail="Services not initialized. Please restart the server."
            )
        
        # Check if vector store is available
        if not rag_service.index or rag_service.index.ntotal == 0:
            raise HTTPException(
                status_code=503,
                detail="Knowledge base is empty. Please run document ingestion first."
            )
        
        query = request.ticket_text
        
        # Stage 1: Retrieve relevant chunks
        results = rag_service.search(query)
        
        # Check if we found any relevant information
        if not results:
            return TicketResponse(
                answer="I couldn't find relevant information in our documentation to answer your question. Please contact support for assistance.",
                references=[],
                action_required="escalate_to_technical"
            )
        
        # Check relevance score (cross-encoder typically scores -10 to +10)
        
        if results[0]["relevance_score"] < -4.0:
            return TicketResponse(
                answer="I couldn't find sufficiently relevant information for your specific query. Please contact support for personalized assistance.",
                references=[],
                action_required="escalate_to_technical"
            )
        
        # Stage 2: Format context for LLM
        context = rag_service.format_context_for_llm(results)
        
        # Stage 3: Generate response using LLM
        response = llm_service.generate_response(query, context)
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (already have proper status codes)
        raise
        
    except Exception as e:
        # Log unexpected errors
        print(f"Unexpected error in resolve_ticket: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.get("/health",include_in_schema=False)
async def health() -> Dict:
    """Health check endpoint"""
    is_healthy = (
        rag_service is not None and
        llm_service is not None and
        rag_service.index is not None
    )
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "llm_model": llm_service.model if llm_service else "not initialized",
        "indexed_chunks": rag_service.index.ntotal if (rag_service and rag_service.index) else 0,
        "services": {
            "rag": rag_service is not None,
            "llm": llm_service is not None
        }
    }


@app.get("/stats",include_in_schema=False)
async def stats() -> Dict:
    """Statistics endpoint"""
    if not rag_service or not rag_service.index:
        return {
            "error": "Vector store not initialized",
            "hint": "Run: python -m src.ingest"
        }
    
    # Count unique documents
    sources = set()
    for metadata in rag_service.metadata:
        sources.add(metadata.get("source", "Unknown"))
    
    return {
        "total_chunks": rag_service.index.ntotal,
        "total_documents": len(sources),
        "document_list": sorted(list(sources)),
        "configuration": {
            "llm_model": llm_service.model if llm_service else None,
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k_retrieve": settings.TOP_K_RETRIEVE,
            "top_k_rerank": settings.TOP_K_RERANK
        }
    }
@app.get("/", summary="Root", description="API root endpoint with basic information")
async def root() -> Dict:
    """Root endpoint with API information"""
    return {
        "name": "Knowledge Assistant API",
        "version": "1.0.0",
        "description": "RAG-powered support ticket resolution",
        "endpoints": {
            "main": "/resolve-ticket (POST)",
            "health": "/health (GET)",
            "stats": "/stats (GET)",
            "docs": "/docs (GET)"
        }
    }