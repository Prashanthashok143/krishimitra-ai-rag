import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI , File , Form , HTTPException, UploadFile , status
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings;
from src.database import DatabaseManager;

from src.services.schemas import (
    AdvisoryQueryRequest,
    AdvisoryQueryResponse,
    RetrievedSource,
    UploadAdvisoryResponse,
)

from src.services.document_processor import DocumentProcessor;
from src.services.embedding_service import EmbeddingService;
from src.services.generator_service import GeneratorService;
from src.services.vector_store import VectorStoreService;


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
    
)
logger = logging.getLogger(__name__);
settings = get_settings();

@asynccontextmanager
async def lifespan(app:FastAPI):
    logger.info("Initializing %s....",settings.APP_NAME);
    try:
        await DatabaseManager.connect_to_mongo();
        logger.info("MongoDB Atlas connection established");
    except Exception:
        logger.exception("Failed to connect to MongoDB Atlas")
        raise
    yield
    logger.info("Shutting down %s...", settings.APP_NAME)
    await DatabaseManager.close_mongo_connection()
    
app = FastAPI(
    title="KrishiMitra AI - Agricultural Advisory Service",
    description="Production-ready RAG API serving grounded PJTSAU agricultural advisories for Telangana farmers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


embedding_service = EmbeddingService();
vector_store = VectorStoreService();
generator_service = GeneratorService();
doc_processor = DocumentProcessor();

@app.get("/health",status_code=status.HTTP_200_OK,tags=["System"]) 
async def health_check():
    return {
        "status":"healthy",
        "service":settings.APP_NAME,
        "environment":settings.ENVIRONMENT
    }  
    
    
@app.post(
    "/api/v1/advisories/upload",
    response_model=UploadAdvisoryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Advisory Ingestion"],
)
async def upload_advisory(
    file:UploadFile = File(...,description="offcial Pdf"),
    crop:str=Form(...,description="target")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported.",
        )
    try:
        content = await file.read()
        
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
             tmp.write(content)
             tmp_path=tmp.name;
        chunks = doc_processor.process_pdf(file_path=tmp_path,crop=crop)
        import os
        os.remove(tmp_path)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No extractable text found in the provided PDF.",
            )
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_service.embed_documents_batch(texts);
        documents_to_insert=[];
        for chunk ,embeding in zip(chunks,embeddings):
            documents_to_insert.append({
                "text":chunk["text"],
                "embedding":embeding,
                "metadata":chunk["metadata"],
            })
            
            
        inserted_count = await vector_store.insert_chunks(documents_to_insert);
        logger.info("Successfully indexed %d chunks in MongoDB Atlas.", inserted_count)

        return UploadAdvisoryResponse(
            filename=file.filename,
            crop=crop,
            total_pages=chunks[-1]["metadata"]["page_number"] if chunks else 0,
            chunks_created=inserted_count,
            message=f"Successfully indexed {inserted_count} chunks from {file.filename}.",
        )

    except Exception as exc:
        logger.error("Failed to process advisory upload: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advisory ingestion failed: {str(exc)}",
        )
        
        
async def query_advisory(payload:AdvisoryQueryRequest):
    try:
        query_vector =  embedding_service.embed_text(
            text=payload.question,
            task_type="RETRIEVAL_QUERY"
        )
        
        retrieved_chunks = await vector_store.search_similar_chunks(
            query_vector=query_vector,
            top_k=payload.top_k,
            crop_filter=payload.crop,    
        )
        answer = await generator_service.generate_grounded_answer(
            question=payload.question,
            retrieved_chunks=retrieved_chunks,
        )
        
        sources = [
            RetrievedSource(
                source_file=chunk.get("metadata", {}).get("source", "Unknown"),
                page_number=chunk.get("metadata", {}).get("page_number", 0),
                crop=chunk.get("metadata", {}).get("crop", "General"),
                relevance_score=round(chunk.get("score", 0.0), 4),
            )
            for chunk in retrieved_chunks
        ]

        return AdvisoryQueryResponse(
            answer=answer,
            sources=sources,
            model_used=settings.OPENROUTER_MODEL,
        )

    except Exception as exc:
        logger.error("Error servicing farmer query: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate advisory response. Please try again shortly.",
        )