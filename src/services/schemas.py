from typing import List, Optional;
from pydantic import BaseModel, Field;



class DocumentChunkMetadata(BaseModel):
    source_file: str
    page_number: int
    chunk_index: int
    crop: Optional[str] = None
    author: Optional[str] = None
    
    
class DocumentChunk(BaseModel):
    text: str
    metadata: DocumentChunkMetadata
    embedding: Optional[List[float]] 

class UploadAdvisoryResponse(BaseModel):
    filename: str
    crop: str
    total_pages: int
    chunks_created: int
    message: str
    
    
class AdvisoryQueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["How to control pink bollworm in cotton"],
    )

    crop: Optional[str] = Field(
        default=None,
        description="Optional crop name to filter the documents for the query.",
        examples=["cotton", "wheat", "rice"],
    )

    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of top relevant documents to retrieve for the query.",
    )
    
class RetrievedSource(BaseModel):
    source_file: str
    page_number: int
    chunk_index: int
    crop: Optional[str] = None
    
class AdvisoryQueryResponse(BaseModel):
    answer: str
    sources: List[RetrievedSource]
    model_used: str