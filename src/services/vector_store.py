from typing import List, Dict, Optional, Any;
from src.database import get_db_collection;
from src.config import get_settings;


class VectorStoreService:
    """
    Service for managing vector store operations, including storing and retrieving document embeddings.
    """
    
    
    async def insert_chunks(self,chunks:List[Dict[str, Any]])->None:
        """
        Insert document chunks into the vector store.
        
        Args:
            chunks (List[Dict[str, Any]]): A list of document chunks to be inserted.
        """
        if not chunks:
            return 0
        collection = await get_db_collection();
        result = await collection.insert_many(chunks);
        return len(result.inserted_ids);
    
    
    
    async def  search_similar_chunks(self,
                                     query_vector: List[float],
                                     top_k:int=4,
                                     crop_filter:Optional[str] = None,
                                     ) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks in the vector store based on a query vector.
        
        """
        
        collection = await get_db_collection();
        # step 1: Build the aggregation pipeline for vector search
        vector_search_stage: Dict[str, Any] = {
            "index":get_settings().MONGODB_VECTOR_INDEX_NAME,
            "queryVector": query_vector,
            "path": "embedding",
            "numCandidates": max(60, top_k * 5),
            "limit":top_k
        }
        
        if crop_filter:
            vector_search_stage["filter"] = {
                "crop": crop_filter
                }
        pipeline=[
            {"$vectorSearch": vector_search_stage},
            {
                "$project":{
                    "text": 1,
                    "metadata": 1,
                    "embedding": 1,
                    "_id": 0,
                    "score":{"$meta":"vectorSearchScore"}
                }
            }
        ]
        
        cursor = await collection.aggregate(pipeline);
        results = await cursor.to_list(length=top_k);
        return results;
        