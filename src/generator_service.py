import logging
from typing import List, Dict, Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from  src.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """You are KrishiMitra AI (రైతు మిత్ర / कृषिमित्र), an authoritative agricultural expert assistant for farmers in Telangana, India.

Your primary duty is to provide safe, actionable, and grounded agricultural advice based EXCLUSIVELY on the provided official advisory excerpts.

STRICT OPERATING RULES:
1. Grounding: Answer using ONLY facts, chemical names, and dosages explicitly stated in the RETRIEVED ADVISORY CONTEXT below.
2. Anti-Hallucination: If the context does not contain enough information to safely answer, state:
   "Based on the official advisories currently available in the database, I do not have verified instructions for this specific query. Please consult your local Mandal Agricultural Officer (MAO) or KVK."
3. Chemical Safety: Always state the exact chemical formulation, percentage, and dosage per litre of water or per acre. Never guess dosages.
4. Source Attribution: Explicitly reference the source file and page numbers provided in the headers (e.g., 'As noted in Source 1, Page 3...').
5. Language: Provide clear, step-by-step guidance in English while preserving official chemical/trade names.
"""
class GeneratorService:
    """
    Handles grounded text generation via OpenRouter's dynamic free tier
    """
    
    def __init__(self):
        settings= get_settings();
        self.model_name = settings.OPENROUTER_MODEL
        self.client=AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer":"http://localhost:8000",
                "X-Title":settings.APP_NAME
            }
        )
        
    def _format_context(self,retrieved_chunks:List[Dict[str,Any]]) -> str:
        if not retrieved_chunks:
            return "NO CHunks"
        
        formatted_sections = [];
        
        for idx, chunk in enumerate(retrieved_chunks,start=1):
            meta = chunk.get("metadata",{});
            header = (
                f"--- SOURCE {idx}: {meta.get('source', 'Unknown')} "
                f"(Page {meta.get('page_number', '?')}, Crop: {meta.get('crop', 'General')}) ---"
            )
            text = chunk.get("text", "").strip()
            formatted_sections.append(f"{header}\n{text}")

        return "\n\n".join(formatted_sections);

    
    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1,max=10))
     
    async def generate_grounded_answer(
         self,
         question:str,
         retrieved_chunks: List[Dict[str,Any]]
     ) -> str:
         """
         Excutes non blocking LLLm inference
         """
         context_block = self._format_context(retrieved_chunks);
         user_content = f""""
         RETRIEVED ADVISORY CONTEXT:
         {context_block}
         FARMER QUESTION:
         {question}
         Provide an accurate, grounded agricultural recommendation 
         based strictly on the advisory context above:"""
         
         try:
             response = await self.client.chat.completions.create(
                 model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_content},
                ],
                 temperature=0.2,
                 max_tokens=1000)
             return response.choices[0].message.content or "No response generated";
         
         except Exception as exc:
             logger.error("OpenRouter infernece failed",exc_info=True)
             raise
        
    