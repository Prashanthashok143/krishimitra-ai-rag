from typing import List

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.config import get_settings


class EmbeddingService:
    """Generates semantic vector embeddings for text using Google's gemini-embedding-001 model."""

    def __init__(self):
        self.client = genai.Client(
            api_key=get_settings().GEMINI_API_KEY,
        )
        self.model = "gemini-embedding-001"

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=10))
    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """
        Generate an embedding vector for the given text using the specified task type.

        Args:
            text (str): The input text to generate embeddings for.
            task_type (str): The task type for the embedding. Default is "RETRIEVAL_DOCUMENT".

        Returns:
            List[float]: A list of floating-point numbers representing the embedding vector.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must not be empty")

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768,
            ),
        )
        print("Embedding response:", response);
        print("Embedding values:", response.embeddings);
        return response.embeddings[0].values

    def embed_documents_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of documents.

        Args:
            texts (List[str]): A list of input texts to generate embeddings for.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text, task_type="RETRIEVAL_DOCUMENT")
            embeddings.append(embedding)
        return embeddings