from pathlib import Path
from typing import List, Dict, Any;
from pypdf import PdfReader;
import re;
import logging;
from langchain_text_splitters import RecursiveCharacterTextSplitter;

logger = logging.getLogger(__name__)

class DocumentProcessor:
    
    
    def __init__(self , chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n",".", " ", ""]
        )
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean the extracted text by removing unwanted characters and whitespace.
        
        Args:
            text (str): The input text to be cleaned.   
        """
        if not text:
            return ""
        text = text.replace("\n", " ").replace("\r", " ").strip()
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
        text = re.sub(r"[^\w\s.,;:!?()\-]", "", text) # Remove unwanted characters
        # Strip null bytes and non-printable control characters
        text = text.replace("\x00", " ")
        # Fix words hyphenated across line breaks (e.g., 'pes-\nticide' -> 'pesticide')
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Collapse multiple spaces and tabs into single space
        text = re.sub(r"[ \t]+", " ", text)
        # Normalize excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip();
    
    
    def process_pdf(self, file_path: str,crop: str)->List[Dict[str, Any]]:
        """
        Process a PDF file and return a list of text chunks.
        
        Args:
            file_path (str): The path to the PDF file.
            crop (str): The cropping option for the PDF pages."""
            
        reader = PdfReader(file_path);
        chunks: List[Dict[str, Any]] = [];
        path = Path(file_path);
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF file path: {file_path}");
        
        filename = path.name;
        logger.info(f"Processing PDF file: {filename}, Crop: {crop}, Total Pages: {len(reader.pages)}");
        
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or "";
            cleaned_text = self.clean_text(text);
            if not cleaned_text:
                logger.warning(f"No text extracted from page {page_number} of {filename}");
                continue;
            logger.info(f"Processing page {page_number} of {filename}");
            if cleaned_text:
                page_chunks = self.splitter.split_text(cleaned_text);
                for idx, chunk in enumerate(page_chunks):
                    chunks.append({
                        "source": filename,
                        "filename": filename,
                        "page_number": page_number,
                        "text": chunk,
                        "chunk_index": idx,
                        "crop": crop
                    });
        logger.info(f"Total chunks created from {filename}: {len(chunks)}");
        return chunks;