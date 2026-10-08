from pathlib import Path
from typing import List, Dict,Set, Any;
from pypdf import PdfReader;
import re;
import logging;
from langchain_text_splitters import RecursiveCharacterTextSplitter;

logger = logging.getLogger(__name__);


# Production Agricultural & Welfare Taxonomy for Telangana
AGRICULTURAL_TAXONOMY = {
    "crops": {
        "Cotton": ["cotton", "kapas", "gossypium"],
        "Chilli": ["chilli", "mirapa", "capsicum"],
        "Paddy": ["paddy", "rice", "vari"],
        "Maize": ["maize", "corn", "mokkajonna"],
        "Pulses": ["redgram", "greengram", "blackgram", "bengalgram"],
    },
    "pests_diseases": {
        "Pink Bollworm": ["pink bollworm", "pectinophora", "rosette flower"],
        "Sucking Pests": ["aphid", "jassid", "whitefly", "hopper burn"],
        "Black Thrips": ["black thrips", "thrips parvispinus"],
        "Anthracnose": ["anthracnose", "die-back", "fruit rot", "colletotrichum"],
        "Wilt": ["fusarium", "wilt", "root rot"],
    },
    "schemes": {
        "Rythu Bharosa": ["rythu bharosa", "investment support", "dharani portal", "pattadar"],
        "Rythu Bima": ["rythu bima", "farmers life insurance", "nominee", "5,00,000"],
        "TSMIP": ["tsmip", "micro irrigation", "drip irrigation", "sprinkler"],
    },
}

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
    
    def _extract_taxonomy(self, text: str) -> Dict[str, Any]:
        """
        Determines the primary category, mentioned crops, and key entities.
        """
        text_lower = text.lower()

        detected_crops: Set[str] = set()
        detected_pests: Set[str] = set()
        detected_schemes: Set[str] = set()

        for crop_name, keywords in AGRICULTURAL_TAXONOMY["crops"].items():
            if any(k in text_lower for k in keywords):
                detected_crops.add(crop_name)

        for pest_name, keywords in AGRICULTURAL_TAXONOMY["pests_diseases"].items():
            if any(k in text_lower for k in keywords):
                detected_pests.add(pest_name)

        for scheme_name, keywords in AGRICULTURAL_TAXONOMY["schemes"].items():
            if any(k in text_lower for k in keywords):
                detected_schemes.add(scheme_name)

        # Infer broad category
        if detected_schemes:
            category = "Government Scheme"
        elif detected_crops or detected_pests:
            category = "Crop Management"
        elif any(k in text_lower for k in ["rainfall", "temperature", "agro-climatic", "weather"]):
            category = "Agro-Climatic Advisory"
        else:
            category = "General Agronomy"

        all_entities = list(detected_pests | detected_schemes)

        return {
            "category": category,
            "crops": list(detected_crops),
            "entities": all_entities,
        }
    def process_pdf(self, file_path: str)->List[Dict[str, Any]]:
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
        logger.info(f"Processing PDF file: {filename}, Total Pages: {len(reader.pages)}");
        
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
                    taxonomy = self._extract_taxonomy(chunk);
                    chunks.append({
                        "source": filename,
                        "filename": filename,
                        "page_number": page_number,
                        "text": chunk,
                        "chunk_index": idx,
                        "category": taxonomy["category"],
                        "crops": taxonomy["crops"],
                        "entities": taxonomy["entities"],
                    });
        logger.info(f"Total chunks created from {filename}: {len(chunks)}");
        return chunks;