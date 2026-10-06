from typing import List, Dict, Any;
from pypdf import PdfReader;
from langchain_text_splitters import RecursiveCharacterTextSplitter;


class DocumentProcessor:
    
    
    def __init__(self , chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n",".", " ", ""]
        )
        
    def process_pdf(self, file_path: str,crop: str)->List[Dict[str, Any]]:
        """
        Process a PDF file and return a list of text chunks.
        
        Args:
            file_path (str): The path to the PDF file.
            crop (str): The cropping option for the PDF pages."""
            
        reader = PdfReader(file_path);
        chunks: List[Dict[str, Any]] = [];
        
        filename = file_path.replace("\\","/").split("/")[-1];
        print("reader",reader);
        print("filename",filename);
        
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or "";
            print("text",text);
            if text:
                page_chunks = self.splitter.split_text(text);
                print("page_chunks",page_chunks);
                for idx, chunk in enumerate(page_chunks):
                    chunks.append({
                        "source": filename,
                        "filename": filename,
                        "page_number": page_number,
                        "text": chunk,
                        "chunk_index": idx,
                        "crop": crop
                    });
            
        return chunks;