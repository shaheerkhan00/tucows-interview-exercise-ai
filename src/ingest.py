import fitz
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from src.config import get_settings
from docx import Document

settings = get_settings()


def load_documents(directory: str) -> List[Dict]:
    """
    Load all Documents from the specified directory.
    Supports PDF, TXT, and DOCX files.
    
    Args:
        directory (str): path to the documents directory
    Returns:
        List[Dict]: list of documents with 'content' and 'metadata'    
    """
    docs = []
    doc_path = Path(directory)
    
    if not doc_path.exists():
        print(f"Warning: Directory {directory} does not exist.")
        return docs
    
    # Load PDF files
    for pdf_file in doc_path.glob("*.pdf"):
        try:
            docs.append(load_pdf(pdf_file))
            print(f"Loaded document: {pdf_file.name}")
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")
    
    # Load TXT files
    for txt_file in doc_path.glob("*.txt"):
        try:
            docs.append(load_txt(txt_file))
            print(f"Loaded document: {txt_file.name}")
        except Exception as e:
            print(f"Error loading {txt_file.name}: {e}")
    
    # Load DOCX files
    for docx_file in doc_path.glob("*.docx"):
        try:
            docs.append(load_docx(docx_file))
            print(f"Loaded document: {docx_file.name}")
        except Exception as e:
            print(f"Error loading {docx_file.name}: {e}")
    
    return docs


def load_pdf(path: Path) -> Dict:
    """
    Load a pdf file and return its text content and metadata.
    
    Args:
        file_path (Path): path to the pdf file
    Returns:
        Dict: dictionary with 'filename','type' and 'content'
    """
    doc = fitz.open(path)
    pages = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page": page_num,
                "text": text
            })
    doc.close()
    return {
        "filename": path.name,
        "type": "pdf",
        "pages": pages
    }

    
def load_txt(path: Path) -> Dict:
    """
    Load a text file and return its content and metadata.
    
    Args:
        path (Path): path to the text file
    Returns:
        Dict: dictionary with 'filename','type' and 'text'
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        "filename": path.name,
        "type": "txt",
        "text": content
    }


def load_docx(path: Path) -> Dict:
    """
    Load a DOCX file and return its content with paragraph tracking.
    
    Args:
        path (Path): path to the DOCX file
    Returns:
        Dict: dictionary with 'filename', 'type' and 'paragraphs'
    """
    doc = Document(path)
    
    # Extract paragraphs with their index
    paragraphs = []
    for idx, para in enumerate(doc.paragraphs, 1):
        if para.text.strip():
            paragraphs.append({
                "paragraph": idx,
                "text": para.text
            })
    
    return {
        "filename": path.name,
        "type": "docx",
        "paragraphs": paragraphs  
    }


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: the text to chunk
        chunk_size: size of each chunk (in characters)
        overlap: number of overlapping characters between chunks
    Returns:
        List: list of text chunks
    """
    if not text.strip():
        return []
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

    
def create_chunks_with_metadata(docs: List[Dict]) -> List[Dict]:
    """
    Create text chunks with metadata from documents.
    Supports PDF (with pages), TXT, and DOCX (with paragraphs).
    """
    chunks = []
    for doc in docs:
        filename = doc["filename"]
        
        if doc["type"] == "pdf":
            for page_info in doc["pages"]:
                page_num = page_info["page"]
                page_text = page_info["text"]
                
                page_chunks = chunk_text(page_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
                
                for chunk_idx, chunk in enumerate(page_chunks):
                    chunks.append({
                        "text": chunk,
                        "source": filename,
                        "page": page_num,  # Page number for PDFs
                        "chunk_index": chunk_idx,
                        "doc_type": "pdf"
                    })
        
        elif doc["type"] == "docx":
            # Process each paragraph separately for better source attribution
            for para_info in doc["paragraphs"]:
                para_num = para_info["paragraph"]
                para_text = para_info["text"]
                
                para_chunks = chunk_text(para_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
                
                for chunk_idx, chunk in enumerate(para_chunks):
                    chunks.append({
                        "text": chunk,
                        "source": filename,
                        "paragraph": para_num,  
                        "chunk_index": chunk_idx,
                        "doc_type": "docx"
                    })
        
        elif doc["type"] == "txt":
            text_chunks = chunk_text(doc["text"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            
            for chunk_idx, chunk in enumerate(text_chunks):
                chunks.append({
                    "text": chunk,
                    "source": filename,
                    "chunk_index": chunk_idx,
                    "doc_type": "txt"
                })
    
    return chunks

      
def ingest_and_index():
    """ 
    Main ingestion pipeline:
    1. Load documents (PDF, TXT, DOCX)
    2. Chunk documents with metadata
    3. Generate embeddings
    4. Build FAISS index
    5. Save index and metadata
    """
    print("Document ingestion Pipeline initiated")
    
    print("Loading Documents...")
    docs = load_documents("data/docs")
    if not docs:
        print("No documents found for ingestion. Exiting pipeline.")
        return
    print(f"Loaded {len(docs)} documents.")
    print("Creating Chunks with Metadata...")
    chunks = create_chunks_with_metadata(docs)
    print(f"Created {len(chunks)} chunks.")
    if not chunks:
        print("No chunks created from documents. Exiting pipeline.")
        return
    print("Generating Embeddings...using model:", settings.EMBEDDING_MODEL)
    embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    print(f"Generated {len(embeddings)} embeddings.")
    print("Building FAISS Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"built index with dimension: {dimension}.")
    print("Saving Index and Metadata to vector store")
    
    # Create directory if not exist
    store_path = Path("data/vector_store")
    store_path.mkdir(parents=True, exist_ok=True)
    
    # Store faiss index
    index_file = store_path / "index.faiss"
    faiss.write_index(index, str(index_file))
    
    # Store metadata
    metadata_file = store_path / "metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(chunks, f)
    print("Ingestion and Indexing completed successfully.")
    print(f"Saved index to {index_file}")
    print(f"Saved metadata to {metadata_file}")
    print(f"Total documents: {len(docs)}, Total chunks: {len(chunks)}, Total embeddings: {len(embeddings)}, Index size: {index.ntotal} vectors, Dimension: {dimension}")
    
    print("Sample Chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        if "page" in chunk:
            source = f"{chunk['source']}, Page {chunk['page']}"
        else:
            source = chunk['source']
        print(f"Chunk {i}: Source: {source}, Text Preview: {chunk['text'][:100]}...")


if __name__ == "__main__":
    ingest_and_index()