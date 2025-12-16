import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import os
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# Embeddings providers
from sentence_transformers import SentenceTransformer

# ChromaDB client
import chromadb
from chromadb.config import Settings

# Optional OpenAI (if provider=openai)
try:
    import openai
except Exception:
    openai = None

load_dotenv()


# --- Configurable constants ---
DEFAULT_EMBED_MODEL_LOCAL = "all-MiniLM-L6-v2"  
# chunking - now based on sentences
MAX_WORDS_PER_CHUNK = 400  # Target words per chunk
MIN_WORDS_PER_CHUNK = 100  # Minimum words (avoid tiny chunks)
SENTENCE_OVERLAP = 1       # Overlap by 1-2 sentences

# --- Utilities & Dataclasses ---
@dataclass
class StatuteChunk:
    id: str
    act: str
    section: str
    heading: str
    chunk_text: str
    source_path: str
    page_start: int
    page_end: int


def extract_text_by_page(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    pages = []
    for p in range(len(doc)):
        page = doc[p]
        text = page.get_text("text")
        pages.append(text)
    doc.close()
    return pages


# Heuristic: identify section headers
SECTION_PATTERNS = [
    # e.g., "Section 123. Heading —"
    re.compile(r'^(Section|SECTION|Sec\.|S\.)\s*(\d+[\w\/\-]*)[.\s\-–—:]*\s*(.*)$', re.IGNORECASE),
    # e.g., "123. Heading"
    re.compile(r'^\s*(\d+[\w\/\-]*)\.\s+([A-Z][\w\s,\'"()-:]+)$'),
    # Chapter headings (fallback)
    re.compile(r'^(CHAPTER|Chapter)\s+([IVXLC0-9]+)\b', re.IGNORECASE),
]


def split_into_sections(pages: List[str]) -> List[Tuple[str, int, int]]:
    """
    Return list of tuples: (section_text, page_start, page_end)
    """
    concatenated = "\n".join([f"\n===PAGE:{i}===\n" + p for i, p in enumerate(pages)])
    headers = []
    for m in re.finditer(r'^(.*)$', concatenated, re.MULTILINE):
        line = m.group(1).strip()
        if not line:
            continue
        for pat in SECTION_PATTERNS:
            mm = pat.match(line)
            if mm:
                headers.append((m.start(), line))
                break

    chunks = []
    if not headers:
        for i, p in enumerate(pages):
            chunks.append((p, i, i))
        return chunks

    header_positions = [pos for pos, _ in headers]
    header_positions.append(len(concatenated))
    for i in range(len(header_positions) - 1):
        start = header_positions[i]
        end = header_positions[i + 1]
        chunk_text = concatenated[start:end].strip()
        page_indices = re.findall(r'===PAGE:(\d+)===', chunk_text)
        if page_indices:
            page_start = int(page_indices[0])
            page_end = int(page_indices[-1])
        else:
            page_start = page_end = 0
        chunk_text = re.sub(r'===PAGE:\d+===', '', chunk_text)
        chunks.append((chunk_text.strip(), page_start, page_end))
    return chunks


def chunk_by_sentences(text: str, max_words=400, min_words=100, overlap_sentences=1) -> List[str]:
    """
    Intelligent sentence-based chunking for legal documents.
    
    Strategy:
    1. Split text into sentences using NLTK
    2. Group sentences until reaching ~max_words
    3. Keep overlap_sentences between chunks for context
    4. Never split mid-sentence
    """
    # Handle empty text
    if not text.strip():
        return []
    
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # If very short, return as-is
    if len(sentences) <= 2:
        return [text.strip()]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_words = len(sentence.split())
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_word_count += sentence_words
        
        # Check if we've reached target size
        if current_word_count >= max_words or i == len(sentences) - 1:
            # Save current chunk
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Start new chunk with overlap
            if i < len(sentences) - 1:
                # Keep last 'overlap_sentences' for context
                overlap_start = max(0, len(current_chunk) - overlap_sentences)
                current_chunk = current_chunk[overlap_start:]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                break
        
        i += 1
    
    # Handle any remaining text
    if current_chunk and i >= len(sentences):
        remaining_text = " ".join(current_chunk).strip()
        if remaining_text and remaining_text not in chunks:
            chunks.append(remaining_text)
    
    return chunks


# --- Embedding providers ---
class LocalEmbedder:
    def __init__(self, model_name=DEFAULT_EMBED_MODEL_LOCAL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(texts, show_progress_bar=False)
        return [emb.tolist() if hasattr(emb, "tolist") else list(map(float, emb)) for emb in embs]


class OpenAIEmbedder:
    def __init__(self, model_name="text-embedding-3-small"):
        if openai is None:
            raise RuntimeError("openai package not installed or failed to import.")
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY in environment to use OpenAI embeddings.")
        openai.api_key = key
        self.model = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        BATCH = 16
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            resp = openai.Embeddings.create(model=self.model, input=batch)
            for r in resp["data"]:
                results.append(r["embedding"])
        return results


# --- ChromaDB helpers ---
def get_chroma_collection(chroma_path: str, collection_name: str):
    """Initialize ChromaDB client and get/create collection."""
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    return client, collection


# --- Main ingestion function ---
def ingest_pdfs(pdf_paths: List[str], act_names: List[str], provider: str, chroma_path: str,
                chroma_collection: str, out_manifest: str = "ingest_manifest.json"):
    assert len(pdf_paths) == len(act_names), "pdf_paths and act_names must correspond"

    # init embedder
    if provider == "local":
        embedder = LocalEmbedder()
    elif provider == "openai":
        embedder = OpenAIEmbedder()
    else:
        raise ValueError("Unknown provider. Use 'local' or 'openai'")

    # init ChromaDB
    chroma_client, collection = get_chroma_collection(chroma_path, chroma_collection)
    print(f"ChromaDB collection '{chroma_collection}' ready at {chroma_path}")

    manifest = []

    for pdf_path, act in zip(pdf_paths, act_names):
        print(f"Ingesting {pdf_path} as act {act} ...")
        pages = extract_text_by_page(pdf_path)
        sections = split_into_sections(pages)

        for sec_index, (section_text, page_start, page_end) in enumerate(tqdm(sections, desc=f"{act} sections")):
            # Try to extract header info
            header_match = None
            first_lines = "\n".join(section_text.splitlines()[:5])
            for pat in SECTION_PATTERNS:
                m = pat.search(first_lines)
                if m:
                    header_match = m
                    break

            if header_match:
                try:
                    groups = header_match.groups()
                    if len(groups) >= 2:
                        section_no = groups[1] if groups[1] else f"{act}_sec_{sec_index}"
                        heading = groups[2] if len(groups) > 2 else ""
                    else:
                        section_no = f"{act}_sec_{sec_index}"
                        heading = first_lines.splitlines()[0][:80]
                except Exception:
                    section_no = f"{act}_sec_{sec_index}"
                    heading = first_lines.splitlines()[0][:80]
            else:
                section_no = f"{act}_sec_{sec_index}"
                heading = section_text.splitlines()[0][:80] if section_text else ""

            # IMPROVED: Use sentence-based chunking
            text_chunks = chunk_by_sentences(
                section_text, 
                max_words=MAX_WORDS_PER_CHUNK,
                min_words=MIN_WORDS_PER_CHUNK,
                overlap_sentences=SENTENCE_OVERLAP
            )
            
            # embed chunks
            embeddings = embedder.embed(text_chunks)

            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings_list = []

            for i, (chunk_text, emb) in enumerate(zip(text_chunks, embeddings)):
                pid = f"{act}__{section_no}__{sec_index}__{i}"
                metadata = {
                    "act": act,
                    "section": section_no,
                    "heading": heading,
                    "source_path": str(Path(pdf_path).absolute()),
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_index": i
                }
                
                ids.append(pid)
                documents.append(chunk_text)
                metadatas.append(metadata)
                embeddings_list.append(emb)

                # manifest
                manifest.append({
                    "id": pid,
                    "act": act,
                    "section": section_no,
                    "heading": heading,
                    "chunk_text": chunk_text[:1000],
                    "source_path": str(Path(pdf_path).absolute()),
                    "page_start": page_start,
                    "page_end": page_end
                })

            # Upsert to ChromaDB
            try:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings_list
                )
            except Exception as e:
                print(f"ChromaDB upsert failed for batch: {e}")

    # write manifest
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Ingestion finished. Total documents in collection: {collection.count()}")
    print(f"Manifest saved to {out_manifest}")


# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser(description="Ingest statute PDFs into ChromaDB vector DB")
    p.add_argument("--pdfs", nargs="+", required=True, help="Paths to statute PDFs")
    p.add_argument("--act_names", nargs="+", required=True, help="Short names for acts (same order as pdfs)")
    p.add_argument("--provider", choices=["local", "openai"], default="local", help="Embedding provider")
    p.add_argument("--chroma_path", default="../vectorstore", help="Path to ChromaDB storage")
    p.add_argument("--chroma_collection", default="statutes_v1", help="ChromaDB collection name")
    p.add_argument("--out_manifest", default="ingest_manifest.json", help="Output JSON manifest")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_pdfs(
        pdf_paths=args.pdfs,
        act_names=args.act_names,
        provider=args.provider,
        chroma_path=args.chroma_path,
        chroma_collection=args.chroma_collection,
        out_manifest=args.out_manifest
    )