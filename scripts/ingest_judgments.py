"""
Ingest JUDIS Supreme Court judgment PDFs into ChromaDB.

Dataset: https://www.kaggle.com/datasets/vangap/indian-supreme-court-judgments
PDFs are located in: D:\\LEGAL_CHATBOT\\data\\judgments\\pdfs\\

The PDFs are from JUDIS (Supreme Court of India) and have a structured header:
  PETITIONER: ...
  RESPONDENT: ...
  DATE OF JUDGMENT: ...
  BENCH: ...
  JUDGMENT: ...  (actual text starts here)

Metadata is extracted directly from each PDF — no separate CSV needed.

Usage:
    python ingest_judgments.py --scan            # preview 5 PDFs
    python ingest_judgments.py --limit 500       # ingest first 500 PDFs (testing)
    python ingest_judgments.py                   # full ingest (48k PDFs)
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
PDF_DIR         = Path(__file__).parent.parent / "data" / "judgments" / "pdfs"
CHROMA_PATH     = str(Path(__file__).parent.parent / "vectorstore")
COLLECTION_NAME = "statutes_v1"
EMBED_MODEL     = "all-MiniLM-L6-v2"
ACT_NAME        = "SUPREME_COURT_JUDGMENT"
BATCH_SIZE      = 32           # smaller batch — PDFs can be large
MAX_WORDS_CHUNK = 400
MIN_WORDS_CHUNK = 60
MIN_JUDGMENT_WORDS = 30        # skip near-empty PDFs

MANIFEST_PATH   = Path(__file__).parent / "judgments_manifest.json"


# ── JUDIS PDF parser ───────────────────────────────────────────────────────────
# Patterns for the JUDIS structured header
_RE_PETITIONER   = re.compile(r'PETITIONER\s*:\s*(.+?)(?=\n\s*Vs\.|\n\s*RESPONDENT)', re.S | re.I)
_RE_RESPONDENT   = re.compile(r'RESPONDENT\s*:\s*(.+?)(?=\nDATE OF JUDGMENT|\nBENCH\s*:)', re.S | re.I)
_RE_DATE         = re.compile(r'DATE OF JUDGMENT\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}/\d{2}/\d{4})', re.I)
_RE_DATE2        = re.compile(r'DATE OF JUDGMENT\s*(\d{2}/\d{2}/\d{4})', re.I)
_RE_BENCH        = re.compile(r'BENCH\s*:\s*(.+?)(?=\nCITATION|\nACT|\nHEADNOTE|\nJUDGMENT)', re.S | re.I)
_RE_JUDGMENT_SEP = re.compile(r'\nJUDGMENT\s*:\s*\n', re.I)
_RE_CITATION     = re.compile(r'CITATION\s*:\s*(.+?)(?=\nACT|\nHEADNOTE|\nJUDGMENT)', re.S | re.I)


def _first_line(text: str) -> str:
    return text.strip().split("\n")[0].strip()[:200]


def _clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"Page \d+ of \d+", "", text)   # remove page headers
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def parse_judis_pdf(pdf_path: Path) -> Optional[Dict]:
    """
    Extract text and metadata from a JUDIS-format PDF.
    Returns None if the PDF is unreadable or too short.
    """
    try:
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
    except Exception:
        return None

    full_text = "\n".join(pages)
    full_text = _clean(full_text)

    if len(full_text.split()) < MIN_JUDGMENT_WORDS:
        return None

    # Extract metadata from header (first page usually has it)
    header = pages[0] if pages else full_text[:2000]

    petitioner = ""
    m = _RE_PETITIONER.search(header)
    if m:
        petitioner = _first_line(m.group(1))

    respondent = ""
    m = _RE_RESPONDENT.search(header)
    if m:
        respondent = _first_line(m.group(1))

    date = ""
    m = _RE_DATE.search(header) or _RE_DATE2.search(header)
    if m:
        date = m.group(1).strip()

    bench = ""
    m = _RE_BENCH.search(header)
    if m:
        bench = _first_line(m.group(1))

    citation = ""
    m = _RE_CITATION.search(header)
    if m:
        citation = _first_line(m.group(1))

    # Extract only the judgment body (after "JUDGMENT:" marker if present)
    judgment_parts = _RE_JUDGMENT_SEP.split(full_text, maxsplit=1)
    judgment_text = judgment_parts[1] if len(judgment_parts) > 1 else full_text

    if len(judgment_text.split()) < MIN_JUDGMENT_WORDS:
        judgment_text = full_text   # fallback to full text

    title = f"{petitioner} vs {respondent}" if petitioner and respondent else pdf_path.stem
    case_id = pdf_path.stem  # use filename as unique case ID

    return {
        "case_id":    case_id,
        "text":       judgment_text,
        "title":      title[:300],
        "petitioner": petitioner,
        "respondent": respondent,
        "date":       date,
        "bench":      bench,
        "citation":   citation,
        "court":      "Supreme Court of India",
        "pages":      len(pages),
    }


# ── Text chunker ───────────────────────────────────────────────────────────────
def chunk_text(text: str, max_words: int = MAX_WORDS_CHUNK) -> List[str]:
    """Split text into word-capped chunks at paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks, current, count = [], [], 0

    for para in paragraphs:
        words = para.split()
        if count + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(para)
        count += len(words)

    if current:
        chunks.append(" ".join(current))

    # Split oversized single paragraphs by sentence
    final = []
    for chunk in chunks:
        if len(chunk.split()) > max_words * 1.5:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            sub, sub_count = [], 0
            for s in sentences:
                w = len(s.split())
                if sub_count + w > max_words and sub:
                    final.append(" ".join(sub))
                    sub, sub_count = [], 0
                sub.append(s)
                sub_count += w
            if sub:
                final.append(" ".join(sub))
        else:
            final.append(chunk)

    return [c for c in final if len(c.split()) >= MIN_WORDS_CHUNK]


# ── Ingestion ──────────────────────────────────────────────────────────────────
def _flush(model, collection, ids, texts, metas):
    embeds = model.encode(texts, show_progress_bar=False,
                          normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeds)


def ingest(pdf_paths: List[Path], limit: Optional[int] = None) -> None:
    if limit:
        pdf_paths = pdf_paths[:limit]
    print(f"\nIngesting {len(pdf_paths)} PDFs ...")

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    model = SentenceTransformer(EMBED_MODEL)

    # Load already-ingested case IDs to support resumable runs
    existing_ids: set = set()
    try:
        existing_data = collection.get(
            where={"act": {"$eq": ACT_NAME}}, include=["metadatas"]
        )
        existing_ids = {m.get("case_id", "") for m in existing_data["metadatas"]}
        if existing_ids:
            print(f"  Resuming — {len(existing_ids)} cases already ingested, skipping them.")
    except Exception:
        pass

    manifest: List[Dict] = []
    doc_ids:  List[str]  = []
    doc_texts: List[str] = []
    doc_metas: List[Dict] = []

    skipped = parsed = chunk_total = 0

    for pdf_path in tqdm(pdf_paths, desc="Parsing & ingesting"):
        case_id = pdf_path.stem
        if case_id in existing_ids:
            skipped += 1
            continue

        record = parse_judis_pdf(pdf_path)
        if not record:
            skipped += 1
            continue

        parsed += 1
        chunks = chunk_text(record["text"])
        if not chunks:
            skipped += 1
            continue

        year = record["date"].split("/")[-1] if "/" in record["date"] \
               else record["date"].split("-")[-1] if "-" in record["date"] else ""

        for c_idx, chunk in enumerate(chunks):
            # Build a stable doc ID from case_id + chunk index
            doc_id = f"jdg_{hashlib.md5(case_id.encode()).hexdigest()[:12]}_{c_idx}"
            meta = {
                "act":         ACT_NAME,
                "section":     case_id,
                "heading":     record["title"][:200],
                "case_id":     case_id,
                "petitioner":  record["petitioner"][:200],
                "respondent":  record["respondent"][:200],
                "court":       record["court"],
                "bench":       record["bench"][:200],
                "citation":    record["citation"][:200],
                "date":        record["date"],
                "year":        year,
                "chunk_index": c_idx,
                "page_start":  0,
                "page_end":    record["pages"],
            }
            doc_ids.append(doc_id)
            doc_texts.append(chunk)
            doc_metas.append(meta)
            manifest.append({"id": doc_id, **meta})
            chunk_total += 1

        if len(doc_ids) >= BATCH_SIZE:
            _flush(model, collection, doc_ids, doc_texts, doc_metas)
            doc_ids, doc_texts, doc_metas = [], [], []

    if doc_ids:
        _flush(model, collection, doc_ids, doc_texts, doc_metas)

    # Save manifest (append mode — don't overwrite existing)
    existing_manifest: List[Dict] = []
    if MANIFEST_PATH.exists():
        try:
            existing_manifest = json.loads(
                MANIFEST_PATH.read_text(encoding="utf-8", errors="replace")
            )
        except Exception:
            pass
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_manifest + manifest, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  PDFs processed : {parsed}")
    print(f"  Skipped        : {skipped}")
    print(f"  Chunks stored  : {chunk_total}")
    print(f"  Collection size: {collection.count()}")
    print(f"  Manifest saved : {MANIFEST_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ingest JUDIS Supreme Court PDFs into ChromaDB")
    parser.add_argument("--scan",  action="store_true", help="Preview 5 PDFs, no ingestion")
    parser.add_argument("--limit", type=int, default=None, help="Ingest only first N PDFs (testing)")
    args = parser.parse_args()

    if not PDF_DIR.exists():
        print(f"ERROR: PDF directory not found: {PDF_DIR}")
        print("Please ensure the PDFs are in data/judgments/pdfs/")
        sys.exit(1)

    pdf_paths = sorted(PDF_DIR.rglob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs in {PDF_DIR}")

    if not pdf_paths:
        print("No PDFs found. Check the folder path.")
        sys.exit(1)

    if args.scan:
        print("\n── Scanning first 5 PDFs ──")
        for pdf in pdf_paths[:5]:
            rec = parse_judis_pdf(pdf)
            if rec:
                chunks = chunk_text(rec["text"])
                print(f"\n  File    : {pdf.name}")
                print(f"  Title   : {rec['title'][:80]}")
                print(f"  Date    : {rec['date']}")
                print(f"  Bench   : {rec['bench'][:80]}")
                print(f"  Pages   : {rec['pages']}")
                print(f"  Words   : {len(rec['text'].split())}")
                print(f"  Chunks  : {len(chunks)}")
                print(f"  Preview : {rec['text'][:200]}...")
            else:
                print(f"\n  File    : {pdf.name}  → PARSE FAILED / TOO SHORT")
        return

    ingest(pdf_paths, limit=args.limit)


if __name__ == "__main__":
    main()

