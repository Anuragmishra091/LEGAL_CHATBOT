#!/usr/bin/env python3
"""
Constitution of India — Article-level ingestion into ChromaDB.

Run modes:
  python ingest_constitution.py --scan [N]   Print raw text of first N pages (default 5)
  python ingest_constitution.py --dry-run    Detect articles/parts, print summary, skip DB write
  python ingest_constitution.py              Full ingestion
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass, asdict

import fitz
from tqdm import tqdm
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
PDF_PATH        = Path(__file__).parent.parent / "data" / "Constitution of India.pdf"
CHROMA_PATH     = str(Path(__file__).parent.parent / "vectorstore")
COLLECTION_NAME = "statutes_v1"
ACT_NAME        = "CONSTITUTION_OF_INDIA"
EMBED_MODEL     = "all-MiniLM-L6-v2"
MANIFEST_OUT    = Path(__file__).parent / "constitution_manifest.json"
MAX_WORDS_SPLIT = 500   # Articles longer than this are split at numbered sub-clauses


# ── Dataclass ──────────────────────────────────────────────────────────────────
@dataclass
class ArticleChunk:
    id:              str
    act:             str
    part:            str         # e.g. "III"
    part_name:       str         # e.g. "FUNDAMENTAL RIGHTS"
    article_no:      str         # e.g. "21" or "21A"
    article_heading: str
    chunk_index:     int         # 0 = full article; 1,2,... = sub-clause splits
    chunk_text:      str
    page_start:      int
    page_end:        int
    cross_refs:      List[str]   # article numbers explicitly cited in this chunk


# ── Text Extraction ────────────────────────────────────────────────────────────
def extract_text_with_pages(pdf_path: Path) -> str:
    """Extract full PDF text with ===PAGE:N=== markers for page tracking."""
    doc = fitz.open(str(pdf_path))
    parts = []
    for i, page in enumerate(doc):
        parts.append(f"\n===PAGE:{i}===\n")
        parts.append(page.get_text("text"))
    doc.close()
    return "".join(parts)


def get_page_for_pos(text: str, pos: int) -> int:
    """Return page number active at character position pos."""
    last = 0
    for m in re.finditer(r'===PAGE:(\d+)===', text[:pos]):
        last = int(m.group(1))
    return last


def clean_text(text: str) -> str:
    """Strip PAGE markers and collapse excessive blank lines."""
    text = re.sub(r'===PAGE:\d+===', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── Patterns ───────────────────────────────────────────────────────────────────

# PART III\nFUNDAMENTAL RIGHTS  or  PART IVA—FUNDAMENTAL DUTIES
RE_PART = re.compile(
    r'PART\s+([IVXLC]+[A-Z]?)\s*[—\-\n]\s*([A-Z][A-Z ,\(\)]+)',
    re.MULTILINE,
)

# Article number at line start: "1. " or "21A. "
RE_ART_NUM = re.compile(r'(?m)^\s{0,4}(\d{1,3}[A-Z]?)\.\s+')

# Footnote line starters — used to reject false positives from page footnotes
RE_FOOTNOTE = re.compile(
    r'^(Subs|Ins|Rep|Omitted|Cl\.|Art\.|Ss\.|w\.e\.f|Provided)',
    re.IGNORECASE,
)

# "(1)", "(2)" at line start — used for sub-clause splitting
RE_SUBCLAUSE = re.compile(r'(?m)^(\s*\(\d+\)\s)')

# "article 21", "articles 14 and 19" — cross-reference extraction
RE_CROSSREF = re.compile(
    r'\barticles?\s+(\d{1,3}[A-Z]?)(?:\s*(?:,\s*|,?\s+(?:and|or|to)\s+)(\d{1,3}[A-Z]?))*',
    re.IGNORECASE,
)


# ── Body-start Detection (skip Table of Contents) ─────────────────────────────
def find_body_start(raw_text: str) -> int:
    """
    Return char position where the actual Constitution body begins (after TOC).
    Anchors to the Preamble which only appears once in the body, not the TOC.
    """
    for pattern in [
        r'WE,?\s+THE\s+PEOPLE\s+OF\s+INDIA',
        r'PREAMBLE\s*\nWE',
    ]:
        m = re.search(pattern, raw_text, re.IGNORECASE)
        if m:
            return m.start()
    return 0  # fallback


def find_body_end(raw_text: str, body_start: int) -> int:
    """
    Return char position where the Schedules begin (end of article text).
    Searches only within body text (after TOC).
    Handles footnote-marker prefixes like '1[FIRST SCHEDULE' in the PDF.
    """
    body = raw_text[body_start:]
    # Match "FIRST SCHEDULE" optionally preceded by footnote markers (digits, [)
    m = re.search(r'(?m)^[0-9\[]*FIRST\s+SCHEDULE', body)
    if m:
        return body_start + m.start()
    return len(raw_text)  # fallback: parse to end


# ── Part Detection ─────────────────────────────────────────────────────────────
def detect_parts(text: str) -> List[Tuple[str, str, int]]:
    """Return [(roman_numeral, part_title, char_position), ...] in document order."""
    results = []
    for m in RE_PART.finditer(text):
        roman = m.group(1).strip()
        name  = re.sub(r'\s+', ' ', m.group(2).strip().rstrip('.,—–'))
        results.append((roman, name, m.start()))
    return results


# ── Article Detection ──────────────────────────────────────────────────────────
def detect_articles(raw_text: str, body_start: int = 0, body_end: int = -1) -> List[Tuple[str, str, int]]:
    """
    Return [(article_no, heading, char_position), ...] in document order.

    Strategy:
    1. Start only from body_start — skips the Table of Contents entirely.
    2. Stop at body_end — skips Schedules and Appendices after Article 395.
    3. Find article number at line start ("1. ", "21A. ").
    4. Search ahead ≤400 chars for the heading-end marker .—
       Correctly handles multi-line headings (e.g. Article 4 spans 3 lines).
    5. Normalize captured heading (collapse whitespace to single spaces).
    6. Reject footnote false positives ("1. Subs. by...", "2. Ins. by...").
    7. Sequential validation: monotonically increasing numbers only.
    """
    if body_end < 0:
        body_end = len(raw_text)
    body = raw_text[body_start:body_end]

    raw = []
    for m in RE_ART_NUM.finditer(body):
        art_no = m.group(1).strip()
        base   = int(re.match(r'\d+', art_no).group())
        if not (1 <= base <= 448):
            continue

        # Search ahead up to 400 chars for .— heading terminator
        window    = body[m.end(): m.end() + 400]
        heading_m = re.search(r'([A-Z].+?)\.—', window, re.DOTALL)
        if not heading_m:
            continue

        # Collapse multi-line heading whitespace
        heading = re.sub(r'\s+', ' ', heading_m.group(1)).strip()

        # Reject footnote lines and very short matches
        if RE_FOOTNOTE.match(heading) or len(heading) < 4:
            continue

        raw.append((base, art_no, heading, body_start + m.start()))

    # ── Sequential validation ──────────────────────────────────────────────────
    # Accept monotonically increasing article numbers.
    # Accept true letter-suffix variants (31 → 31A → 31B).
    # Reject repeated bare numbers (footnote "1." after Article 1 already seen).
    validated = []
    prev_base = 0
    prev_art  = ""

    for base, art_no, heading, pos in raw:
        is_letter_variant = (
            base == prev_base
            and art_no != prev_art
            and bool(re.match(r'\d+[A-Z]$', art_no))
        )
        is_forward = prev_base < base <= prev_base + 20

        if is_letter_variant:
            validated.append((art_no, heading, pos))
            prev_art = art_no
        elif is_forward:
            validated.append((art_no, heading, pos))
            prev_base = base
            prev_art  = art_no

    return validated


# ── Cross-reference Extraction ─────────────────────────────────────────────────
def extract_cross_refs(text: str) -> List[str]:
    """Return all article numbers explicitly mentioned in text."""
    refs = set()
    for m in RE_CROSSREF.finditer(text):
        for num in re.findall(r'\d{1,3}[A-Z]?', m.group()):
            refs.add(num)
    return sorted(refs)


# ── Sub-clause Splitting ───────────────────────────────────────────────────────
def split_long_article(
    article_no: str,
    heading:    str,
    text:       str,
    part:       str,
    part_name:  str,
    page_start: int,
    page_end:   int,
) -> List[ArticleChunk]:
    """
    If article text exceeds MAX_WORDS_SPLIT, split at numbered sub-clauses
    (1), (2), (3)...  Each split chunk is prefixed with the article header
    so every chunk is self-contained for retrieval.
    """
    cross_refs = extract_cross_refs(text)
    id_base    = f"CONST__art_{article_no}"
    prefix     = f"Article {article_no}. {heading}.\n"

    if len(text.split()) <= MAX_WORDS_SPLIT:
        return [ArticleChunk(
            id=f"{id_base}__0", act=ACT_NAME,
            part=part, part_name=part_name,
            article_no=article_no, article_heading=heading,
            chunk_index=0,
            chunk_text=(prefix + text).strip(),
            page_start=page_start, page_end=page_end,
            cross_refs=cross_refs,
        )]

    # re.split with a capturing group keeps the delimiters in the result list:
    # ['pre-text', '(1) ', 'text1', '(2) ', 'text2', ...]
    splits  = RE_SUBCLAUSE.split(text)
    chunks  = []
    idx     = 0
    current = prefix + splits[0]

    i = 1
    while i < len(splits):
        marker = splits[i]
        body   = splits[i + 1] if i + 1 < len(splits) else ""
        candidate = current + marker + body

        if len(candidate.split()) > MAX_WORDS_SPLIT and len(current.split()) >= 50:
            chunks.append(ArticleChunk(
                id=f"{id_base}__{idx}", act=ACT_NAME,
                part=part, part_name=part_name,
                article_no=article_no, article_heading=heading,
                chunk_index=idx,
                chunk_text=current.strip(),
                page_start=page_start, page_end=page_end,
                cross_refs=cross_refs,
            ))
            idx    += 1
            current = prefix + marker + body
        else:
            current = candidate
        i += 2

    if current.strip():
        chunks.append(ArticleChunk(
            id=f"{id_base}__{idx}", act=ACT_NAME,
            part=part, part_name=part_name,
            article_no=article_no, article_heading=heading,
            chunk_index=idx,
            chunk_text=current.strip(),
            page_start=page_start, page_end=page_end,
            cross_refs=cross_refs,
        ))

    # Fallback: return whole article if splitting produced nothing
    return chunks or [ArticleChunk(
        id=f"{id_base}__0", act=ACT_NAME,
        part=part, part_name=part_name,
        article_no=article_no, article_heading=heading,
        chunk_index=0,
        chunk_text=(prefix + text).strip(),
        page_start=page_start, page_end=page_end,
        cross_refs=cross_refs,
    )]


# ── Constitution Parser ────────────────────────────────────────────────────────
def parse_constitution(raw_text: str) -> List[ArticleChunk]:
    """Detect Parts → detect Articles → assign Parts → extract text → chunk."""
    body_start = find_body_start(raw_text)
    body_end   = find_body_end(raw_text, body_start)
    print(f"  Body text: chars {body_start:,} → {body_end:,} (TOC + Schedules excluded)")

    parts    = detect_parts(raw_text)
    articles = detect_articles(raw_text, body_start=body_start, body_end=body_end)
    print(f"  Detected {len(parts)} Parts, {len(articles)} Articles")

    def get_part_at(pos: int) -> Tuple[str, str]:
        """Return the Part active at char position pos."""
        part, name = "PREAMBLE", "Preamble"
        for roman, pname, ppos in parts:
            if ppos <= pos:
                part, name = roman, pname
            else:
                break
        return part, name

    all_chunks: List[ArticleChunk] = []

    for idx, (art_no, heading, art_pos) in enumerate(
        tqdm(articles, desc="Parsing articles")
    ):
        end_pos  = articles[idx + 1][2] if idx + 1 < len(articles) else body_end
        art_text = clean_text(raw_text[art_pos:end_pos])

        # Skip very short matches — likely false positives
        if len(art_text.split()) < 5:
            continue

        pg_start        = get_page_for_pos(raw_text, art_pos)
        pg_end          = get_page_for_pos(raw_text, end_pos - 1)
        part, part_name = get_part_at(art_pos)

        chunks = split_long_article(
            art_no, heading, art_text,
            part, part_name, pg_start, pg_end,
        )
        all_chunks.extend(chunks)

    return all_chunks


# ── ChromaDB Storage ───────────────────────────────────────────────────────────
def embed_and_store(
    chunks: List[ArticleChunk],
    chroma_path: str,
    collection_name: str,
) -> None:
    model      = SentenceTransformer(EMBED_MODEL)
    client     = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 32
    for i in tqdm(range(0, len(chunks), BATCH), desc="Embedding & storing"):
        batch = chunks[i : i + BATCH]
        texts = [c.chunk_text for c in batch]
        embs  = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=[c.id for c in batch],
            documents=texts,
            embeddings=embs,
            metadatas=[
                {
                    "act":             c.act,
                    "part":            c.part,
                    "part_name":       c.part_name,
                    "article_no":      c.article_no,
                    "article_heading": c.article_heading,
                    "chunk_index":     c.chunk_index,
                    # Compatible with existing StatuteRetriever field names:
                    "section":         f"Art.{c.article_no}",
                    "heading":         c.article_heading,
                    "page_start":      c.page_start,
                    "page_end":        c.page_end,
                    "cross_refs":      ",".join(c.cross_refs),
                }
                for c in batch
            ],
        )

    print(f"\nStored {len(chunks)} chunks → collection '{collection_name}'")


# ── Manifest ───────────────────────────────────────────────────────────────────
def save_manifest(chunks: List[ArticleChunk]) -> None:
    data = []
    for c in chunks:
        d = asdict(c)
        d["chunk_text"] = d["chunk_text"][:500]   # truncate for readability
        data.append(d)
    MANIFEST_OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Manifest → {MANIFEST_OUT}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Constitution of India into ChromaDB (article-level)"
    )
    parser.add_argument(
        "--scan", type=int, nargs="?", const=5, metavar="N",
        help="Print raw text of first N pages and exit (default: 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and show article summary without writing to ChromaDB",
    )
    parser.add_argument("--pdf",        default=str(PDF_PATH))
    parser.add_argument("--chroma",     default=CHROMA_PATH)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"ERROR: PDF not found: {pdf}")
        return

    print(f"Reading {pdf} ...")
    raw_text = extract_text_with_pages(pdf)
    print(f"  {len(raw_text):,} characters extracted")

    # ── Scan mode ─────────────────────────────────────────────────────────────
    if args.scan is not None:
        for m in re.finditer(
            r'===PAGE:(\d+)===\n(.*?)(?====PAGE:|$)', raw_text, re.DOTALL
        ):
            if int(m.group(1)) >= args.scan:
                break
            print(f"\n{'='*60}  PAGE {m.group(1)}  {'='*60}")
            print(m.group(2)[:2000])
        return

    # ── Parse ──────────────────────────────────────────────────────────────────
    chunks = parse_constitution(raw_text)
    print(f"\nTotal chunks generated: {len(chunks)}")

    # ── Dry run ────────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n── First 25 detected articles ──")
        for c in chunks[:25]:
            print(
                f"  Art.{c.article_no:5s} | Part {c.part:5s} | "
                f"{c.part_name[:25]:25s} | "
                f"{c.article_heading[:40]:40s} | "
                f"{len(c.chunk_text.split()):4d}w"
            )

        from collections import Counter
        print("\n── Chunks per Part ──")
        part_name_map = {c.part: c.part_name for c in chunks}
        for part, cnt in sorted(Counter(c.part for c in chunks).items()):
            print(f"  PART {part:5s} | {cnt:3d} chunks | {part_name_map[part]}")

        print("\n── Articles with cross-references (sample) ──")
        for c in chunks:
            if c.cross_refs and c.chunk_index == 0:
                print(f"  Art.{c.article_no:5s} → cites: {', '.join(c.cross_refs[:8])}")
        return

    # ── Full ingestion ─────────────────────────────────────────────────────────
    embed_and_store(chunks, args.chroma, args.collection)
    save_manifest(chunks)
    print("\nConstitution ingestion complete.")


if __name__ == "__main__":
    main()
