#!/usr/bin/env python3
"""
Constitution of India — Knowledge Graph Builder.

Builds two types of edges and stores them in a SQLite database:

  HARD edges  — extracted from explicit cross-references in article text
                (e.g. "subject to article 22", "notwithstanding article 13")
                confidence = 1.0, edge_source = 'hard'

  SOFT edges  — inferred from embedding cosine similarity within each Part
                (only articles in the SAME Part are compared, reducing noise)
                confidence = similarity score, edge_source = 'soft'

Run modes:
  python constitution_graph.py --hard-only    Build only hard edges
  python constitution_graph.py --soft-only    Build only soft edges
  python constitution_graph.py                Build both (default)
  python constitution_graph.py --stats        Print graph statistics and exit
  python constitution_graph.py --validate     Validate soft threshold using hard edges
"""

import re
import json
import sqlite3
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_PATH     = str(Path(__file__).parent.parent / "vectorstore")
COLLECTION_NAME = "statutes_v1"
ACT_NAME        = "CONSTITUTION_OF_INDIA"
GRAPH_DB        = str(Path(__file__).parent.parent / "vectorstore" / "constitution_graph.db")
EMBED_MODEL     = "all-MiniLM-L6-v2"

SOFT_THRESHOLD  = 0.82   # Minimum cosine similarity for a soft edge
MANIFEST_PATH   = Path(__file__).parent / "constitution_manifest.json"


# ── Relationship type inference from context words ─────────────────────────────
# Checked in the 60 chars BEFORE "article N" in the text
RELATIONSHIP_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'notwithstanding',          re.IGNORECASE), 'OVERRIDES'),
    (re.compile(r'subject\s+to',             re.IGNORECASE), 'RESTRICTED_BY'),
    (re.compile(r'save\s+as\s+provided\s+in',re.IGNORECASE), 'EXCEPTION_TO'),
    (re.compile(r'as\s+defined\s+in',        re.IGNORECASE), 'DEFINED_BY'),
    (re.compile(r'enforced?\s+(?:by|under)', re.IGNORECASE), 'ENFORCED_BY'),
    (re.compile(r'amend(?:ment)?\s+(?:of|to|under)',
                                             re.IGNORECASE), 'AMENDED_BY'),
    (re.compile(r'in\s+accordance\s+with',   re.IGNORECASE), 'GOVERNED_BY'),
    (re.compile(r'referred?\s+to\s+in',      re.IGNORECASE), 'REFERENCED_BY'),
]

# Cross-reference detector
RE_CROSSREF = re.compile(
    r'\barticles?\s+(\d{1,3}[A-Z]?)(?:\s*(?:,\s*|,?\s+(?:and|or|to)\s+)(\d{1,3}[A-Z]?))*',
    re.IGNORECASE,
)

RE_ART_NO = re.compile(r'\d{1,3}[A-Z]?')


# ── SQLite Setup ───────────────────────────────────────────────────────────────
def init_db(db_path: str) -> sqlite3.Connection:
    """Create the graph database and edges table if not already present."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            source_article  TEXT NOT NULL,
            target_article  TEXT NOT NULL,
            relationship    TEXT NOT NULL,
            confidence      REAL NOT NULL,
            edge_source     TEXT NOT NULL,   -- 'hard' or 'soft'
            PRIMARY KEY (source_article, target_article, relationship)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_source ON edges(source_article)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_target ON edges(target_article)
    """)
    conn.commit()
    return conn


def upsert_edge(
    conn: sqlite3.Connection,
    source: str,
    target: str,
    relationship: str,
    confidence: float,
    edge_source: str,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO edges
           (source_article, target_article, relationship, confidence, edge_source)
           VALUES (?, ?, ?, ?, ?)""",
        (source, target, relationship, confidence, edge_source),
    )


# ── Load Manifest ──────────────────────────────────────────────────────────────
def load_articles_from_manifest() -> Dict[str, Dict]:
    """
    Returns {article_no: {text, part, part_name, heading, cross_refs_str}}.
    Only chunk_index=0 (the first/full chunk) is used for graph building.
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}. Run ingest_constitution.py first."
        )
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8", errors="replace"))
    articles: Dict[str, Dict] = {}
    for entry in data:
        if entry.get("act") != ACT_NAME:
            continue
        art_no = entry["article_no"]
        # Use chunk_index=0 (first/representative chunk) for graph building
        if entry.get("chunk_index", 0) == 0:
            articles[art_no] = {
                "text":        entry.get("chunk_text", ""),
                "part":        entry.get("part", ""),
                "part_name":   entry.get("part_name", ""),
                "heading":     entry.get("article_heading", ""),
                "cross_refs":  entry.get("cross_refs", ""),  # comma-separated string
            }
    return articles


# ── Hard Edge Builder ──────────────────────────────────────────────────────────
def infer_relationship(text: str, match_pos: int) -> str:
    """
    Look at the 60 characters before a cross-reference match to infer
    the relationship type from keyword context.
    """
    context = text[max(0, match_pos - 60): match_pos].lower()
    for pattern, rel_type in RELATIONSHIP_RULES:
        if pattern.search(context):
            return rel_type
    return "CROSS_REFERENCES"


def build_hard_edges(
    articles: Dict[str, Dict],
    conn: sqlite3.Connection,
) -> int:
    """
    Parse cross-references from each article's text and create hard edges.
    Returns count of edges added.
    """
    known_articles: Set[str] = set(articles.keys())
    count = 0

    for src_art, info in articles.items():
        text = info["text"]
        for m in RE_CROSSREF.finditer(text):
            rel = infer_relationship(text, m.start())
            # Extract all article numbers from this cross-reference match
            cited = RE_ART_NO.findall(m.group())
            for cited_art in cited:
                if cited_art == src_art:
                    continue  # skip self-references
                if cited_art not in known_articles:
                    continue  # skip references to omitted/non-existent articles
                upsert_edge(conn, src_art, cited_art, rel, 1.0, "hard")
                count += 1

    conn.commit()
    return count


# ── Soft Edge Builder ──────────────────────────────────────────────────────────
def build_soft_edges(
    articles: Dict[str, Dict],
    conn: sqlite3.Connection,
    threshold: float = SOFT_THRESHOLD,
) -> int:
    """
    Compute cosine similarity between article embeddings WITHIN each Part only.
    Creates soft edges for pairs above the threshold.
    Returns count of edges added.
    """
    model = SentenceTransformer(EMBED_MODEL)

    # Group articles by Part
    parts: Dict[str, List[str]] = {}
    for art_no, info in articles.items():
        part = info["part"]
        parts.setdefault(part, []).append(art_no)

    count = 0

    for part, art_list in parts.items():
        if len(art_list) < 2:
            continue  # nothing to compare

        texts = [articles[a]["text"] for a in art_list]
        embs  = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        # Cosine similarity: since embeddings are L2-normalized, dot product = cosine sim
        sim_matrix = np.dot(embs, embs.T)

        for i, j in combinations(range(len(art_list)), 2):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                src = art_list[i]
                tgt = art_list[j]
                upsert_edge(conn, src, tgt, "RELATED_TO", sim, "soft")
                upsert_edge(conn, tgt, src, "RELATED_TO", sim, "soft")
                count += 2

        print(f"  Part {part:5s}: {len(art_list):3d} articles, "
              f"{count} soft edges so far")

    conn.commit()
    return count


# ── Threshold Validation ───────────────────────────────────────────────────────
def validate_threshold(articles: Dict[str, Dict], thresholds=None) -> None:
    """
    Use hard-edge article pairs as ground truth to find the optimal
    soft-edge threshold. Prints recall at each threshold.
    """
    if thresholds is None:
        thresholds = [0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.90]

    print("\nBuilding embeddings for all articles...")
    model = SentenceTransformer(EMBED_MODEL)
    art_list = list(articles.keys())
    texts    = [articles[a]["text"] for a in art_list]
    embs     = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    idx_map  = {a: i for i, a in enumerate(art_list)}

    # Collect known hard-linked pairs (same part only, for fair comparison)
    hard_pairs: Set[Tuple[str, str]] = set()
    for src, info in articles.items():
        cross_refs_str = info.get("cross_refs", "")
        if not cross_refs_str:
            continue
        for tgt in cross_refs_str.split(","):
            tgt = tgt.strip()
            if tgt and tgt != src and tgt in articles:
                if articles[src]["part"] == articles[tgt]["part"]:
                    hard_pairs.add((min(src, tgt), max(src, tgt)))

    if not hard_pairs:
        print("  No same-part hard-linked pairs found for validation.")
        return

    print(f"\n  {len(hard_pairs)} same-part hard-linked pairs (ground truth)\n")
    print(f"  {'Threshold':>10}  {'Recall':>7}  {'# Soft Edges':>12}")
    print("  " + "-" * 34)

    for thresh in thresholds:
        recalled  = 0
        soft_total = 0
        for (a, b) in hard_pairs:
            if a not in idx_map or b not in idx_map:
                continue
            sim = float(np.dot(embs[idx_map[a]], embs[idx_map[b]]))
            if sim >= thresh:
                recalled += 1

        # Count total soft edges at this threshold (same-part only)
        for art_no, info in articles.items():
            part = info["part"]
            same_part = [x for x in articles if articles[x]["part"] == part and x != art_no]
            for other in same_part:
                if art_no < other:
                    sim = float(np.dot(embs[idx_map[art_no]], embs[idx_map[other]]))
                    if sim >= thresh:
                        soft_total += 1

        recall = recalled / len(hard_pairs) * 100 if hard_pairs else 0
        print(f"  {thresh:>10.2f}  {recall:>6.1f}%  {soft_total:>12,}")


# ── Graph Statistics ───────────────────────────────────────────────────────────
def print_stats(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    hard = conn.execute("SELECT COUNT(*) FROM edges WHERE edge_source='hard'").fetchone()[0]
    soft = conn.execute("SELECT COUNT(*) FROM edges WHERE edge_source='soft'").fetchone()[0]

    print(f"\n── Graph Statistics ──")
    print(f"  Total edges : {rows:,}")
    print(f"  Hard edges  : {hard:,}")
    print(f"  Soft edges  : {soft:,}")

    print("\n── Relationship type breakdown (hard edges) ──")
    for rel, cnt in conn.execute(
        "SELECT relationship, COUNT(*) FROM edges WHERE edge_source='hard' "
        "GROUP BY relationship ORDER BY COUNT(*) DESC"
    ):
        print(f"  {rel:25s}: {cnt:4d}")

    print("\n── Most-connected articles (top 10) ──")
    for art, cnt in conn.execute(
        "SELECT source_article, COUNT(*) as c FROM edges "
        "GROUP BY source_article ORDER BY c DESC LIMIT 10"
    ):
        print(f"  Art.{art:5s}: {cnt:3d} outgoing edges")

    print("\n── Sample hard edges ──")
    for src, tgt, rel, conf in conn.execute(
        "SELECT source_article, target_article, relationship, confidence "
        "FROM edges WHERE edge_source='hard' LIMIT 15"
    ):
        print(f"  Art.{src:5s} --{rel:20s}--> Art.{tgt}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Constitution knowledge graph (hard + soft edges)"
    )
    parser.add_argument("--hard-only",  action="store_true", help="Build only hard edges")
    parser.add_argument("--soft-only",  action="store_true", help="Build only soft edges")
    parser.add_argument("--stats",      action="store_true", help="Print stats and exit")
    parser.add_argument("--validate",   action="store_true",
                        help="Validate soft threshold against hard edge ground truth")
    parser.add_argument("--threshold",  type=float, default=SOFT_THRESHOLD,
                        help=f"Cosine similarity threshold for soft edges (default {SOFT_THRESHOLD})")
    parser.add_argument("--db",         default=GRAPH_DB, help="Path to SQLite graph DB")
    args = parser.parse_args()

    conn = init_db(args.db)

    # ── Stats only ─────────────────────────────────────────────────────────────
    if args.stats:
        print_stats(conn)
        conn.close()
        return

    # ── Load articles ──────────────────────────────────────────────────────────
    print("Loading articles from manifest...")
    articles = load_articles_from_manifest()
    print(f"  Loaded {len(articles)} articles")

    # ── Validate threshold ─────────────────────────────────────────────────────
    if args.validate:
        validate_threshold(articles)
        conn.close()
        return

    # ── Clear existing edges before rebuild ────────────────────────────────────
    if not args.soft_only:
        conn.execute("DELETE FROM edges WHERE edge_source='hard'")
        conn.commit()
    if not args.hard_only:
        conn.execute("DELETE FROM edges WHERE edge_source='soft'")
        conn.commit()

    # ── Hard edges ─────────────────────────────────────────────────────────────
    if not args.soft_only:
        print("\nBuilding hard edges (cross-reference extraction)...")
        h = build_hard_edges(articles, conn)
        print(f"  {h:,} hard edges stored")

    # ── Soft edges ─────────────────────────────────────────────────────────────
    if not args.hard_only:
        print(f"\nBuilding soft edges (similarity >= {args.threshold}, within-Part only)...")
        s = build_soft_edges(articles, conn, threshold=args.threshold)
        print(f"  {s:,} soft edges stored")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_stats(conn)
    conn.close()
    print(f"\nGraph saved → {args.db}")


if __name__ == "__main__":
    main()
