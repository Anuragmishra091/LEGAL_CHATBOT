"""
Constitution Graph-RAG Retriever.

Flow:
  1. Semantic search (ChromaDB) → seed article chunks
  2. Graph expansion (SQLite) → 1-hop neighbors of each seed
  3. Fetch neighbor chunks from ChromaDB
  4. Merge, deduplicate, re-rank by relevance
  5. Return enriched result list with relationship context for the LLM

Backward-compatible with StatuteRetriever: results have the same
keys (text, act, section, heading, relevance_score) plus extra
graph-specific keys (article_no, part, relationship, edge_source).
"""

import sqlite3
from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_PATH     = str(Path(__file__).parent.parent / "vectorstore")
COLLECTION_NAME = "statutes_v1"
ACT_NAME        = "CONSTITUTION_OF_INDIA"
GRAPH_DB        = str(Path(__file__).parent.parent / "vectorstore" / "constitution_graph.db")
EMBED_MODEL     = "all-MiniLM-L6-v2"

# How many graph neighbors to include per seed article
MAX_NEIGHBORS_PER_SEED = 3


class ConstitutionGraphRetriever:
    """
    Graph-aware retriever for the Constitution of India.

    Combines ChromaDB semantic search with the knowledge graph to retrieve
    not just the most similar articles but also their constitutionally linked
    neighbors, giving the LLM a complete legal context.
    """

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        graph_db_path: str = GRAPH_DB,
        model_name: str = EMBED_MODEL,
    ):
        # ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        # Embedding model
        self.model = SentenceTransformer(model_name)

        # Knowledge graph
        self.graph_db_path = graph_db_path
        self._graph_available = Path(graph_db_path).exists()
        if not self._graph_available:
            print(
                f"[ConstitutionGraphRetriever] WARNING: Graph DB not found at "
                f"{graph_db_path}. Run constitution_graph.py first. "
                f"Falling back to semantic-only retrieval."
            )

    # ── Graph helpers ──────────────────────────────────────────────────────────
    def _get_neighbors(
        self,
        article_nos: List[str],
        max_per_seed: int = MAX_NEIGHBORS_PER_SEED,
    ) -> List[Dict]:
        """
        Query the graph for neighbors of the given article numbers.
        Returns list of {article_no, relationship, confidence, edge_source}.
        Hard edges are returned first (sorted by confidence desc).
        """
        if not self._graph_available or not article_nos:
            return []

        placeholders = ",".join("?" * len(article_nos))
        conn = sqlite3.connect(self.graph_db_path)
        rows = conn.execute(
            f"""
            SELECT source_article, target_article, relationship,
                   confidence, edge_source
            FROM edges
            WHERE source_article IN ({placeholders})
               OR target_article IN ({placeholders})
            ORDER BY
                CASE edge_source WHEN 'hard' THEN 0 ELSE 1 END,
                confidence DESC
            """,
            article_nos + article_nos,
        ).fetchall()
        conn.close()

        neighbors: Dict[str, Dict] = {}
        seed_set = set(article_nos)

        for src, tgt, rel, conf, esrc in rows:
            # The neighbor is whichever end is NOT the seed
            neighbor = tgt if src in seed_set else src
            if neighbor in seed_set:
                continue  # don't add seeds as neighbors
            if neighbor not in neighbors:
                neighbors[neighbor] = {
                    "article_no":   neighbor,
                    "relationship": rel,
                    "confidence":   conf,
                    "edge_source":  esrc,
                    "via_seed":     src if src in seed_set else tgt,
                }
            # Keep the highest-confidence / most specific edge
            elif conf > neighbors[neighbor]["confidence"]:
                neighbors[neighbor].update(
                    relationship=rel, confidence=conf, edge_source=esrc
                )

        # Limit per seed: sort by (hard first, confidence), take top N overall
        sorted_neighbors = sorted(
            neighbors.values(),
            key=lambda x: (0 if x["edge_source"] == "hard" else 1, -x["confidence"]),
        )
        return sorted_neighbors[: max_per_seed * len(article_nos)]

    # ── Chunk fetcher ──────────────────────────────────────────────────────────
    def _fetch_article_chunks(self, article_nos: List[str]) -> List[Dict]:
        """
        Retrieve chunk_index=0 (representative chunk) for each article number
        from ChromaDB using metadata filter.
        """
        results = []
        for art_no in article_nos:
            try:
                res = self.collection.get(
                    where={
                        "$and": [
                            {"act":         {"$eq": ACT_NAME}},
                            {"article_no":  {"$eq": art_no}},
                            {"chunk_index": {"$eq": 0}},
                        ]
                    },
                    include=["documents", "metadatas"],
                )
                if res["documents"]:
                    doc  = res["documents"][0]
                    meta = res["metadatas"][0]
                    results.append({
                        "text":          doc,
                        "act":           ACT_NAME,
                        "article_no":    art_no,
                        "section":       meta.get("section", f"Art.{art_no}"),
                        "heading":       meta.get("heading", ""),
                        "part":          meta.get("part", ""),
                        "part_name":     meta.get("part_name", ""),
                        "page":          f"{meta.get('page_start', 0)}-{meta.get('page_end', 0)}",
                        "relevance_score": 0.0,  # filled in by re-rank
                        "relationship":  None,
                        "edge_source":   None,
                        "is_neighbor":   True,
                    })
            except Exception:
                pass
        return results

    # ── Re-rank ────────────────────────────────────────────────────────────────
    def _rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Score all candidates (seeds + neighbors) against the query embedding.
        Updates relevance_score in-place.  Neighbors get a small penalty (×0.85)
        to keep seeds at the top while still surfacing relevant neighbors.
        """
        if not candidates:
            return candidates

        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        texts     = [c["text"] for c in candidates]
        embs      = self.model.encode(texts, show_progress_bar=False,
                                      normalize_embeddings=True)
        import numpy as np
        sims = np.dot(embs, query_emb).tolist()

        for candidate, sim in zip(candidates, sims):
            penalty = 0.85 if candidate.get("is_neighbor") else 1.0
            candidate["relevance_score"] = round(float(sim) * penalty, 4)

        return sorted(candidates, key=lambda x: x["relevance_score"], reverse=True)

    # ── Main search ───────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        n_results: int = 5,
        expand_graph: bool = True,
    ) -> List[Dict]:
        """
        Search for relevant Constitution articles using Graph-RAG.

        Parameters
        ----------
        query        : User question / search string.
        n_results    : Number of results to return (seeds + neighbors combined).
        expand_graph : If False, behaves like a plain semantic search.

        Returns
        -------
        List of result dicts, each with keys:
          text, act, section, heading, relevance_score, part, part_name,
          article_no, page, relationship, edge_source, is_neighbor
        """
        # ── Step 1: semantic seed retrieval ────────────────────────────────────
        seed_count = max(3, n_results)  # fetch more seeds to allow graph expansion
        raw = self.collection.query(
            query_texts=[query],
            n_results=min(seed_count, self.collection.count()),
            where={"act": {"$eq": ACT_NAME}},
        )

        seen_articles: set = set()
        seeds: List[Dict] = []

        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            art_no = meta.get("article_no", "")
            if art_no in seen_articles:
                continue
            seen_articles.add(art_no)
            seeds.append({
                "text":           doc,
                "act":            ACT_NAME,
                "article_no":     art_no,
                "section":        meta.get("section", f"Art.{art_no}"),
                "heading":        meta.get("heading", ""),
                "part":           meta.get("part", ""),
                "part_name":      meta.get("part_name", ""),
                "page":           f"{meta.get('page_start',0)}-{meta.get('page_end',0)}",
                "relevance_score": round(1 - float(dist), 4),
                "relationship":   "SEMANTIC_MATCH",
                "edge_source":    "semantic",
                "is_neighbor":    False,
            })

        if not expand_graph or not self._graph_available:
            return seeds[:n_results]

        # ── Step 2: graph expansion ────────────────────────────────────────────
        seed_article_nos = [s["article_no"] for s in seeds if s["article_no"]]
        neighbors_meta   = self._get_neighbors(seed_article_nos)
        neighbor_art_nos = [
            n["article_no"] for n in neighbors_meta
            if n["article_no"] not in seen_articles
        ]

        # ── Step 3: fetch neighbor chunks from ChromaDB ────────────────────────
        neighbor_chunks = self._fetch_article_chunks(neighbor_art_nos)

        # Attach graph relationship info to neighbor chunks
        neighbor_rel_map = {n["article_no"]: n for n in neighbors_meta}
        for chunk in neighbor_chunks:
            art_no = chunk["article_no"]
            if art_no in neighbor_rel_map:
                chunk["relationship"] = neighbor_rel_map[art_no]["relationship"]
                chunk["edge_source"]  = neighbor_rel_map[art_no]["edge_source"]
                chunk["via_seed"]     = neighbor_rel_map[art_no].get("via_seed", "")
            seen_articles.add(art_no)

        # ── Step 4: merge + re-rank ────────────────────────────────────────────
        all_candidates = seeds + neighbor_chunks
        ranked         = self._rerank(query, all_candidates)

        return ranked[:n_results]


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = ConstitutionGraphRetriever()

    queries = [
        "Can fundamental rights be suspended during emergency?",
        "What is the right to equality?",
        "Amendment procedure of the Constitution",
    ]

    for q in queries:
        print(f"\n{'='*70}")
        print(f"Query: {q}")
        print("=" * 70)
        results = retriever.search(q, n_results=5)
        for i, r in enumerate(results, 1):
            neighbor_tag = f" [neighbor via Art.{r.get('via_seed','')}]" \
                           if r.get("is_neighbor") else " [seed]"
            rel = r.get("relationship", "")
            print(
                f"  {i}. Art.{r['article_no']:5s} | "
                f"Part {r['part']:5s} | "
                f"Score {r['relevance_score']:.3f} | "
                f"{rel:22s}{neighbor_tag}"
            )
            print(f"     {r['heading']}")
