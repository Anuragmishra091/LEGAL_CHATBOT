"""
Evaluation metrics for the Legal Chatbot RAG system.

Metrics implemented:
  Level 1 (Automated — no ground truth required):
    1. Faithfulness      — Is every claim in the answer supported by the context?
    2. Answer Relevancy  — Does the answer address the question? (embedding similarity)
    3. Context Relevancy — Are retrieved chunks relevant to the query?
    4. Citation Accuracy  — Are cited sections/articles verifiable in the context?

  Level 2 (Ground truth comparison):
    5. Keyword Recall     — % of expected keywords present in the answer
    6. Source Accuracy    — Did retriever find the expected source documents?
    7. Semantic Correctness — Embedding similarity between answer & expected answer

All metrics return a score between 0.0 and 1.0.
"""

import re
import os
import requests
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Shared embedding model (same as retriever)
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _cosine_sim(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts."""
    model = _get_model()
    embs = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — Automated Metrics (no ground truth needed)
# ══════════════════════════════════════════════════════════════════════════════

def faithfulness(answer: str, context: str) -> float:
    """
    Measures if the answer is grounded in the provided context (no hallucination).

    Method: Use LLM-as-judge to check if each claim in the answer can be found
    in the context. Falls back to embedding overlap if API unavailable.

    Returns: 0.0 (completely hallucinated) to 1.0 (fully grounded)
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        # Fallback: embedding-based faithfulness
        return _embedding_faithfulness(answer, context)

    prompt = f"""You are a legal accuracy evaluator. Given a CONTEXT (retrieved legal text) and an ANSWER, determine what fraction of claims in the ANSWER are supported by the CONTEXT.

CONTEXT:
{context[:3000]}

ANSWER:
{answer}

Instructions:
1. List each factual claim in the ANSWER.
2. For each claim, check if it is supported by the CONTEXT.
3. Return ONLY a single decimal number between 0.0 and 1.0 representing the fraction of claims that are supported.
   - 1.0 = all claims are supported by context
   - 0.0 = no claims are supported
   
Return ONLY the number, nothing else."""

    try:
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "meta/llama-3.3-70b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout=30,
        )
        text = response.json()["choices"][0]["message"]["content"].strip()
        score = float(re.search(r"[0-9]+\.?[0-9]*", text).group())
        return min(max(score, 0.0), 1.0)
    except Exception:
        return _embedding_faithfulness(answer, context)


def _embedding_faithfulness(answer: str, context: str) -> float:
    """
    Fallback faithfulness using embedding similarity.
    Splits answer into sentences and checks each against context.
    """
    model = _get_model()
    sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    ctx_emb = model.encode([context], normalize_embeddings=True)[0]
    sent_embs = model.encode(sentences, normalize_embeddings=True)

    similarities = np.dot(sent_embs, ctx_emb)
    # A sentence is "grounded" if similarity > 0.4
    grounded = sum(1 for s in similarities if s > 0.4)
    return grounded / len(sentences)


def answer_relevancy(question: str, answer: str) -> float:
    """
    Measures if the answer is relevant to the question asked.
    Uses embedding cosine similarity between question and answer.

    Returns: 0.0 (completely irrelevant) to 1.0 (perfectly relevant)
    """
    if not answer or "no relevant" in answer.lower():
        return 0.0
    return max(0.0, _cosine_sim(question, answer))


def context_relevancy(question: str, sources: List[Dict]) -> float:
    """
    Measures if the retrieved context chunks are relevant to the query.
    Avg cosine similarity between the question and each retrieved chunk.

    Returns: 0.0 to 1.0
    """
    if not sources:
        return 0.0

    model = _get_model()
    q_emb = model.encode([question], normalize_embeddings=True)[0]
    texts = [s.get("text", "") for s in sources if s.get("text")]
    if not texts:
        return 0.0

    src_embs = model.encode(texts, normalize_embeddings=True)
    sims = np.dot(src_embs, q_emb)
    return float(np.mean(sims))


def citation_accuracy(answer: str, sources: List[Dict]) -> float:
    """
    Checks if legal citations in the answer (Article N, Section N) actually
    exist in the retrieved sources.

    Returns: fraction of cited items found in sources (0.0 to 1.0)
    """
    # Extract citations from answer
    cited = set()
    for m in re.finditer(r'(?:article|art\.?)\s*(\d{1,3}[A-Z]?)', answer, re.I):
        cited.add(f"Art.{m.group(1)}")
    for m in re.finditer(r'(?:section)\s*(\d{1,4})', answer, re.I):
        cited.add(f"Sec.{m.group(1)}")

    if not cited:
        return 1.0  # No citations made → not penalised

    # Check which citations exist in sources
    source_text = " ".join(s.get("text", "") for s in sources)
    source_sections = set()
    for s in sources:
        sec = s.get("section", "")
        if sec:
            source_sections.add(sec)
        heading = s.get("heading", "")
        if heading:
            source_sections.add(heading)

    found = 0
    for citation in cited:
        # Check if the cited number appears in source text
        num = re.search(r'\d+[A-Z]?', citation).group()
        if num in source_text or citation in source_text:
            found += 1
        elif any(num in sec for sec in source_sections):
            found += 1

    return found / len(cited)


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — Ground Truth Comparison Metrics
# ══════════════════════════════════════════════════════════════════════════════

def keyword_recall(answer: str, expected_keywords: List[str]) -> float:
    """
    Fraction of expected keywords found in the generated answer.
    Case-insensitive matching.

    Returns: 0.0 to 1.0
    """
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def source_accuracy(retrieved_sources: List[Dict], expected_sources: List[str]) -> float:
    """
    Checks if the retriever found documents from the expected sources.
    Matches against 'act', 'section', or 'article_no' fields.

    Returns: fraction of expected sources found (0.0 to 1.0)
    """
    if not expected_sources:
        return 1.0

    # Build a set of identifiers from retrieved sources
    retrieved_ids = set()
    for s in retrieved_sources:
        retrieved_ids.add(s.get("act", ""))
        retrieved_ids.add(s.get("section", ""))
        article = s.get("article_no", "")
        if article:
            retrieved_ids.add(f"Art.{article}")

    found = 0
    for expected in expected_sources:
        if expected in retrieved_ids:
            found += 1
        elif any(expected in rid for rid in retrieved_ids):
            found += 1

    return found / len(expected_sources)


def semantic_correctness(answer: str, expected_answer: str) -> float:
    """
    Embedding cosine similarity between generated answer and expected answer.
    Measures if the answer conveys the same meaning.

    Returns: 0.0 to 1.0
    """
    if not answer or not expected_answer:
        return 0.0
    return max(0.0, _cosine_sim(answer, expected_answer))


# ══════════════════════════════════════════════════════════════════════════════
# Aggregated evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_single(
    question: str,
    answer: str,
    sources: List[Dict],
    expected_answer: str = None,
    expected_keywords: List[str] = None,
    expected_sources: List[str] = None,
) -> Dict[str, float]:
    """
    Run all metrics on a single QA pair.
    Returns dict of metric_name → score.
    """
    # Build context string from sources
    context = "\n".join(s.get("text", "") for s in sources)

    results = {}

    # Level 1 — automated
    results["faithfulness"]      = faithfulness(answer, context)
    results["answer_relevancy"]  = answer_relevancy(question, answer)
    results["context_relevancy"] = context_relevancy(question, sources)
    results["citation_accuracy"] = citation_accuracy(answer, sources)

    # Level 2 — ground truth (if provided)
    if expected_keywords:
        results["keyword_recall"] = keyword_recall(answer, expected_keywords)
    if expected_sources:
        results["source_accuracy"] = source_accuracy(sources, expected_sources)
    if expected_answer:
        results["semantic_correctness"] = semantic_correctness(answer, expected_answer)

    return results
