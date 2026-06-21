from Retrieval_function import StatuteRetriever
from graph_retriever import ConstitutionGraphRetriever
import re
import requests
import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

# Keywords that signal a Constitution / fundamental-rights query
# NOTE: 'supreme court' and 'high court' intentionally excluded — they appear
# in judgment queries too. Use more specific constitutional terms instead.
_CONSTITUTION_PATTERNS = re.compile(
    r'\b(constitution|fundamental\s+rights?|article\s+\d+|'
    r'preamble|citizenship|directive\s+principles?|'
    r'writ|habeas\s+corpus|mandamus|certiorari|quo\s+warranto|'
    r'freedom\s+of\s+speech|right\s+to\s+life|'
    r'untouchability|election\s+commission|official\s+language|'
    r'constitutional\s+provision|constitutional\s+amendment|'
    r'part\s+III|part\s+IV|part\s+XVIII|part\s+XX)\b',
    re.IGNORECASE,
)

# Keywords that signal a judgment/case-law query — takes priority over constitution routing
_JUDGMENT_PATTERNS = re.compile(
    r'\b(judgment|judgement|ruling|ruled|decided|case\s+law|verdict|'
    r'\bvs\b|\.\s+vs\s+\.| v\.\s|petitioner|respondent|bench|'
    r'supreme\s+court\s+(decided|ruled|held|said|judgment|case|order)|'
    r'high\s+court\s+(decided|ruled|held)|'
    r'bhopal|union\s+carbide|pucl|cauvery|'
    r'what\s+did\s+the\s+(supreme|high)\s+court)\b',
    re.IGNORECASE,
)


def _is_constitution_query(question: str) -> bool:
    """Return True if the question is about the Constitution of India.
    Judgment/case-law queries take priority even if constitutional terms appear.
    """
    if _JUDGMENT_PATTERNS.search(question):
        return False
    return bool(_CONSTITUTION_PATTERNS.search(question))


class LegalChatbot:
    def __init__(
        self,
        model_name: str = "meta/llama-3.3-70b-instruct",
        provider: str = "nvidia",
    ):
        self.retriever        = StatuteRetriever()
        self.graph_retriever  = ConstitutionGraphRetriever()
        self.model_name = model_name
        self.provider = provider
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")

        if not self.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment variables!")

        # NVIDIA NIM OpenAI-compatible endpoint
        self.nvidia_api_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    def _call_nvidia(self, messages: list) -> str:
        headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.9,
        }

        try:
            response = requests.post(
                self.nvidia_api_url,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            try:
                if "response" in locals() and response is not None:
                    error_detail = response.json()
                    return f"Error calling NVIDIA API: {e} - {error_detail}"
            except Exception:
                pass
            return f"Error calling NVIDIA API: {e}"

    def ask(self, question: str, chat_history: list = None, n_context: int = 5) -> Dict:
        use_graph = _is_constitution_query(question)

        if use_graph:
            results = self.graph_retriever.search(question, n_results=n_context)
        else:
            results = self.retriever.search(question, n_results=n_context)

        if not results:
            return {
                "answer": "No relevant legal provisions found in the database.",
                "sources": [],
                "context_used": 0,
            }

        context_parts = []
        for i, r in enumerate(results, 1):
            if use_graph and r.get("article_no"):
                # Constitution result: include Part, Article, and relationship context
                rel = r.get("relationship", "")
                edge = r.get("edge_source", "")
                neighbor_note = ""
                if r.get("is_neighbor") and rel and rel != "SEMANTIC_MATCH":
                    certainty = "explicit cross-reference" if edge == "hard" \
                                else "thematically related"
                    neighbor_note = f" [Related to Article {r.get('via_seed','')}: {rel} — {certainty}]"
                header = (
                    f"[Source {i}: Constitution of India, "
                    f"Part {r.get('part','')} — {r.get('part_name','')}, "
                    f"Article {r.get('article_no','')} — {r['heading']}"
                    f"{neighbor_note}]"
                )
            else:
                header = f"[Source {i}: {r['act']}, Section {r['section']} - {r['heading']}]"
            context_parts.append(f"{header}\n{r['text']}")

        context = "\n\n".join(context_parts)

        if use_graph:
            law_scope = (
                "You are an expert Indian legal assistant specializing in the "
                "Constitution of India, BNS, BNSS, and BSA laws."
            )
            extra_instructions = (
                "7. For Constitution queries: cite the specific Article number and Part.\n"
                "8. Sources marked '[Related to Article X: ...]' are constitutionally "
                "linked articles — use them to give a complete answer.\n"
                "9. Distinguish between explicit cross-references (certain) and "
                "thematically related articles (probable)."
            )
        else:
            law_scope = (
                "You are an expert Indian legal assistant specializing in "
                "BNS, BNSS, and BSA laws."
            )
            extra_instructions = ""

        system_message = {
            "role": "system",
            "content": (
                f"{law_scope}\n\n"
                "Use the following retrieved legal statutes to answer questions:\n\n"
                f"LEGAL CONTEXT:\n{context}\n\n"
                "Instructions:\n"
                "1. Answer ONLY using information from the context above.\n"
                "2. Cite specific sections/articles and act names.\n"
                "3. Use precise legal terminology.\n"
                "4. If multiple provisions apply, list all.\n"
                "5. If the context doesn't contain enough information, say so.\n"
                "6. Use the conversation history to understand follow-up questions "
                "and resolve pronouns or references to earlier topics.\n"
                f"{extra_instructions}"
            ),
        }

        messages = [system_message]

        # Append prior conversation turns for memory
        if chat_history:
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Append the current user question
        messages.append({"role": "user", "content": question})

        answer = self._call_nvidia(messages)

        return {
            "answer": answer,
            "sources": results,
            "context_used": len(results),
        }


if __name__ == "__main__":
    chatbot = LegalChatbot()
    result = chatbot.ask("What are the provisions for bail under BNSS?")
    print("Answer:\n", result["answer"])
    print("\nContext used:", result["context_used"])
