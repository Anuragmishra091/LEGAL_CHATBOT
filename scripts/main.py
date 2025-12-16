from Retrieval_function import StatuteRetriever
import requests
import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()


class LegalChatbot:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",  # ✅ Router-supported model
        provider: str = "huggingface",
    ):
        self.retriever = StatuteRetriever()
        self.model_name = model_name
        self.provider = provider
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

        if provider == "huggingface" and not self.hf_api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment variables!")

        # HF Router OpenAI-compatible endpoint
        self.hf_api_url = "https://router.huggingface.co/v1/chat/completions"

    def _call_huggingface(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.9,
        }

        try:
            response = requests.post(
                self.hf_api_url,
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
                    return f"Error calling HuggingFace API: {e} - {error_detail}"
            except Exception:
                pass
            return f"Error calling HuggingFace API: {e}"

    def ask(self, question: str, n_context: int = 5) -> Dict:
        results = self.retriever.search(question, n_results=n_context)

        if not results:
            return {
                "answer": "No relevant legal provisions found in the database.",
                "sources": [],
                "context_used": 0,
            }

        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {r['act']}, Section {r['section']} - {r['heading']}]\n{r['text']}"
            )
        context = "\n\n".join(context_parts)

        prompt = f"""You are an expert Indian legal assistant specializing in BNS, BNSS, and BSA laws.

Based on the following legal statutes, answer the question:

LEGAL CONTEXT:
{context}

QUESTION: {question}

Instructions:
1. Answer ONLY using information from the context above.
2. Cite specific sections and act names.
3. Use precise legal terminology.
4. If multiple provisions apply, list all.
5. If the context doesn't contain enough information, say so.

ANSWER:"""

        answer = self._call_huggingface(prompt)

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
