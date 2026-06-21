import chromadb
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class StatuteRetriever:
    def __init__(self, chroma_path: str = str(Path(__file__).parent.parent / "vectorstore"), 
                 collection_name: str = "statutes_v1",
                 model_name: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        try:
            self.collection = self.client.get_collection(collection_name)
        except KeyError:
            # Collection corrupted, recreate it
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        self.model = SentenceTransformer(model_name)
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant statute chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        formatted_results = []
        for doc, metadata, distance in zip(
            results['documents'][0], 
            results['metadatas'][0],
            results['distances'][0]
        ):
            formatted_results.append({
                'text': doc,
                'act': metadata['act'],
                'section': metadata['section'],
                'heading': metadata['heading'],
                'page': f"{metadata['page_start']}-{metadata['page_end']}",
                'relevance_score': 1 - distance  # Convert distance to similarity
            })
        
        return formatted_results

if __name__ == "__main__":
    retriever = StatuteRetriever()
    results = retriever.search("provisions for anticipatory bail")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result['act']}] Section {result['section']}")
        print(f"   Score: {result['relevance_score']:.3f}")
        print(f"   {result['text'][:200]}...")