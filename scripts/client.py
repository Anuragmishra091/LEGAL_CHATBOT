import chromadb

# Connect to your ChromaDB
client = chromadb.PersistentClient(path="../vectorstore")
collection = client.get_collection("statutes_v1")

# Check collection stats
print(f"Total documents: {collection.count()}")

# Sample query to test
results = collection.query(
    query_texts=["What are the provisions for bail?"],
    n_results=5
)

print("\nSample results:")
for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\n{i+1}. Act: {metadata['act']}, Section: {metadata['section']}")
    print(f"   Text: {doc[:200]}...")