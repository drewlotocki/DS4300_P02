import redis
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
import psutil
import time
import csv
import os


# Initialize models
#embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
#embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = "nomic-embed-text"

# Using reddis for search
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"
CSV_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"{embedding_model}_query_benchmark_results.csv"))

# def cosine_similarity(vec1, vec2):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embedding(text: str, embedding_model) -> list:
    if isinstance(embedding_model, str):
        response = ollama.embeddings(model=embedding_model, prompt=text)
        return response["embedding"]
    else:
        return embedding_model.encode(text).tolist()

# ollama model
"""def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]"""


def time_query(query_func, *args, **kwargs):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024) 

    start_time = time.time()
    result = query_func(*args, **kwargs)
    elapsed_time = time.time() - start_time

    mem_after = process.memory_info().rss / (1024 * 1024)  
    mem_usage = mem_after - mem_before

    return result, elapsed_time, mem_usage


def search_embeddings(query, top_k=3):

    query_embedding = get_embedding(query)

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)

        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into the expected format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):

    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    print(f"context_str: {context_str}")

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'No Answer Found'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model="llama3.2:latest", messages=[{"role": "user", "content": prompt}]
    )
    # mistral:latest
    return response["message"]["content"]


def log_query_results(query, model_name, elapsed_time, mem_usage):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Query", "Model", "Time (s)", "Memory Usage (MB)"])
        writer.writerow([query, model_name, elapsed_time, mem_usage])


def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Time the search query
        context_results, elapsed_time, mem_usage = time_query(search_embeddings, query)

        # Log the results
        log_query_results(query, "InstructorXL", elapsed_time, mem_usage)

        """# Search for relevant embeddings
        context_results = search_embeddings(query)"""

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search()
