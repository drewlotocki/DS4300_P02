import redis
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
import psutil
import time
import csv
import os

# Starter Prompt
starter_prompt_1 = """You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'No Answer Found'."""

starter_prompt_2 = """You are a powerful Data Engineering AI.  
Use the provided context to generate precise and relevant insights. If the data is insufficient, reply 'No Answer Found'."""  

# Initialize models
embedding_model_1 = "nomic-embed-text"
embedding_model_2 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedding_model_3 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# LLM Model
ollama_model_1 = "llama3.2:latest"
ollama_model_2 = "mistral:latest"
ollama_model_3 = "dolphin-phi:latest"

# Important inputs
embedding_model, starter_prompt, llm_model = embedding_model_3, starter_prompt_1, ollama_model_3

# Using reddis for search
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# Build csv file path
model_name = (embedding_model if isinstance(embedding_model, str) else embedding_model._modules['0'].auto_model.config._name_or_path)
CSV_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"{model_name[len('sentence-transformers/'):]}_query_benchmark_results.csv"))
VECTOR_DIM = len(embedding_model.encode("test")) if not isinstance(embedding_model, str) else 768


# Get the embedding for a given text
def get_embedding(text: str, embedding_model) -> list:
    if isinstance(embedding_model, str):
        response = ollama.embeddings(model=embedding_model, prompt=text)
        return response["embedding"]
    else:
        return embedding_model.encode(text).tolist()


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

    query_embedding = get_embedding(query, embedding_model)

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


def generate_rag_response(query, context_results, starter_prompt, llm_model):

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
    prompt = f"""{starter_prompt}

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model= llm_model, messages=[{"role": "user", "content": prompt}]
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
        log_query_results(query, model_name, elapsed_time, mem_usage)

        """ Search for relevant embeddings
        context_results = search_embeddings(query)"""

        # Generate RAG response
        response = generate_rag_response(query, context_results, starter_prompt, llm_model)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search()
