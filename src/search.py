import redis
import chromadb
import pymongo
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
from scipy.spatial.distance import cosine
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



# Using reddis for search
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
chroma_collection = chroma_client.get_or_create_collection(name="pdf_embeddings", metadata={"hnsw:space": "cosine"})
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["pdf_embeddings_db"]
mongo_collection = mongo_db["embeddings"]

INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Important inputs
embedding_model, starter_prompt, llm_model, database = embedding_model_3, starter_prompt_1, ollama_model_3, "chroma"

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


def search_embeddings(query, top_k=3, database= "redis"):
    query_embedding = get_embedding(query, embedding_model)
    
    try:
        # Redis Search
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )
        redis_results = redis_client.ft(INDEX_NAME).search(q, query_params={"vec": np.array(query_embedding, dtype=np.float32).tobytes()})
        top_redis_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in redis_results.docs
        ][:top_k]
        for result in top_redis_results:
            print(f"Redis ---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")
        
        # ChromaDB Search
        chroma_results = chroma_collection.query(query_embeddings=[query_embedding], n_results=5)
        metadata = chroma_results.get("metadatas", [[]])
        distances = chroma_results.get("distances", [[]])
        top_chroma_results = [
            {
                "file": result["file"],
                "page": result["page"],
                "chunk": result["chunk"],
                "similarity": distances[0][i]
            }
            for i, result in enumerate(metadata[0][:top_k])
        ]
        for result in top_chroma_results:
            print(f"ChromaDB ---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")
        
        # MongoDB Search
        mongo_results = mongo_collection.find({}, {"file": 1, "page": 1, "chunk": 1, "embedding": 1})
        scored_results = []

        for doc in mongo_results:
            if "embedding" not in doc or not doc["embedding"]:
                continue 
            
            doc_embedding = np.array(doc["embedding"], dtype=np.float32)
            if doc_embedding.size == 0 or len(doc_embedding) != len(query_embedding):
                continue  

            similarity = 1 - cosine(query_embedding, doc_embedding)
            scored_results.append({
                "file": doc["file"],
                "page": doc["page"],
                "chunk": doc["chunk"],
                "similarity": similarity,
            })

        scored_results.sort(key=lambda x: x["similarity"], reverse=True)
        top_mongo_results = scored_results[:top_k]

        for result in top_mongo_results:
            print(f"MongoDB ---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")
        
        # Return results for the db
        if database == "redis":
            return top_redis_results
        elif database == "chroma":
            return top_chroma_results
        elif database == "mongo":
            return top_mongo_results
        else:
            print("Not redis, chroma, or mongo.")
            return []
    
    except Exception as e:
        print(f"Error: {e}")
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

    print(f"\ncontext_str: {context_str}")

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
        context_results, elapsed_time, mem_usage = time_query(search_embeddings, query, 3, database)

        # Log the results
        log_query_results(query, model_name, elapsed_time, mem_usage)

        # Generate RAG response
        response = generate_rag_response(query, context_results, starter_prompt, llm_model)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search()
