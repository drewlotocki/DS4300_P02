import ollama
import redis
import chromadb
import numpy as np
import os
import fitz
import time
import csv
import sys
from redis.commands.search.query import Query

# Initialize Redis and ChromaDB connections
redis_client = redis.Redis(host="localhost", port=6379, db=0)
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"
CSV_FILE = "benchmark_results.csv"

# Utility function to measure execution time
def time_it(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

# Clear both Redis and ChromaDB stores
def clear_stores():
    redis_client.flushdb()
    try:
        chroma_client.delete_collection(name="pdf_embeddings")
    except:
        pass
    global chroma_collection
    chroma_collection = chroma_client.get_or_create_collection(name="pdf_embeddings", metadata={"hnsw:space": "cosine"})

# Create Redis HNSW index
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass
    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA file TEXT page TEXT chunk TEXT 
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )

# Get embedding vector
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Store embedding in Redis and ChromaDB
# Store embedding in Redis
def store_embedding_redis(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )


# Store embedding in ChromaDB
def store_embedding_chroma(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    chroma_collection.add(
        ids=[key], 
        embeddings=[embedding], 
        metadatas=[{"file": file, "page": page, "chunk": chunk}]
    )


# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return [(page_num, page.get_text()) for page_num, page in enumerate(doc)]

# Split text into chunks
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# Process PDFs and measure ingestion time
def process_pdfs(data_dir):
    total_redis_time, total_chroma_time = 0, 0  # Accumulate times

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            redis_time, chroma_time = 0, 0
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    
                    # Store in Redis
                    _, redis_elapsed = time_it(store_embedding_redis, file_name, str(page_num), str(chunk), embedding)
                    redis_time += redis_elapsed
                    
                    # Store in ChromaDB
                    _, chroma_elapsed = time_it(store_embedding_chroma, file_name, str(page_num), str(chunk), embedding)
                    chroma_time += chroma_elapsed
            
            print(f"Processed {file_name} - Redis: {redis_time:.2f}s, ChromaDB: {chroma_time:.2f}s")
            total_redis_time += redis_time
            total_chroma_time += chroma_time

    return total_redis_time, total_chroma_time

# Measure query time
def query_stores(query_text: str):
    embedding = get_embedding(query_text)
    query_embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

    # Query Redis
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("file", "page", "chunk", "vector_distance")
        .dialect(2)
    )
    res, redis_query_time = time_it(redis_client.ft(INDEX_NAME).search, q, query_params={"vec": query_embedding_bytes})
    
    print("Redis Search Results:")
    for doc in res.docs:
        print(f"File: {doc.file}, Page: {doc.page}\nChunk: {doc.chunk}\nDistance: {doc.vector_distance}\n")
    
    # Query ChromaDB
    chroma_results, chroma_query_time = time_it(chroma_collection.query, query_embeddings=[embedding], n_results=5)
    
    print("ChromaDB Search Results:")
    for doc_id, metadata, distance in zip(
        chroma_results["ids"][0], chroma_results["metadatas"][0], chroma_results["distances"][0]
    ):
        print(f"File: {metadata['file']}, Page: {metadata['page']}\nChunk: {metadata['chunk']}\nDistance: {distance}\n")
    
    return redis_query_time, chroma_query_time

# Get storage space used
def get_storage_space():
    # Get Redis memory usage
    redis_size = redis_client.info("memory")["used_memory"] / (1024 * 1024)
    

    # Get Chroma memory usage
    total_size = 0
    items = chroma_client.get_collection("pdf_embeddings").get(include=["documents", "embeddings", "metadatas"])
    if "embeddings" in items:
        for embedding in items["embeddings"]:
            if embedding is not None:
                total_size += sys.getsizeof(embedding) + sum(sys.getsizeof(x) for x in embedding)

    # Calculate size for metadata
    if "metadatas" in items:
        for metadata in items["metadatas"]:
            if metadata is not None:
                total_size += sys.getsizeof(metadata)
                for key, value in metadata.items():
                    total_size += sys.getsizeof(key) + sys.getsizeof(value)

    # Calculate size for documents
    if "documents" in items:
        for doc_id in items["documents"]:
            if doc_id is not None:
                total_size += sys.getsizeof(doc_id)
    # Convert to MB
    chroma_size = total_size / (1024 * 1024)

    return redis_size, chroma_size

# Write benchmark results to CSV
def write_results(redis_time, chroma_time, redis_query, chroma_query, redis_size, chroma_size):
    # get the number of embeddings for each store
    num_embeddings = chroma_client.get_collection("pdf_embeddings").count()
    num_keys = redis_client.dbsize()
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Redis Ingestion", "ChromaDB Ingestion", "Redis Query", "ChromaDB Query", "Redis Storage (MB)", "ChromaDB Storage (MB)","Number of embeddings in Redis","Number of embeddings Chroma"])
        writer.writerow([redis_time, chroma_time, redis_query, chroma_query, redis_size, chroma_size, num_embeddings, num_keys])

# Main function
def main():
    clear_stores()
    create_hnsw_index()
    redis_time, chroma_time = process_pdfs("../data/")
    redis_query, chroma_query = query_stores("What is the capital of France?")
    redis_size, chroma_size = get_storage_space()
    write_results(redis_time, chroma_time, redis_query, chroma_query, redis_size, chroma_size)
    print("\n---Done processing PDFs and benchmarking---\n")

if __name__ == "__main__":
    main()
