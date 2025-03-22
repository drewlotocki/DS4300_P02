
import ollama
import redis
import chromadb
import numpy as np
import pymongo
import os
import fitz
import time
import csv
import sys
import string
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

# Initialize Redis and ChromaDB connections
redis_client = redis.Redis(host="localhost", port=6379, db=0)
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["pdf_embeddings_db"]
mongo_collection = mongo_db["embeddings"]

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"
CSV_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "ingest_benchmark_results.csv"))

# Initialize models
embedding_model_1 = "nomic-embed-text"
embedding_model_2 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedding_model_3 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
 

# Utility function to measure execution time
def time_it(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

# Clear both Redis, Mongodb, and ChromaDB stores
def clear_stores():
    redis_client.flushdb()
    mongo_collection.delete_many({})
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

# Store embedding in Redis, ChromaDB and, MongoDb

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
def store_embedding_chroma(chroma_collection,file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    chroma_collection.add(
        ids=[key], 
        embeddings=[embedding], 
        metadatas=[{"file": file, "page": page, "chunk": chunk}]
    )
# Store embedding in MongoDB
def store_embedding_mongo(mongo_collection, file: str, page: str, chunk: str, embedding: list):
    mongo_collection.insert_one({
        "file": file,
        "page": page,
        "chunk": chunk,
        "embedding": embedding
    })

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return [(page_num, page.get_text()) for page_num, page in enumerate(doc)]

# Split text into chunks
def split_text_into_chunks(text, chunk_size=200, overlap=50, white_space=False):
    if white_space:
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())   
    words = text.split() 
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# Process PDFs and measure ingestion time
def process_pdfs(data_dir, embedding_model,chunk_size=200, overlap=50, white_space=False):
    total_redis_time, total_chroma_time, total_mongo_time = 0, 0, 0

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            redis_time, chroma_time, mongo_time = 0, 0, 0
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size, overlap, white_space)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, embedding_model)
                    
                    _, redis_elapsed = time_it(store_embedding_redis, file_name, str(page_num), str(chunk), embedding)
                    redis_time += redis_elapsed
                    
                    _, chroma_elapsed = time_it(store_embedding_chroma, chroma_collection, file_name, str(page_num), str(chunk), embedding)
                    chroma_time += chroma_elapsed
                    
                    _, mongo_elapsed = time_it(store_embedding_mongo, mongo_collection, file_name, str(page_num), str(chunk), embedding)
                    mongo_time += mongo_elapsed
            
            print(f"Processed {file_name} - Redis: {redis_time:.2f}s, ChromaDB: {chroma_time:.2f}s, MongoDB: {mongo_time:.2f}s")
            total_redis_time += redis_time
            total_chroma_time += chroma_time
            total_mongo_time += mongo_time
    return total_redis_time, total_chroma_time, total_mongo_time

# Measure query time
def query_stores(query_text: str, embedding_model):
    embedding = get_embedding(query_text, embedding_model)
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
        print(f"File: {doc.file}, Page: {doc.page}\nChunk: {doc.chunk}\nDistance: {float(doc.vector_distance):.4f}\n")

    # Query ChromaDB
    chroma_results, chroma_query_time = time_it(chroma_collection.query, query_embeddings=[embedding], n_results=5)

    print("ChromaDB Search Results:")
    for i, metadata in enumerate(chroma_results["metadatas"][0][:5]):
        print(f"File: {metadata['file']}, Page: {metadata['page']}\nChunk: {metadata['chunk']}\nDistance: {chroma_results['distances'][0][i]:.4f}\n")

    # Query MongoDB 
    mongo_query_time, mongo_results = 0, []
    all_docs = list(mongo_collection.find({}, {"file": 1, "page": 1, "chunk": 1, "embedding": 1}))
    if all_docs:
        start_time = time.time()
        query_vector = np.array(embedding, dtype=np.float32)

        for doc in all_docs:
            stored_embedding = np.array(doc["embedding"], dtype=np.float32) if isinstance(doc["embedding"], list) else np.frombuffer(doc["embedding"], dtype=np.float32)
            similarity = np.dot(query_vector, stored_embedding) / (np.linalg.norm(query_vector) * np.linalg.norm(stored_embedding))
            mongo_results.append((doc["file"], doc["page"], doc["chunk"], similarity))

        mongo_results.sort(key=lambda x: x[3], reverse=True)
        mongo_query_time = time.time() - start_time

        print("MongoDB Search Results:")
        for file, page, chunk, sim in mongo_results[:5]:
            print(f"File: {file}, Page: {page}\nChunk: {chunk}\nSimilarity: {(1-sim):.4f}\n")

    return redis_query_time, chroma_query_time, mongo_query_time

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
    
    if "metadatas" in items:
        for metadata in items["metadatas"]:
            if metadata is not None:
                total_size += sys.getsizeof(metadata)
                for key, value in metadata.items():
                    total_size += sys.getsizeof(key) + sys.getsizeof(value)
    
    if "documents" in items:
        for doc_id in items["documents"]:
            if doc_id is not None:
                total_size += sys.getsizeof(doc_id)
    
    chroma_size = total_size / (1024 * 1024)
    
    # Get MongoDB memory usage
    mongo_size = 0
    db_list = ['admin', 'config', 'local', 'pdf_embeddings_db', 'vector_db']
    for db_name in db_list:
        db = mongo_client.get_database(db_name)
        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            for doc in collection.find():
                mongo_size += sys.getsizeof(doc)
                for key, value in doc.items():
                    mongo_size += sys.getsizeof(key) + sys.getsizeof(value)
    
    mongo_size = mongo_size / (1024 * 1024)
    
    return redis_size, chroma_size, mongo_size

# Write benchmark results to CSV
def write_results(redis_time, chroma_time, mongo_time, redis_query, chroma_query, mongo_query, redis_size, chroma_size, mongo_size, chunk_size, overlap, white_space, embedding_model):
    # get the number of embeddings
    num_embeddings = chroma_client.get_collection("pdf_embeddings").count()
    num_keys = redis_client.dbsize()
    num_docs = mongo_client['pdf_embeddings_db']['embeddings'].count_documents({})
    
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Redis Ingestion", "ChromaDB Ingestion", "MongoDB Ingestion", 
                "Redis Query", "ChromaDB Query", "MongoDB Query", 
                "Redis Storage (MB)", "ChromaDB Storage (MB)", "MongoDB Storage (MB)",
                "Number of embeddings in Redis", "Number of embeddings in Chroma", "Number of embeddings in MongoDB", 
                "Chunk Size", "Overlap", "White space and Punctuation removal", "Embedding Model"
            ])
        writer.writerow([
            redis_time, chroma_time, mongo_time, 
            redis_query, chroma_query, mongo_query, 
            redis_size, chroma_size, mongo_size, 
            num_keys, num_embeddings, num_docs, 
            chunk_size, overlap, white_space, embedding_model
        ])

# Main function
def main():
    # Define the inputs
    chunk_size, overlap, white_space, embedding_model = 200, 50, True, embedding_model_3
    
    # Clear stores and create HNSW index
    clear_stores()
    create_hnsw_index()
    
    # Time ingestion and querying
    redis_time, chroma_time, mongo_time = process_pdfs("../data/", embedding_model, chunk_size, overlap, white_space)
    redis_query, chroma_query, mongo_query = query_stores("What is a vector database?", embedding_model)
    redis_size, chroma_size, mongo_size = get_storage_space()
    
    # Model type
    model_name = (embedding_model if isinstance(embedding_model, str) else embedding_model._modules['0'].auto_model.config._name_or_path)
    
    # Write benchmark results to CSV
    write_results(redis_time, chroma_time, mongo_time, redis_query, chroma_query, mongo_query, redis_size, chroma_size, mongo_size, chunk_size, overlap, white_space, model_name)
    print("\n---Done processing PDFs and benchmarking---\n")


if __name__ == "__main__":
    main()