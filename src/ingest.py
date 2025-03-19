import ollama
import redis
import chromadb
import numpy as np
from redis.commands.search.query import Query
import os
import fitz

# Initialize Redis and ChromaDB connections
redis_client = redis.Redis(host="localhost", port=6379, db=0)
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# Clear both Redis and ChromaDB stores
def clear_stores():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")
    
    try:
        print("Clearing ChromaDB collection...")
        chroma_client.delete_collection(name="pdf_embeddings")
        print("ChromaDB store cleared.")
    except:
        print("Clearing ChromaDB collection...")
        print("ChromaDB store cleared.")

    global chroma_collection
    chroma_collection = chroma_client.get_or_create_collection(name="pdf_embeddings")


def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def store_embedding(file: str, page: str, chunk: str, embedding: list):
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
    
    chroma_collection.add(ids=[key], embeddings=[embedding], metadatas=[{"file": file, 
                                                                         "page": page, 
                                                                         "chunk": chunk, 
                                                                         }])
    print(f"Stored embedding for: {chunk}")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


def query_stores(query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    embedding = get_embedding(query_text)
    
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    print("Redis Search Results:")
    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")
    
    chroma_results = chroma_collection.query(query_embeddings=[embedding])
    print("ChromaDB Search Results:")
    for i, (doc_id, score) in enumerate(zip(chroma_results["ids"][0], chroma_results["distances"][0])):
        print(f"{doc_id} \n ----> {score}\n")
     


def main():
    clear_stores()
    create_hnsw_index()
    process_pdfs("../data/")
    print("\n---Done processing PDFs---\n")
    query_stores("What is the capital of France?")


if __name__ == "__main__":
    main()
