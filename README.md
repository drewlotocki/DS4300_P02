# Ollama RAG Ingest and Search

## Prerequisites

- Ollama app set up ([Ollama.com](Ollama.com))
- Pull Ollama Models llama3.2:latest, mistral:latest, dolphin-phi:latest 'ollama pull (model name)'
- Redis Stack running (Docker container is fine) on port 6379.  If that port is mapped to another port in 
Docker, change the port number in the creation of the Redis client in both python files in `src`.
- Chroma running (Docker container is fine) on port 8000. If that port is mapped to another port in 
Docker, change the port number in the creation of the Chroma client in both python files in `src`.
- MongoDB dwolaoad and configure the compass ([mongodb.com/try/download/compass]) for your operating system. Then run Mongodb Community Server (Docker container is fine) on port 27017. If that port is mapped to another port in 
Docker, change the port number in the creation of the Chroma client in both python files in `src`.

- ### Requiremnts
- 'requirements.txt' - contains all the items that one needs to import for the project. One needs to intall eah of them into their envioment using the command `pip install (requirment library name)`

## Source Code
- `src/ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Redis-stack, Chroma, and MongoDb
- `src/search.py` - simple question answering using 

## Configure runs



## Parameters

- 
