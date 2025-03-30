# Ollama RAG Ingest and Search
## Cody Ho and Andrew Lotocki

# Setup the Enivroment

## Prerequisites

- Ollama app set up ([Ollama.com](Ollama.com))
- Pull Ollama Models llama3.2:latest, mistral:latest, dolphin-phi:latest 'ollama pull (model name)'
- Redis Stack running (Docker container is fine) on port 6379.  If that port is mapped to another port in 
Docker, change the port number in the creation of the Redis client in both python files in `src`.
- Chroma running (Docker container is fine) on port 8000. If that port is mapped to another port in 
Docker, change the port number in the creation of the Chroma client in both python files in `src`.
- MongoDB dwolaoad and configure the compass ([mongodb.com/try/download/compass]) for your operating system. Then run Mongodb Community Server (Docker container is fine) on port 27017. If that port is mapped to another port in 
Docker, change the port number in the creation of the Chroma client in both python files in `src`.

### Requiremnts
- 'requirements.txt' - contains all the items that one needs to import for the project. One needs to intall eah of them into their envioment using the command `pip install (requirment library name)`
- Make sure your IDE is compatiable with Jupyter lab and make sure to install it if you wish to run the practical_2.ipynb notebook. 

## Source Code
- `src/ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Redis-stack, Chroma, and MongoDb
- `src/search.py` - simple question answering llm file 

# Configure Runs

## Parameters
- In line 273 of `src/ingest.py` one can adjust the ingestion parameters (chunk_size, overlap, white_space, embedding_model)
where the chunk size and overlap are integers, white space is boolean (Tells the script to remove white space and special charecters), the embedding model is a variable with the possible options detailed in lines 29-31. 

- In line 48 of `src/search.py` one can adjust the parameters embedding_model, starter_prompt, llm_model, database. Where embedding model options are detailed in lines 24-26, the starter prompt options are in lines 16-21, the llm model options are in lines 29-31, and the databse you wish to query from (they should all work the same after running ingest) are the choice of one of the strings ("redis", "chroma", "mongo")

- Note: After running ingest the embedding model in search must be the same one you chose for ingest.
- Note: You must run `src/ingest.py` before `src/search.py`


## Extras

- The Practical_2.ipynb file comtains all the results graphed in the presentation
- The results folder contains all of the information compiled when running the script
- The data folder contains all the documents one can add to the project (they must be in pdf form)
