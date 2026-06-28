from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Documents Folder
DOCUMENT_FOLDER = BASE_DIR / "documents"

# Vector Database
VECTOR_DB = BASE_DIR / "vector_db"

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama Model
OLLAMA_MODEL = "phi3:mini"

# Chunk Settings
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50