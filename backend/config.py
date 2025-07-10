import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Embedding model name for sentence-transformers
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Path to store FAISS index and metadata
FAISS_INDEX_PATH = "faiss_index.index"

# Groq API key (loaded from .env)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
