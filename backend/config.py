import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

FAISS_INDEX_PATH = "faiss_index.index"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
