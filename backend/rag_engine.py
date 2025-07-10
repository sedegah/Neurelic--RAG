import os
import tempfile
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from .file_utils import parse_file
from .config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, GROQ_API_KEY
import requests

# Call Groq API for chat completion
async def chatgroq_qa(query: str, context: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    # You can adjust the model as needed (e.g., "llama3-70b-8192")
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer only from the provided context. If the answer is not in the context, say 'Not found in the provided documents.'"},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ],
        "max_tokens": 512,
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Load or create FAISS index
if os.path.exists(FAISS_INDEX_PATH):
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_INDEX_PATH + '.meta', 'r', encoding='utf-8') as f:
        metadata_store = eval(f.read())
else:
    faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    metadata_store = {}

# Memory buffer for chat
memory = ConversationBufferMemory(return_messages=True)

async def process_documents(files: List[Any]) -> List[str]:
    doc_ids = []
    for file in files:
        # Save to temp file
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        # Parse file
        text_chunks = parse_file(tmp_path)
        print(f"[DEBUG] Parsed {file.filename}: {len(text_chunks)} chunks")
        for i, chunk in enumerate(text_chunks):
            print(f"[DEBUG] Chunk {i}: {chunk['text'][:100]}...")
        # Embed and add to FAISS
        for chunk in text_chunks:
            emb = embedding_model.encode([chunk['text']])[0]
            faiss_index.add(np.array([emb], dtype=np.float32))
            idx = faiss_index.ntotal - 1
            metadata_store[idx] = {
                'source': file.filename,
                'chunk': chunk['text'],
                'meta': chunk.get('meta', {})
            }
        doc_ids.append(file.filename)
        os.remove(tmp_path)
    print(f"[DEBUG] Metadata store now has {len(metadata_store)} entries.")
    # Persist FAISS and metadata
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_INDEX_PATH + '.meta', 'w', encoding='utf-8') as f:
        f.write(str(metadata_store))
    return doc_ids

async def answer_query(query: str, chat_history: str = None) -> Tuple[str, List[Dict]]:
    # Embed query
    query_emb = embedding_model.encode([query])[0]
    # Search FAISS
    D, I = faiss_index.search(np.array([query_emb], dtype=np.float32), k=5)
    retrieved_chunks = []
    for idx in I[0]:
        if idx in metadata_store:
            retrieved_chunks.append(metadata_store[idx])
    # Compose context
    context = '\n'.join([c['chunk'] for c in retrieved_chunks])
    print(f"[DEBUG] Retrieved {len(retrieved_chunks)} chunks for query: '{query}'")
    for i, c in enumerate(retrieved_chunks):
        print(f"[DEBUG] Retrieved Chunk {i}: {c['chunk'][:100]}...")
    print(f"[DEBUG] Context sent to Groq: {context[:300]}...")
    # Use Groq Chat API to answer
    answer = await chatgroq_qa(query, context)
    print(f"[DEBUG] Groq Answer: {answer}")
    # Update memory
    memory.save_context({"input": query}, {"output": answer})
    return answer, retrieved_chunks
