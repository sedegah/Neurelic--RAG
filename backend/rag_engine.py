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

async def chatgroq_qa(query: str, context: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer only from the provided context. "
                    "If the answer is not in the context, say 'Not found in the provided documents.'"
                )
            },
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ],
        "max_tokens": 512,
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

if os.path.exists(FAISS_INDEX_PATH):
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_INDEX_PATH + '.meta', 'r', encoding='utf-8') as f:
        metadata_store = eval(f.read())
else:
    faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    metadata_store = {}

memory = ConversationBufferMemory(return_messages=True)

async def process_documents(files: List[Any]) -> List[str]:
    doc_ids = []
    for file in files:
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        text_chunks = parse_file(tmp_path)

        for i, chunk in enumerate(text_chunks):
            print(f"[DEBUG] Chunk {i}: {chunk['text'][:100]}...")

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

    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_INDEX_PATH + '.meta', 'w', encoding='utf-8') as f:
        f.write(str(metadata_store))

    return doc_ids

async def answer_query(query: str, chat_history: str = None) -> Tuple[str, List[Dict]]:
    query_emb = embedding_model.encode([query])[0]
    D, I = faiss_index.search(np.array([query_emb], dtype=np.float32), k=5)

    retrieved_chunks = []
    for idx in I[0]:
        if idx in metadata_store:
            retrieved_chunks.append(metadata_store[idx])

    context = '\n'.join([c['chunk'] for c in retrieved_chunks])

    answer = await chatgroq_qa(query, context)
    memory.save_context({"input": query}, {"output": answer})

    return answer, retrieved_chunks
