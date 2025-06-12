import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load models
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')

# Load and encode documents
def load_documents(path="documents.json"):
    with open(path, "r") as f:
        docs = json.load(f)
    return docs

def encode_docs(docs):
    return embedding_model.encode(docs)

def create_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def retrieve(query, docs, embeddings, index, k=2):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding), k)
    return [docs[i] for i in indices[0]]

def generate_answer(query, context_docs):
    context = "\n".join(context_docs)
    prompt = f"Given the following information:\n{context}\n\nAnswer the query: {query}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model.generate(inputs.input_ids, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
