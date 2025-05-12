import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize the models
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Sentence transformer for embeddings
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Tokenizer for GPT-2
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')  # GPT-2 model for text generation

# Sample documents (replace with your own)
documents = [
    "Python is a high-level, interpreted programming language that is widely used for web development, data science, automation, and more.",
    "JavaScript is primarily used for building interactive web pages and is an essential technology of the World Wide Web.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve over time without being explicitly programmed."
]

# Function to encode documents using Sentence-Transformer
def encode_documents(documents):
    return embedding_model.encode(documents)

# Initialize FAISS index for fast retrieval
def initialize_faiss_index(doc_embeddings):
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # FAISS index with L2 distance
    index.add(np.array(doc_embeddings))  # Add document embeddings to the index
    return index

# Retrieve top-k most relevant documents based on query
def retrieve_documents(query, doc_embeddings, k=2):
    query_embedding = embedding_model.encode([query])  # Encode the query
    index = initialize_faiss_index(doc_embeddings)  # Create FAISS index for documents
    _, indices = index.search(np.array(query_embedding), k)  # Search for most similar documents
    return [documents[idx] for idx in indices[0]]

# Function to generate a response using GPT-2 model
def generate_response(query, retrieved_docs):
    prompt = f"Given the following information:\n\n{retrieved_docs[0]}\n\n{retrieved_docs[1]}\n\nAnswer the query: {query}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)  # Tokenize the prompt
    outputs = gpt2_model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)  # Generate response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Complete RAG Pipeline
def rag_system(query, documents):
    # Encode documents
    doc_embeddings = encode_documents(documents)
    
    # Retrieve the most relevant documents
    retrieved_docs = retrieve_documents(query, doc_embeddings)
    
    # Generate the response based on the query and retrieved documents
    response = generate_response(query, retrieved_docs)
    return response

# Test the system with a query
query = "What is machine learning?"
response = rag_system(query, documents)
print("Generated Response:", response)
