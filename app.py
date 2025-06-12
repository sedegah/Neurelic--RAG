import streamlit as st
import json
from ragify_engine import load_documents, encode_docs, create_index, retrieve, generate_answer

st.set_page_config(page_title="RAGify", layout="wide")

# Load documents
docs = load_documents()
doc_embeddings = encode_docs(docs)
index = create_index(doc_embeddings)

# UI
st.title("RAGify: Retrieval-Augmented Generation Demo")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Retrieving and generating answer..."):
        relevant_docs = retrieve(query, docs, doc_embeddings, index)
        answer = generate_answer(query, relevant_docs)

    st.subheader("🔍 Retrieved Documents")
    for i, doc in enumerate(relevant_docs):
        st.markdown(f"**Doc {i+1}:** {doc}")

    st.subheader(" Answer")
    st.markdown(answer)

# Add new document
st.sidebar.title("Add New Document")
new_doc = st.sidebar.text_area("Document Text")
if st.sidebar.button("Add to Knowledge Base") and new_doc:
    with open("documents.json", "r+") as f:
        data = json.load(f)
        data.append(new_doc)
        f.seek(0)
        json.dump(data, f, indent=2)
    st.sidebar.success("Document added! Please reload the page.")
