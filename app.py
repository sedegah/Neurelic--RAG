# app.py
import json
import streamlit as st
from neurelic import RAGSystem

st.set_page_config(page_title="Neurelic Demo", layout="wide")

# --- Initialise system & index ---
rag  = RAGSystem()                      # default config
docs = json.load(open("documents.json"))
rag.index_documents(docs)

# --- UI ---
st.title("🔍 Neurelic – Ask your knowledge base")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking…"):
        answer = rag.query(query)
    st.subheader("Answer")
    st.markdown(answer)

    st.subheader("Debug / Stats")
    st.json(rag.get_stats())
