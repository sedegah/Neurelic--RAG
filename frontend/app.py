import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG System", layout="wide")
st.title("Neurelic – RAG-Powered Knowledge Assistant ")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose files (PDF, DOCX, TXT, CSV, XLSX)",
    type=["pdf", "docx", "txt", "csv", "xlsx"],
    accept_multiple_files=True
)

if st.sidebar.button("Upload") and uploaded_files:
    files = [("files", (f.name, f, f.type)) for f in uploaded_files]
    with st.spinner("Uploading and processing documents..."):
        res = requests.post(f"{BACKEND_URL}/upload/", files=files)
    if res.status_code == 200:
        st.sidebar.success("Documents uploaded and indexed!")
    else:
        st.sidebar.error(f"Upload failed: {res.json().get('error')}")

st.markdown("---")
st.header("Chat with your documents")

user_query = st.text_input("Ask a question:")

if st.button("Send") and user_query:
    with st.spinner("Retrieving answer..."):
        chat_history = '\n'.join([f"User: {q}\nAI: {a}" for q, a in st.session_state['chat_history']])
        res = requests.post(
            f"{BACKEND_URL}/query/",
            data={"query": user_query, "chat_history": chat_history}
        )
    if res.status_code == 200:
        answer = res.json()["answer"]
        sources = res.json()["sources"]
        st.session_state['chat_history'].append((user_query, answer))
        st.markdown(f"**You:** {user_query}")
        st.markdown(f"**RAG:** {answer}")
        with st.expander("Show sources"):
            for src in sources:
                st.write(f"**Source:** {src['source']}")
                st.write(src['chunk'])
    else:
        st.error(f"Error: {res.json().get('error')}")

if st.session_state['chat_history']:
    st.markdown("---")
    st.subheader("Chat History")
    for q, a in st.session_state['chat_history']:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**RAG:** {a}")
