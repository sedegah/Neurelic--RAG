
# Neurelic – RAG-Powered Knowledge Assistant

Neurelic is a **Retrieval-Augmented Generation (RAG)** system designed to provide smart, context-aware responses by combining document retrieval with advanced language generation. Built using a modular architecture, Neurelic enables querying large document sets using semantic search and delivers AI-generated answers powered by transformer models.

>  Ask questions.  Search documents.  Get intelligent answers.

---

## Features

-  **Semantic Document Search** – Uses embeddings for accurate information retrieval.
-  **Language Generation** – Integrates large language models for natural language responses.
-  **Multi-format Document Support** – Easily index and query JSON or raw text docs.
-  **Modular Architecture** – Pluggable components for embeddings, retrieval, and generation.
-  **Streamlit UI** – Simple, responsive interface for user-friendly querying.

---

## 🛠 Installation

1. **Clone the repository**
```bash
git clone https://github.com/sedegah/Neurelic--RAG.git
cd Neurelic--RAG
````

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

##  Run the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

##  How It Works

1. **Indexing:** `RAGSystem` loads and indexes documents using sentence embeddings (via `SentenceTransformers`).
2. **Querying:** On user input, the system retrieves the top relevant passages using FAISS similarity.
3. **Generation:** The selected context is passed to a language model to generate a final answer.

---

##  Example Use Cases

* Internal knowledge base assistant
* Research paper Q\&A system
* Educational tools
* Legal and compliance document querying

---

##  License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more info.

---

##  Author

**Kimathi Elikplim Sedegah**
[Portfolio](https://kimathisedegah.vercel.app) · [GitHub](https://github.com/sedegah)

---

##  Contributions

Contributions, issues, and feature requests are welcome!
Feel free to [open an issue](https://github.com/sedegah/Neurelic--RAG/issues) or submit a pull request.

```

---

Let me know if you’d like a version with setup screenshots, hosted demo link, or deployment guide (e.g., on Vercel or Hugging Face Spaces).
```
