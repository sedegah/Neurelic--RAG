from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from .rag_engine import process_documents, answer_query

app = FastAPI()

# Allow CORS for local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        doc_ids = await process_documents(files)
        return {"status": "success", "doc_ids": doc_ids}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/query/")
async def query(query: str = Form(...), chat_history: str = Form(None)):
    try:
        answer, sources = await answer_query(query, chat_history)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
