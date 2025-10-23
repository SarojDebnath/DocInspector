import os

# Resolve OpenMP runtime conflicts on Windows (FAISS/NumPy/MKL)
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from src.document_chat.chat_service import ChatSessionService
from utils.document_ops import FastAPIFileAdapter,read_pdf_via_handler
from logger import GLOBAL_LOGGER as log
from chainlit.utils import mount_chainlit

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")  # <--- keep consistent with save_local()

app = FastAPI(title="Document Portal API", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/logo", StaticFiles(directory=str(BASE_DIR / "logo")), name="logo")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Chainlit app at /chainlit (minimal integration)
mount_chainlit(app=app, target="chainlit_docchat.py", path="/chatbot")

# Global chat session service for multi-doc chat with memory
_chat_service = ChatSessionService(
    faiss_base=FAISS_BASE,
    upload_base=UPLOAD_BASE,
    index_name=FAISS_INDEX_NAME,
)

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    log.info("Serving UI homepage.")
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
def health() -> Dict[str, str]:
    log.info("Health check passed.")
    return {"status": "ok", "service": "document-portal"}

# ---------- ANALYZE ----------
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        log.info(f"Received file for analysis: {file.filename}")
        dh = DocHandler()
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        text = read_pdf_via_handler(dh, saved_path)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        log.info("Document analysis complete.")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Error during document analysis")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# ---------- COMPARE ----------
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    try:
        log.info(f"Comparing files: {reference.filename} vs {actual.filename}")
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        _ = ref_path, act_path
        combined_text = dc.combine_documents()
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        log.info("Document comparison completed.")
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

# ---------- CHAT: INDEX ----------
@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        log.info(f"Indexing chat session. Session ID: {session_id}, Files: {[f.filename for f in files]}")
        wrapped = [FastAPIFileAdapter(f) for f in files]
        # this is my main class for storing a data into VDB
        # created a object of ChatIngestor
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        # NOTE: ensure your ChatIngestor saves with index_name="index" or FAISS_INDEX_NAME
        # e.g., if it calls FAISS.save_local(dir, index_name=FAISS_INDEX_NAME)
        ci.built_retriver(  # if your method name is actually build_retriever, fix it there as well
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        log.info(f"Index created successfully for session: {ci.session_id}")
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat index building failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

# ---------- CHAT: QUERY ----------
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    try:
        log.info(f"Received chat query: '{question}' | session: {session_id}")
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # build retriever + chain
        response = rag.invoke(question, chat_history=[])
        log.info("Chat query handled successfully.")

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

# ---------- CHAT SESSIONS (generalized multi-doc chat) ----------
@app.post("/chat/session/start")
async def chat_session_start(session_id: Optional[str] = Form(None)) -> Any:
    try:
        sid = _chat_service.get_or_create_session(session_id)
        return {"session_id": sid}
    except Exception as e:
        log.exception("Failed to start chat session")
        raise HTTPException(status_code=500, detail=f"Failed to start chat session: {e}")


@app.post("/chat/session/add_docs")
async def chat_session_add_docs(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        sid = _chat_service.get_or_create_session(session_id)
        wrapped = [FastAPIFileAdapter(f) for f in files]
        info = _chat_service.add_documents(
            session_id=sid,
            files=wrapped,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
        )
        return info
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to add docs to session")
        raise HTTPException(status_code=500, detail=f"Failed to add docs: {e}")


@app.post("/chat/session/message")
async def chat_session_message(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    k: int = Form(5),
) -> Any:
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        answer = _chat_service.send_message(session_id=session_id, content=message, k=k)
        return {"answer": answer, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to send chat message")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {e}")


@app.get("/chat/session/history")
async def chat_session_history(session_id: Optional[str] = None) -> Any:
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        history = _chat_service.get_history(session_id)
        return {"session_id": session_id, "history": history}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to fetch chat history")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {e}")


@app.post("/chat/session/reset")
async def chat_session_reset(session_id: Optional[str] = Form(None)) -> Any:
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        _chat_service.reset(session_id)
        return {"session_id": session_id, "status": "reset"}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to reset chat session")
        raise HTTPException(status_code=500, detail=f"Failed to reset chat session: {e}")

# ---------- CHAT: COMPARE AGAINST SESSION ----------
@app.post("/chat/compare_session")
async def chat_compare_session(
    session_id: str = Form(...),
    candidate: UploadFile = File(...),
) -> Any:
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        session_dir = os.path.join(UPLOAD_BASE, session_id)
        if not os.path.isdir(session_dir):
            raise HTTPException(status_code=404, detail=f"Session data not found at: {session_dir}")

        # Load reference docs (all previously uploaded for this session)
        ref_paths: List[Path] = []
        for ext in ("*.pdf", "*.docx", "*.txt"):
            ref_paths.extend(Path(session_dir).glob(ext))
        ref_docs = load_documents(ref_paths)
        if not ref_docs:
            raise HTTPException(status_code=400, detail="No reference documents in session to compare against")

        # Save candidate temporarily into the same session dir
        cand_path = Path(session_dir) / candidate.filename
        with open(cand_path, "wb") as f:
            f.write(await candidate.read())
        act_docs = load_documents([cand_path])

        combined_text = concat_for_comparison(ref_docs, act_docs)
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        return {"rows": df.to_dict(orient="records"), "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Session comparison failed")
        raise HTTPException(status_code=500, detail=f"Session comparison failed: {e}")

# ---------- CHAT: ANALYZE ALL DOCS IN SESSION ----------
@app.post("/chat/analyze_session")
async def analyze_session(
    session_id: str = Form(...),
) -> Any:
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        session_dir = os.path.join(UPLOAD_BASE, session_id)
        if not os.path.isdir(session_dir):
            raise HTTPException(status_code=404, detail=f"Session data not found at: {session_dir}")

        paths: List[Path] = []
        for ext in ("*.pdf", "*.docx", "*.txt"):
            paths.extend(Path(session_dir).glob(ext))
        docs = load_documents(paths)
        if not docs:
            raise HTTPException(status_code=400, detail="No documents found in session")

        text = concat_for_analysis(docs)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Session analysis failed")
        raise HTTPException(status_code=500, detail=f"Session analysis failed: {e}")

# command for executing the fast api
# uvicorn api.main:app --port 8080 --reload
#uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload