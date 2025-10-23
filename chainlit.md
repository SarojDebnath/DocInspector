
![KAIDO](/logo/kaido.png)

## Document Portal Chatbot

A minimal, production-friendly chat UI for interacting with your documents. Upload multiple PDFs/DOCX/TXT, build a FAISS index per session, and ask questions with chat memory. The chatbot uses Azure OpenAI for both embeddings and generation and reuses the same pipeline as the main API.

#### Use `/upload` for uploading new files

### What you can do
- Upload multiple documents and chat across them
- Add more files at any time using the chat
- Ask natural questions like “what’s new” or “summarize X”

### Tech Stack
- LangChain core for chains and prompts
- Azure OpenAI (AzureChatOpenAI + AzureOpenAIEmbeddings)
- FAISS vector store (on-disk per session)
- FastAPI backend (existing), Chainlit UI (mounted at /chatbot)
- PyMuPDF/Docx2txt/Text loaders for ingestion
- Structured logging with `logger/custom_logger.py`

### How it works (high-level)
1. Uploaded files are saved under `data/<session_id>`.
2. Text is chunked and embedded with Azure embeddings.
3. Chunks and metadata are stored in a FAISS index at `faiss_index/<session_id>`.
4. The chat uses a retrieval-augmented generation flow and keeps conversational memory in session.

Open the chatbot at `/chatbot` once the API is running.
