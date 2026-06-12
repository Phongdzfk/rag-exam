# ============================================================
#  server.py - Student Server (FastAPI)
#  Chạy: python server.py
#  2 endpoint bắt buộc: POST /upload, POST /ask
#  Parse request LINH HOẠT (thầy có đổi tên biến vẫn chạy).
# ============================================================
import os
import time
import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import config
import rag
from rag import STORE
from llm import answer_question

app = FastAPI(title=f"Student Server - {config.STUDENT_ID}")

# Các tên biến có thể gặp (thầy báo có đổi tên biến trong request /upload)
TEXT_KEYS = ["text", "content", "document", "doc", "data", "body",
             "raw_text", "document_text", "doc_text", "payload"]
DOCID_KEYS = ["doc_id", "id", "docId", "document_id", "docid"]
QUESTION_KEYS = ["question", "query", "q", "prompt", "text", "content"]


def pick(data: dict, keys: list[str]):
    """Lấy giá trị theo danh sách tên biến khả dĩ."""
    for k in keys:
        if k in data and data[k] not in (None, ""):
            return data[k]
    return None


def log_qa(lines: list[str]):
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
    with open(config.LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" + "=" * 60 + "\n")


@app.on_event("startup")
def startup():
    print(f"[SERVER] Student: {config.STUDENT_ID} | {config.MY_SERVER_URL}")
    # warm model ngay lúc bật server (tải về nếu chưa có - làm khi CÒN MẠNG)
    try:
        rag.get_model()
    except Exception as e:
        print(f"[SERVER] ⚠ Không load được embedding model: {e}")
        print("[SERVER] -> Sẽ chạy chế độ BM25-only (set USE_EMBEDDINGS=False trong config.py)")
    # warm reranker luôn lúc khởi động (load lười thì câu 1 phải gánh 5-15s)
    rag.get_reranker()
    # load lại vector db cũ nếu có (sau khi restart không cần embed lại)
    if STORE.load():
        print("[SERVER] ✓ VectorDB cũ đã sẵn sàng (evaluate với document_received=True được).")
    else:
        print("[SERVER] Chưa có VectorDB - lần đầu evaluate dùng document_received=False.")


@app.get("/health")
def health():
    return {"status": "ok", "student": config.STUDENT_ID,
            "vectordb_ready": STORE.ready, "chunks": len(STORE.chunks)}


# ----------------------- POST /upload -----------------------
@app.post("/upload")
async def upload(request: Request):
    t0 = time.time()
    try:
        data = await request.json()
    except Exception:
        body = (await request.body()).decode("utf-8", errors="ignore")
        data = {"text": body}

    if isinstance(data, str):
        data = {"text": data}

    text = pick(data, TEXT_KEYS)
    if text is None:
        # bí quá: lấy giá trị string dài nhất trong request
        strings = [v for v in data.values() if isinstance(v, str)]
        text = max(strings, key=len) if strings else None

    if not text:
        print(f"[UPLOAD] ✗ Không tìm thấy field chứa tài liệu. Keys nhận được: {list(data.keys())}")
        return JSONResponse(status_code=422, content={
            "status": "error", "doc_id": None, "chunks": 0,
            "message": f"Không tìm thấy text. Keys: {list(data.keys())}"})

    doc_id = pick(data, DOCID_KEYS) or "doc_1"
    print(f"[UPLOAD] Nhận tài liệu {doc_id}, {len(text)} ký tự. Bắt đầu chunk + embed ...")

    STORE.build(text)
    STORE.save()

    elapsed = time.time() - t0
    print(f"[UPLOAD] ✓ Xong trong {elapsed:.1f}s, {len(STORE.chunks)} chunks.")
    return {"status": "success", "doc_id": doc_id, "chunks": len(STORE.chunks)}


# ------------------------ POST /ask ------------------------
@app.post("/ask")
async def ask(request: Request):
    t0 = time.time()
    try:
        data = await request.json()
    except Exception:
        data = {"question": (await request.body()).decode("utf-8", errors="ignore")}
    if isinstance(data, str):
        data = {"question": data}

    question = pick(data, QUESTION_KEYS)
    if not question:
        strings = [v for v in data.values() if isinstance(v, str)]
        question = max(strings, key=len) if strings else ""

    # retrieve: Tầng 3 (per-option) + Tầng 4 (rerank) + Tầng 2 (neighbors)
    contexts, budget = [], config.CONTEXT_CHAR_BUDGET
    for chunk in rag.retrieve(question):
        if budget - len(chunk) < 0:
            break
        contexts.append(chunk)
        budget -= len(chunk)

    # gọi LLM (có retry + fallback, không bao giờ trả rỗng)
    answer, raw = answer_question(question, contexts)
    elapsed = time.time() - t0

    print(f"[ASK] ({elapsed:.1f}s) -> {answer} | Q: {question[:80]}...")
    log_qa([
        f"[{datetime.datetime.now():%H:%M:%S}] elapsed={elapsed:.1f}s answer={answer}",
        f"Q: {question}",
        f"CONTEXTS ({len(contexts)}):",
        *[f"  - {c[:150]}..." for c in contexts],
        f"LLM RAW: {raw}",
    ])

    return {"answer": answer, "sources": contexts}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=config.HOST, port=config.PORT, reload=False)
