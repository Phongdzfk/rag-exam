# ============================================================
#  mock_teacher.py - GIẢ LẬP TEACHER SERVER ĐỂ THI THỬ Ở NHÀ
#  (Hôm thi thật KHÔNG dùng file này)
#
#  Cách dùng (3 terminal):
#    T1: python -m uvicorn mock_teacher:app --port 8000
#    T2: python server.py
#    T3: sửa config.py: TEACHER_BASE = "http://127.0.0.1:8000/api/v1"
#        python exam.py register
#        python exam.py evaluate          -> chấm 10 câu, in điểm
#        python exam.py result / reset
#
#  Proxy LLM giả lập: nếu có Ollama đang chạy (localhost:11434) thì
#  forward sang Ollama (giống proxy thật). Không có thì dùng heuristic
#  (chọn đáp án trùng từ với tài liệu nhất) - đủ để test luồng + đo thời gian.
# ============================================================
import re
import time
import requests as rq
from fastapi import FastAPI, Body, Header
from fastapi.responses import JSONResponse

OLLAMA = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"   # đổi theo model bạn đã `ollama pull`
QUESTION_TIMEOUT = 60
UPLOAD_TIMEOUT = 120

# ----------------- ĐỀ THI MẪU (tự thay tài liệu + câu hỏi của bạn) -----------------
DOCUMENT = """
RAG (Retrieval-Augmented Generation) là kỹ thuật kết hợp truy xuất thông tin với mô hình sinh văn bản,
giúp mô hình trả lời dựa trên tài liệu bên ngoài thay vì chỉ dựa vào kiến thức đã huấn luyện.
Quy trình RAG gồm ba bước chính: lập chỉ mục (indexing), truy xuất (retrieval) và sinh câu trả lời (generation).
Chunking là quá trình cắt tài liệu lớn thành các đoạn nhỏ gọi là chunk, giúp truy xuất chính xác hơn.
Kích thước chunk quá lớn làm giảm độ chính xác truy xuất, quá nhỏ làm mất ngữ cảnh.
Embedding là vector số biểu diễn ngữ nghĩa của văn bản trong không gian nhiều chiều.
VectorDB là cơ sở dữ liệu chuyên lưu trữ và tìm kiếm embedding theo độ tương đồng, ví dụ FAISS, ChromaDB, Milvus.
Cosine similarity là độ đo phổ biến nhất để so sánh hai vector embedding.
BM25 là thuật toán xếp hạng dựa trên tần suất từ khóa, thuộc nhóm tìm kiếm thưa (sparse retrieval).
Hybrid search kết hợp tìm kiếm dày đặc (dense, embedding) và thưa (sparse, BM25) để tăng độ chính xác.
FastAPI là framework Python hiệu năng cao để xây dựng API, hỗ trợ async và tự sinh tài liệu OpenAPI.
Pydantic được FastAPI dùng để khai báo và kiểm tra schema dữ liệu.
Top-k retrieval nghĩa là lấy ra k đoạn văn bản liên quan nhất với câu truy vấn.
Prompt engineering là kỹ thuật thiết kế câu lệnh đầu vào để mô hình ngôn ngữ trả lời chính xác hơn.
Temperature bằng 0 khiến mô hình trả lời ổn định và ít ngẫu nhiên nhất.
""".strip()

# Có test_document.txt (tạo bằng make_testdoc.py hoặc tự kiếm) -> dùng làm tài liệu thi
import os as _os
if _os.path.exists("test_document.txt"):
    DOCUMENT = open("test_document.txt", encoding="utf-8").read()
    print(f"[MOCK] Dùng test_document.txt ({len(DOCUMENT)/1e6:.2f}MB) làm tài liệu thi.")

QUESTIONS = [
    ("RAG là viết tắt của gì? A. Random Access Generation B. Retrieval-Augmented Generation C. Rapid AI Generation D. Recursive Answer Generation", "B"),
    ("Quy trình RAG gồm mấy bước chính? A. 2 B. 3 C. 4 D. 5", "B"),
    ("Chunking là gì? A. Cắt tài liệu lớn thành các đoạn nhỏ B. Nén tài liệu C. Mã hóa tài liệu D. Dịch tài liệu", "A"),
    ("Kích thước chunk quá lớn gây ra điều gì? A. Mất ngữ cảnh B. Giảm độ chính xác truy xuất C. Tăng tốc độ D. Không ảnh hưởng", "B"),
    ("VectorDB nào KHÔNG được nhắc đến trong tài liệu? A. FAISS B. ChromaDB C. Milvus D. MongoDB", "D"),
    ("Độ đo phổ biến nhất để so sánh hai embedding là gì? A. Euclidean B. Manhattan C. Cosine similarity D. Hamming", "C"),
    ("BM25 thuộc nhóm tìm kiếm nào? A. Dense B. Sparse C. Hybrid D. Neural", "B"),
    ("Hybrid search kết hợp những gì? A. Hai LLM B. Dense và sparse retrieval C. Hai VectorDB D. CPU và GPU", "B"),
    ("FastAPI dùng thư viện nào để kiểm tra schema? A. Marshmallow B. Cerberus C. Pydantic D. Voluptuous", "C"),
    ("Temperature bằng 0 khiến mô hình thế nào? A. Trả lời ổn định nhất B. Sáng tạo nhất C. Nhanh nhất D. Chậm nhất", "A"),
]

app = FastAPI(title="MOCK Teacher Server")
STATE = {}  # student_id -> {server_url, score, status, current_question}


def st(sid):
    return STATE.setdefault(sid, {"server_url": None, "score": 0.0,
                                  "status": "registered", "current_question": 0})


# --------------- proxy LLM ---------------
def heuristic_answer(user_msg: str) -> str:
    """Chọn option trùng từ với phần tài liệu trong prompt nhiều nhất."""
    doc_part = user_msg.split("CÂU HỎI")[0]
    ctx = set(re.findall(r"\w+", doc_part.lower()))
    opts = re.findall(r"([A-D])[\.\):\-]\s*(.+?)(?=\s+[A-D][\.\):\-]\s|$)", user_msg, re.S)
    best, score_b = "A", -1
    for letter, text in opts:
        toks = re.findall(r"\w+", text.lower())
        s = sum(t in ctx for t in toks) / max(len(toks), 1)
        if s > score_b:
            best, score_b = letter, s
    return best


@app.post("/api/v1/proxy/chat/completions")
def proxy(data: dict = Body(...)):
    # sync def -> FastAPI chạy trong threadpool, không khóa event loop
    # thử forward sang Ollama cho giống thật
    try:
        data["model"] = OLLAMA_MODEL
        r = rq.post(f"{OLLAMA}/chat/completions", json=data, timeout=50)
        if r.status_code == 200:
            return JSONResponse(r.json())
    except Exception:
        pass
    # không có Ollama -> heuristic
    ans = heuristic_answer(data["messages"][-1]["content"])
    return {"id": "mock", "object": "chat.completion", "created": 0, "model": "mock",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ans},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1}}


# --------------- competition ---------------
@app.post("/api/v1/competition/register")
def register(data: dict = Body(...), x_student_id: str = Header(default="UNKNOWN")):
    s = st(x_student_id)
    s.update(server_url=data.get("server_url"), score=0.0,
             status="registered", current_question=0)
    print(f"[MOCK] {x_student_id} đăng ký {s['server_url']}")
    return {"message": "Đăng ký thành công!", "student_id": x_student_id,
            "server_url": s["server_url"]}


@app.post("/api/v1/competition/evaluate")
def evaluate(data: dict = Body(default={}), x_student_id: str = Header(default="UNKNOWN")):
    # sync def QUAN TRỌNG: evaluate gọi ngược /ask của sinh viên, /ask lại gọi /proxy
    # của server này -> nếu async + blocking requests sẽ deadlock.
    received = bool(data.get("document_received", False))
    s = st(x_student_id)
    if not s["server_url"]:
        return JSONResponse(status_code=400, content={"message": "Chưa register!"})

    base = s["server_url"].rstrip("/")
    s["status"] = "evaluating"
    s["score"] = 0.0

    # 1) upload tài liệu (nếu sinh viên chưa có)
    if not received:
        print(f"[MOCK] Gửi tài liệu -> {base}/upload (timeout {UPLOAD_TIMEOUT}s)")
        try:
            r = rq.post(f"{base}/upload", json={"text": DOCUMENT, "doc_id": "mock_doc"},
                        timeout=UPLOAD_TIMEOUT)
            print(f"[MOCK] /upload -> {r.status_code} {r.text[:120]}")
        except Exception as e:
            s["status"] = "upload_failed"
            print(f"[MOCK] /upload LỖI: {e} (giống thật: lần đầu timeout là bình thường)")
            return JSONResponse(status_code=504, content={
                "message": f"Upload thất bại/timeout: {e}. Đợi server embed xong rồi evaluate lại với document_received=true."})

    # 2) bơm câu hỏi
    correct = 0
    for i, (q, key) in enumerate(QUESTIONS, 1):
        s["current_question"] = i
        t0 = time.time()
        try:
            r = rq.post(f"{base}/ask", json={"question": q}, timeout=QUESTION_TIMEOUT)
            elapsed = time.time() - t0
            ans = str(r.json().get("answer", "")).strip().upper()[:1]
            ok = (ans == key) and elapsed <= QUESTION_TIMEOUT
            correct += ok
            print(f"[MOCK] Câu {i}: đáp={ans} đúng={key} {'✓' if ok else '✗'} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"[MOCK] Câu {i}: LỖI {e} -> 0 điểm")
        s["score"] = correct * 10.0 / len(QUESTIONS)

    s["status"] = "finished"
    print(f"[MOCK] {x_student_id} XONG: {s['score']}/10")
    return {"message": x_student_id, "final_score": s["score"]}


@app.post("/api/v1/competition/reset")
def reset(x_student_id: str = Header(default="UNKNOWN")):
    s = st(x_student_id)
    s.update(score=0.0, status="reset", current_question=0)
    return {"status": "success",
            "message": f"Đã reset trạng thái thi của sinh viên {x_student_id}",
            "score": s["score"]}


@app.get("/api/v1/competition/result")
def result(x_student_id: str = Header(default="UNKNOWN")):
    s = st(x_student_id)
    return {"student_id": x_student_id, "score": s["score"],
            "status": s["status"], "current_question": s["current_question"]}
