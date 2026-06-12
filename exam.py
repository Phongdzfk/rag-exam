# ============================================================
#  exam.py - Gọi Teacher Server. Dùng:
#    python exam.py register            # đăng ký server của mình
#    python exam.py evaluate            # bắt đầu thi LẦN ĐẦU (document_received=False)
#    python exam.py evaluate --received # các lần SAU (đã có vector db, bỏ qua upload)
#    python exam.py result              # xem điểm + trạng thái hiện tại
#    python exam.py reset               # reset điểm để thi lại
#    python exam.py testllm             # test gọi proxy LLM xem sống chưa
#    python exam.py testlocal           # test /upload, /ask trên server CỦA MÌNH
# ============================================================
import sys
import json
import requests

import config

HEADERS = {"X-Student-ID": config.STUDENT_ID.upper(),
           "Content-Type": "application/json"}


def show(resp):
    print(f"HTTP {resp.status_code}")
    try:
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(resp.text[:1000])


def register():
    url = f"{config.TEACHER_BASE}/competition/register"
    print(f"POST {url}  server_url={config.MY_SERVER_URL}")
    show(requests.post(url, headers=HEADERS,
                       json={"server_url": config.MY_SERVER_URL}, timeout=15))


def evaluate(received: bool):
    url = f"{config.TEACHER_BASE}/competition/evaluate"
    print(f"POST {url}  document_received={received}")
    print("(Lần đầu nếu báo lỗi gửi tài liệu là BÌNH THƯỜNG - server mình đang embed,")
    print(" thầy để timeout upload 2 phút. Đợi server embed xong là chạy tiếp.)")
    # 100 câu có thể chạy 5-17 phút -> timeout phải đủ dài.
    # Mở terminal khác chạy `python exam.py result` để xem đang ở câu mấy.
    show(requests.post(url, headers=HEADERS,
                       json={"document_received": received}, timeout=7200))


def reset():
    url = f"{config.TEACHER_BASE}/competition/reset"
    print(f"POST {url}")
    show(requests.post(url, headers=HEADERS, timeout=15))


def result():
    url = f"{config.TEACHER_BASE}/competition/result"
    print(f"GET {url}")
    show(requests.get(url, headers=HEADERS, timeout=15))


def testllm():
    from llm import get_client
    print("Gọi thử proxy LLM ...")
    resp = get_client().chat.completions.create(
        model=config.LLM_MODEL, max_tokens=20,
        messages=[{"role": "user", "content": "Trả lời đúng 1 từ: 1+1=?"}])
    print("✓ LLM trả lời:", resp.choices[0].message.content)


def testlocal():
    base = f"http://127.0.0.1:{config.PORT}"
    print(f"Test {base}/health ...")
    show(requests.get(f"{base}/health", timeout=5))
    print(f"\nTest {base}/upload ...")
    doc = ("RAG (Retrieval-Augmented Generation) là kỹ thuật kết hợp truy xuất "
           "tài liệu với mô hình sinh. VectorDB lưu trữ embedding của các chunk. "
           "Chunking là quá trình cắt tài liệu thành các đoạn nhỏ. "
           "FastAPI là framework Python để xây dựng API nhanh.")
    show(requests.post(f"{base}/upload", json={"text": doc}, timeout=300))
    print(f"\nTest {base}/ask ...")
    show(requests.post(f"{base}/ask", json={
        "question": "RAG là gì? A. Một loại database B. Kỹ thuật kết hợp truy xuất với mô hình sinh C. Một framework web D. Một mô hình embedding"},
        timeout=120))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "register":
        register()
    elif cmd == "evaluate":
        evaluate("--received" in sys.argv)
    elif cmd == "reset":
        reset()
    elif cmd == "result":
        result()
    elif cmd == "testllm":
        testllm()
    elif cmd == "testlocal":
        testlocal()
    else:
        print(__doc__ or "Lệnh: register | evaluate [--received] | result | reset | testllm | testlocal")
