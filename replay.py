# ============================================================
#  replay.py - Bắn lại câu hỏi vào server CỦA MÌNH để test prompt
#  KHÔNG tốn lượt nộp, KHÔNG đụng gì đến Teacher Server.
#
#  Dùng:
#    python replay.py                  # bắn lại TẤT CẢ câu trong qa_log.txt
#    python replay.py 5                # chỉ bắn lại câu số 5 trong log
#    python replay.py "RAG là gì? A... B... C... D..."   # bắn 1 câu tự gõ
#
#  Quy trình tune giữa giờ thi:
#    1. Soi storage/qa_log.txt tìm câu nghi sai (lập luận vô lý / format hỏng)
#    2. Sửa prompt trong llm.py -> Ctrl+C restart server.py
#    3. python replay.py  -> so đáp án + lập luận mới (chi tiết trong qa_log)
#    4. Ổn rồi mới: python exam.py reset && python exam.py evaluate --received
#
#  Lưu ý: mỗi lần replay sẽ ghi THÊM vào qa_log.txt (entry mới nhất ở cuối).
# ============================================================
import sys
import time

import requests

import config

BASE = f"http://127.0.0.1:{config.PORT}"


def ask(q: str):
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/ask", json={"question": q}, timeout=120)
        ans = r.json().get("answer")
    except Exception as e:
        print(f"  ✗ LỖI: {e} (server.py có đang chạy không?)")
        return None
    print(f"  -> {ans}  ({time.time()-t0:.1f}s) | {q[:75]}{'...' if len(q) > 75 else ''}")
    return ans


def load_questions():
    try:
        lines = open(config.LOG_PATH, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        sys.exit(f"Chưa có {config.LOG_PATH} - chạy evaluate hoặc testlocal trước đã.")
    qs, seen = [], set()
    for ln in lines:
        if ln.startswith("Q: ") and ln[3:] not in seen:
            seen.add(ln[3:])
            qs.append(ln[3:])
    return qs


if __name__ == "__main__":
    arg = " ".join(sys.argv[1:]).strip()

    if arg and not arg.isdigit():
        # bắn 1 câu tự gõ
        ask(arg)
    else:
        qs = load_questions()
        if not qs:
            sys.exit("Không tìm thấy câu hỏi nào trong qa_log.txt")
        if arg.isdigit():
            i = int(arg)
            if not (1 <= i <= len(qs)):
                sys.exit(f"Chỉ có {len(qs)} câu (1-{len(qs)})")
            print(f"Bắn lại câu {i}/{len(qs)}:")
            ask(qs[i - 1])
        else:
            print(f"Bắn lại {len(qs)} câu từ {config.LOG_PATH}"
                  f" (lập luận + tokens chi tiết xem entry mới ở CUỐI qa_log):")
            for idx, q in enumerate(qs, 1):
                print(f"[{idx}/{len(qs)}]", end="")
                ask(q)