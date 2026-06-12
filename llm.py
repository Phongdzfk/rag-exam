# ============================================================
#  llm.py - Gọi LLM qua Proxy của thầy + PROMPT (TUNE Ở ĐÂY)
# ============================================================
import re
import config
from openai import OpenAI

# ╔══════════════════════════════════════════════════════════╗
# ║   PROMPT - ĐÂY LÀ CHỖ SỬA TAY ĐỂ KÉO ĐIỂM LÚC THI        ║
# ║   Nguyên tắc: NGẮN GỌN (tiết kiệm token), RÕ RÀNG,       ║
# ║   ép trả lời đúng 1 ký tự.                                ║
# ╚══════════════════════════════════════════════════════════╝
# TẦNG 1 (CoT): cho LLM giải thích ngắn rồi mới chốt -> ăn câu phủ định/đếm/suy luận.
SYSTEM_PROMPT = (
    "Bạn là chuyên gia làm bài trắc nghiệm. Dựa CHỦ YẾU vào TÀI LIỆU. "
    "Giải thích ngắn gọn 1-2 câu, rồi BẮT BUỘC kết thúc bằng dòng đúng định dạng:\n"
    "Đáp án: X\n(X là một trong A, B, C, D)"
)

USER_TEMPLATE = """### TÀI LIỆU:
{context}

### CÂU HỎI:
{question}

Giải thích ngắn rồi dòng cuối ghi: Đáp án: X"""

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.LLM_BASE_URL or f"{config.TEACHER_BASE}/proxy",
            api_key=config.LLM_API_KEY or config.STUDENT_ID,  # khi thi: MSSV làm API key
            timeout=config.LLM_TIMEOUT,
        )
    return _client


def parse_answer(raw: str) -> str | None:
    """CoT-aware: ưu tiên dòng 'Đáp án: X' CUỐI CÙNG; không có thì lấy chữ cái CUỐI
    (phần giải thích nhắc đến nhiều option nên lấy chữ đầu tiên là SAI)."""
    if not raw:
        return None
    up = raw.upper()
    m = re.findall(r"ĐÁP ÁN\s*[:\-]?\s*\(?([ABCD])\)?", up)
    if m:
        return m[-1]
    m = re.findall(r"\b([ABCD])\b", up)
    if m:
        return m[-1]
    m = re.findall(r"([ABCD])", up)
    return m[-1] if m else None


def lexical_fallback(question: str, contexts: list[str]) -> str:
    """Phương án cứu cánh khi LLM chết: chọn option trùng từ với context nhiều nhất."""
    try:
        ctx_tokens = set(re.findall(r"\w+", " ".join(contexts).lower()))
        # tách các option dạng "A. ..." / "B) ..." / "C: ..."
        opts = re.findall(r"([A-D])[\.\):\-]\s*(.+?)(?=\s+[A-D][\.\):\-]\s|$)", question, re.S)
        if not opts:
            return "A"
        best, best_score = "A", -1.0
        for letter, text in opts:
            toks = re.findall(r"\w+", text.lower())
            if not toks:
                continue
            score = sum(1 for t in toks if t in ctx_tokens) / len(toks)
            if score > best_score:
                best, best_score = letter, score
        return best
    except Exception:
        return "A"


def answer_question(question: str, contexts: list[str]) -> tuple[str, str]:
    """
    Trả về (answer_letter, raw_llm_response).
    Có retry: mỗi lần retry cắt bớt 1 nửa context (phòng tràn token / lỗi mạng).
    KHÔNG BAO GIỜ trả về rỗng - tệ nhất cũng đoán bằng lexical_fallback.
    """
    ctxs = list(contexts)
    raw = ""
    for attempt in range(config.LLM_RETRY + 1):
        try:
            context_str = "\n---\n".join(ctxs) if ctxs else "(không có tài liệu)"
            resp = get_client().chat.completions.create(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        context=context_str, question=question)},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            ans = parse_answer(raw)
            if ans:
                return ans, raw
        except Exception as e:
            raw = f"[LLM ERROR lần {attempt}] {e}"
            print(raw)
        # retry với ít context hơn (phòng tràn sequence length)
        ctxs = ctxs[: max(1, len(ctxs) // 2)]

    return lexical_fallback(question, contexts), raw
