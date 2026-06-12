# 🚀 IMPROVE.md — LỘ TRÌNH NÂNG CẤP RAG SO VỚI BẢN GỐC

> ✅ **TẦNG 1–4 ĐÃ ĐƯỢC APPLY SẴN VÀO CODE** (bản hiện tại). File này giữ làm tài liệu để hiểu từng tầng, chỉnh knob, và biết cách ROLLBACK từng tầng nếu cần.

> Triết lý: **chạy baseline lấy điểm sàn trước → nâng từng tầng → đo bằng mock sau MỖI tầng → tệ hơn thì revert ngay.**
> Mỗi tầng dưới đây độc lập, có code paste sẵn, có cách đo, có cách rollback.
> Token budget tính theo **4096 tokens** (đã xác nhận).

---

## QUY TRÌNH ĐO (làm sau mỗi tầng — bắt buộc)

```bash
# 1. Bật mock + server, chạy bộ câu hỏi cố định trong mock_teacher.py
python exam.py register && python exam.py evaluate
# 2. Chạy 2 LẦN (LLM có nhiễu), ghi cả 2 điểm
# 3. Soi storage/qa_log.txt các câu sai: lỗi do retrieval (context lạc đề)
#    hay do LLM (context đúng mà chọn sai)?
```

Bảng theo dõi (tự điền):

| Tầng | Điểm lần 1 | Điểm lần 2 | Thời gian/câu | Giữ? |
|---|---|---|---|---|
| 0. Baseline | | | | — |
| 1. CoT | | | | |
| 2. Neighbor | | | | |
| 3. Per-option | | | | |
| 4. Reranker | | | | |

> Mock mặc định chỉ có 10 câu dễ — nên **thay DOCUMENT + QUESTIONS trong mock_teacher.py bằng tài liệu môn học + 15-20 câu tự chế có cả câu phủ định, câu đếm số, câu suy luận**. Bộ test khó mới phân biệt được tầng nào ăn tiền.

---

## NGÂN SÁCH TOKEN VỚI 4096 (đọc trước khi chỉnh budget)

Tiếng Việt với tokenizer GPT ăn ~2–2.5 ký tự/token. Phân bổ an toàn:

| Khoản | Tokens |
|---|---|
| System prompt + template | ~120 |
| Câu hỏi (4 đáp án) | ~150–250 |
| Output CoT (tầng 1) | ~160 |
| **Còn cho context** | **~3500 tokens ≈ 5500–6500 ký tự** |

→ Sau khi lên tầng 1, set trong `config.py`:
```python
CONTEXT_CHAR_BUDGET = 5500
TOP_K = 5   # budget rộng hơn nên lấy thêm chunk được
```
Nếu gặp lỗi `context length exceeded` → hạ về 4500.

---

## TẦNG 1 — CoT: CHO LLM SUY LUẬN NGẮN (dễ nhất, lãi nhất)

**Tại sao:** baseline ép trả lời ngay trong 8 token → chết các câu phủ định ("cái nào KHÔNG..."), câu đếm, câu suy luận. Cho nghĩ 1–2 câu trước khi chốt tăng độ chính xác rõ rệt. Chi phí: +2–5s/câu (vẫn lọt 60s).

**Sửa `config.py`:**
```python
LLM_MAX_TOKENS = 160
CONTEXT_CHAR_BUDGET = 5500
TOP_K = 5
```

**Sửa `llm.py` — thay SYSTEM_PROMPT và USER_TEMPLATE:**
```python
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
```

**Sửa `llm.py` — THAY TOÀN BỘ hàm `parse_answer`** (quan trọng! bản cũ lấy chữ cái ĐẦU TIÊN — với CoT phần giải thích sẽ nhắc nhiều option, phải lấy theo pattern hoặc chữ CUỐI):
```python
def parse_answer(raw: str) -> str | None:
    if not raw:
        return None
    up = raw.upper()
    m = re.findall(r"ĐÁP ÁN\s*[:\-]?\s*\(?([ABCD])\)?", up)
    if m:
        return m[-1]                       # ưu tiên dòng "Đáp án: X" cuối cùng
    m = re.findall(r"\b([ABCD])\b", up)
    if m:
        return m[-1]                       # CoT -> lấy chữ CUỐI, không phải đầu
    m = re.findall(r"([ABCD])", up)
    return m[-1] if m else None
```

**Rollback:** trả lại prompt cũ + `LLM_MAX_TOKENS=8` (hàm parse mới vẫn tương thích đáp án 1 ký tự, không cần trả lại).

---

## TẦNG 2 — NEIGHBOR EXPANSION (small-to-big, ~10 phút code)


**Rollback:** `neighbors=0` ngay trong lời gọi, hoặc trả lại hàm search cũ.


## TẦNG 4 — RERANKER CROSS-ENCODER (mạnh nhất về retrieval, nặng nhất)

**Rollback:** `USE_RERANK = False`.

---

## TẦNG 5 — SELF-CONSISTENCY (vote đa số) — chỉ khi proxy nhanh

Gọi LLM 3 lần (temperature 0.3), lấy đáp án xuất hiện nhiều nhất. Tăng ổn định câu khó nhưng **x3 tải lên proxy** — hôm thi cả lớp dồn vào 1 proxy thì rủi ro nghẽn/timeout. Mặc định ĐỪNG bật; chỉ bật nếu đo thấy mỗi call < 5s.

Sửa `answer_question` trong `llm.py`: bọc lời gọi LLM trong vòng `for _ in range(3)`, gom các đáp án vào list, trả `Counter(answers).most_common(1)[0][0]`. Nhớ `temperature=0.3` và canh tổng thời gian < 45s.

---

## ĐỔI MODEL EMBEDDING — SO SÁNH 3 ỨNG VIÊN

| Model | Kích thước | Tiếng Việt | Lưu ý |
|---|---|---|---|
| `intfloat/multilingual-e5-small` (mặc định) | ~470MB | Tốt | Cần prefix query:/passage: (code tự xử lý) |
| `bkai-foundation-models/vietnamese-bi-encoder` | ~540MB | Rất tốt | Nền PhoBERT — NÊN tách từ bằng pyvi |
| `keepitreal/vietnamese-sbert` | ~540MB | Tốt (data NLI/STS) | Nền PhoBERT — NÊN tách từ bằng pyvi; max 256 token → giữ CHUNK_SIZE ≤ 600 |

**Dùng model nền PhoBERT đúng cách (bkai / keepitreal):** PhoBERT muốn input đã tách từ ("học sinh" → "học_sinh"). Thêm `pyvi` vào `requirements.txt`, rồi trong `rag.py` thêm:
```python
def _vi_seg(text: str) -> str:
    try:
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)
    except Exception:
        return text
```
và áp dụng cho CẢ passage lẫn query khi `"e5" not in EMBED_MODEL` — trong `build()`: `passages = [_vi_seg(c) for c in self.chunks]`, trong `_all_scores()`: `q_text = _vi_seg(query)`. (Chỉ tách từ bản đưa vào encode; `self.chunks` giữ nguyên văn để đưa cho LLM.)

**Quy tắc chọn:** đo cả 3 trên mock với bộ câu hỏi tự chế, model nào điểm retrieval tốt nhất (soi qa_log: context có chứa đáp án không) thì chốt TRƯỚC ngày thi. Giữa giờ thi đổi model = xóa vectordb + evaluate `document_received=False` = đốt 1/5 lượt nộp → chỉ đổi khi qa_log cho thấy retrieval lạc đề rõ ràng.

---

"""Bạn là chuyên gia giải trắc nghiệm dựa trên tài liệu. Quy tắc:
1. Ưu tiên thông tin trong TÀI LIỆU, phải dựa theo TÀI LIỆU mà trả lời. Tài liệu có thể lẫn đoạn không liên quan - hãy bỏ qua chúng.
2. Câu hỏi phủ định (KHÔNG / NGOẠI TRỪ / SAI): tìm phương án KHÔNG xuất hiện hoặc trái với tài liệu.
3. Câu hỏi về số lượng hoặc số liệu: liệt kê ngắn gọn các mục/con số tìm thấy trong tài liệu trước, rồi mới kết luận.
4. Nếu các phương án gần giống nhau, chọn phương án khớp CHÍNH XÁC nhất với câu chữ trong tài liệu.
5. Nếu tài liệu không đủ thông tin, suy luận hợp lý nhất và vẫn BẮT BUỘC chọn một đáp án.
Trả lời: giải thích tối đa 2 câu (KHÔNG lặp lại các lựa chọn), rồi dòng cuối cùng đúng định dạng:
Đáp án: X
(X là đúng MỘT ký tự trong A, B, C, D. Ví dụ dòng cuối hợp lệ: "Đáp án: B")"""