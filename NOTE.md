# 📒 NOTE ALL-IN-ONE — RAG COMPETITION

> Đọc file này là đủ để: chạy được → tối ưu điểm → tự fix bug lúc thi.

---

## 0. TÓM TẮT LUẬT THI (đọc lại cho nhớ)

- 10–15 phút đầu **CÒN MẠNG**: clone code, cài thư viện, tải model. Sau đó **CẮT MẠNG**, chỉ gọi được Teacher Server.
- Thi thật: **100 câu**, mỗi câu quá **60 giây** = 0 điểm câu đó, **tối đa 5 lần nộp** (evaluate). Quá 5 lần điểm cao cũng không tính → **ĐỪNG NỘP BỪA**.
- Điểm cao nhất trong các lần nộp = điểm chính.
- Proxy LLM giới hạn **2048–4096 tokens** → context phải gọn.
- Teacher Server: `http://192.168.50.218:8000/api/v1` (CHECK LẠI IP TRÊN BẢNG HÔM THI).

---

## 1. CHECKLIST 10–15 PHÚT ĐẦU (CÒN MẠNG) — LÀM ĐÚNG THỨ TỰ

```bash
# B1. Clone code về
git clone <link-repo-cua-ban>
cd rag-exam

# B2. Cài thư viện (nếu lab có sẵn python + pip)
pip install -r requirements.txt

# B3. TẢI MODEL NGAY (quan trọng nhất, tải ~vài trăm MB)
python warmup.py
# -> thấy "WARMUP XONG" là model đã nằm trong ./models, ngắt mạng vẫn chạy
```

**Trong lúc model đang tải, tranh thủ sửa `config.py`:**
1. `STUDENT_ID` = MSSV VIẾT HOA
2. `TEACHER_BASE` = IP thầy cho hôm đấy (nhớ có `/api/v1` ở cuối!)
3. `MY_SERVER_URL` = IP máy mình + `:5000`

**Cách lấy IP máy mình:**
- Windows: mở cmd → `ipconfig` → lấy dòng `IPv4 Address` của card đang nối LAN (vd `192.168.1.15`)
- Linux: `ip a` hoặc `hostname -I`
- ⚠ KHÔNG dùng `127.0.0.1` hay `localhost` — thầy gọi vào sẽ fail!

---

## 2. CHẠY THI (SAU KHI CẮT MẠNG)

Mở **2 cửa sổ terminal**:

**Terminal 1 — bật server (để chạy suốt buổi):**
```bash
python server.py
```
Thấy log `Student: <MSSV>` + load model OK là sống.

**Terminal 2 — điều khiển cuộc thi:**
```bash
python exam.py testllm      # test proxy LLM sống chưa (nên làm trước)
python exam.py testlocal    # test /upload /ask của chính mình (nên làm trước)

python exam.py register                # đăng ký
python exam.py evaluate                # LẦN ĐẦU: document_received=False
                                       #   thầy sẽ gửi tài liệu -> server mình embed
python exam.py result                  # theo dõi điểm + đang ở câu mấy

# Các lần nộp SAU (đã có vector db rồi, khỏi upload lại):
python exam.py reset                   # reset điểm
python exam.py evaluate --received     # document_received=True -> chỉ bơm câu hỏi
```

**Lưu ý luồng lần đầu:** gọi `evaluate` lần đầu mà báo lỗi gửi tài liệu là **BÌNH THƯỜNG** (server mình đang embed, thầy để timeout upload 2 phút). Nhìn Terminal 1, khi thấy `[UPLOAD] ✓ Xong` + `Đã lưu VectorDB` là tài liệu đã vào. Từ đó về sau luôn dùng `--received`.

**Restart server không mất dữ liệu:** VectorDB tự lưu vào `storage/vectordb.pkl`, bật lại server là tự load. Sửa code thoải mái, Ctrl+C → `python server.py` → `evaluate --received`.

---

## 3. CHIẾN THUẬT NỘP (CHỈ CÓ 5 LẦN!)

1. **Lần 1:** chạy nguyên bản để lấy điểm nền + lấy **log câu hỏi**.
2. Mở `storage/qa_log.txt` → đọc các câu trả lời sai/lạ: context lấy ra có liên quan không? LLM trả lời gì?
   - Context lạc đề → chỉnh retrieval (mục 4).
   - Context đúng mà LLM chọn sai → chỉnh prompt (mục 4).
3. Sửa → **test lại bằng `python exam.py testlocal`** hoặc tự POST vài câu trong log vào `/ask` → ổn rồi mới nộp lần 2.
4. Giữ lại ít nhất 1 lần nộp dự phòng cuối giờ.
5. Mỗi lần nộp xong ghi lại điểm + thay đổi gì (để biết cái gì work, lỡ tệ hơn thì revert).

---

## 4. TỐI ƯU Ở ĐÂU — CÁC KNOB

### 4.1. `config.py` (sửa xong PHẢI restart server)

| Knob | Mặc định | Khi nào chỉnh |
|---|---|---|
| `TOP_K` | 4 | Trả lời thiếu thông tin → tăng 5–6. Context nhiễu/tràn token → giảm 3 |
| `CHUNK_SIZE` | 500 | Câu hỏi chi tiết, đáp án nằm trong 1 câu → giảm 300–400. Hỏi tổng hợp → tăng 700–800 |
| `CHUNK_OVERLAP` | 100 | Đáp án hay bị cắt đôi giữa 2 chunk → tăng 150–200 |
| `CONTEXT_CHAR_BUDGET` | 3500 | Thầy xác nhận seq len 4096 → tăng lên 6500–7000. Bị lỗi tràn token → giảm 2500 |
| `HYBRID_ALPHA` | 0.6 | Câu hỏi nhiều thuật ngữ/tên riêng chính xác → giảm 0.4 (thiên BM25). Câu hỏi diễn đạt khác tài liệu → tăng 0.7–0.8 (thiên embedding) |
| `LLM_TIMEOUT` | 40 | Proxy chậm (nhiều người gọi cùng lúc) → có thể giảm 30 để kịp retry trong 60s |

⚠ Đổi `CHUNK_SIZE`/`CHUNK_OVERLAP` thì phải **xóa `storage/vectordb.pkl` + evaluate lại với `document_received=False`** để chunk lại từ đầu. Tốn 1 lần nộp → cân nhắc, chỉnh prompt/TOP_K trước.

### 4.2. Prompt — đầu file `llm.py` (chỗ ăn điểm nhất, không tốn re-embed)

Mẹo đã kiểm chứng:
- Giữ system prompt **ngắn**. Mỗi token thừa là bớt chỗ cho context.
- Ép định dạng cứng: *"CHỈ trả lời đúng 1 ký tự A/B/C/D"* — đã có sẵn.
- Nếu LLM hay trả lời "Đáp án: B" → vẫn ổn, `parse_answer` tự bóc được chữ B.
- Có thể thêm 1 câu: `"Nếu nhiều đáp án có vẻ đúng, chọn đáp án ĐÚNG NHẤT theo tài liệu."`
- Thử đổi thứ tự: đặt CÂU HỎI trước TÀI LIỆU nếu thấy LLM bỏ qua câu hỏi (ít gặp với context ngắn).

### 4.3. Logic xử lý — `server.py` / `rag.py`
- Đáp án trắc nghiệm nằm ngay trong câu hỏi (A./B./C./D.) → retrieval đang search bằng cả câu hỏi + options, thường tốt. Nếu muốn thử: chỉ search bằng phần câu hỏi (cắt trước "A.").
- `storage/qa_log.txt` ghi đủ: câu hỏi, chunks lấy ra, raw LLM, thời gian → đây là "mắt" của bạn lúc tune.

---

## 5. FIX BUG NHANH — TRA THEO TRIỆU CHỨNG

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| `register` báo 404/Connection refused | Sai IP thầy hoặc **thiếu `/api/v1`** | Check `TEACHER_BASE` trong config — lỗi kinh điển thầy đã cảnh báo |
| `register` OK nhưng thầy không gọi được vào máy mình | **Firewall Windows chặn** hoặc khai sai `MY_SERVER_URL` | 1) Tắt firewall: `netsh advfirewall set allprofiles state off` (cmd Admin) hoặc Allow Python khi popup hiện. 2) Check IP bằng `ipconfig`. 3) Nhờ bạn cùng LAN `curl http://<IP-mình>:5000/health` |
| Model tải lỗi / hết giờ mạng chưa tải xong | Mạng chậm | Đặt `USE_EMBEDDINGS = False` trong config → chạy BM25-only, vẫn có điểm. (Phòng trước: thi thử hôm T6 tải sẵn, model nằm `./models`, hôm thi thật cùng máy thì khỏi tải) |
| `OSError ... model not found` khi ĐÃ cắt mạng | Model chưa cache | Như trên: `USE_EMBEDDINGS = False` |
| LLM lỗi 401/403 | API key sai | `STUDENT_ID` phải là MSSV **VIẾT HOA**, đúng mã trên lớp |
| LLM lỗi context length exceeded | Tràn token | Giảm `CONTEXT_CHAR_BUDGET` (2500) hoặc `TOP_K` (3). Code đã tự retry với nửa context |
| Câu nào cũng quá 60s | Proxy nghẽn / embed query chậm | Giảm `LLM_TIMEOUT`=25 + `LLM_RETRY`=1; encode 1 query chỉ ~0.1s nên thủ phạm thường là proxy — hỏi thầy |
| Trả lời toàn "A" | LLM chết hẳn, đang chạy fallback | Xem `qa_log.txt` dòng `LLM ERROR` → fix theo lỗi đó (thường 401 hoặc mạng) |
| Upload báo 422 / không tìm thấy text | Thầy đổi tên biến lạ hơn dự kiến | Xem log `Keys nhận được: [...]` ở Terminal 1 → thêm tên key đó vào `TEXT_KEYS` đầu `server.py` → restart |
| Sửa config xong không thấy thay đổi | Quên restart | Ctrl+C → `python server.py` |
| `Address already in use` | Server cũ chưa tắt | Windows: `netstat -ano \| findstr 5000` → `taskkill /PID <pid> /F`. Hoặc đổi `PORT` + `MY_SERVER_URL` + register lại |
| Bật lại server mất hết tài liệu | Không thể — đã tự lưu | Nếu vẫn mất: check file `storage/vectordb.pkl` tồn tại không; lỡ xóa thì evaluate `document_received=False` lại |
| evaluate lần đầu báo lỗi upload | **BÌNH THƯỜNG** | Server mình đang embed (timeout thầy 2 phút). Đợi Terminal 1 báo xong → lần sau `--received` |

---

## 6. TEST Ở NHÀ / TRÊN LAPTOP CÁ NHÂN — THI THỬ FULL BẰNG MOCK

Trong folder có sẵn `mock_teacher.py` — giả lập Teacher Server đầy đủ: register, evaluate (gửi tài liệu mẫu + bơm 10 câu + chấm điểm), reset, result, cả proxy LLM.

```bash
pip install -r requirements.txt
python warmup.py                                   # tải model (cần mạng lần đầu)
```
Sửa tạm `config.py`: `TEACHER_BASE = "http://127.0.0.1:8000/api/v1"` rồi mở 3 terminal:
```bash
# T1: thầy giả
python -m uvicorn mock_teacher:app --port 8000
# T2: server của mình
python server.py
# T3: thi
python exam.py register
python exam.py evaluate              # -> in final_score /10 + log từng câu bên T1
python exam.py reset
python exam.py evaluate --received   # nộp lại không upload
```
Xem `storage/qa_log.txt` để tập tune như thi thật. **Thi xong nhớ đổi `TEACHER_BASE` về IP thầy!**

**LLM khi test ở nhà** — mock proxy hoạt động theo 2 chế độ:
- Có **Ollama** chạy sẵn (`ollama pull qwen2.5:3b`) → mock tự forward sang Ollama, giống proxy thật nhất.
- Không có → mock dùng heuristic (chọn đáp án trùng từ với tài liệu) — đủ để test luồng, đo thời gian, nhưng điểm không phản ánh chất lượng prompt.
- Muốn test prompt với LLM xịn: set trong config `LLM_BASE_URL="https://api.openai.com/v1"` + `LLM_API_KEY="sk-..."` (hoặc Groq/OpenRouter free tier). Nhớ trả về `None` trước khi thi.

Tự thay `DOCUMENT` và `QUESTIONS` trong `mock_teacher.py` bằng tài liệu môn học của bạn để luyện sát đề hơn.

## 6b. CHIẾN LƯỢC MODEL EMBEDDING

- Mặc định `multilingual-e5-small`: đủ tốt cho 1 document vài trăm chunk, nhanh trên CPU. Điểm thường rơi ở prompt/LLM chứ không phải retrieval.
- `warmup.py` tải sẵn cả model dự phòng `bkai/vietnamese-bi-encoder`.
- **Đổi model GIỮA GIỜ THI là đắt**: phải xóa `storage/vectordb.pkl`, restart, evaluate lại `document_received=False` → tốn 1/5 lượt nộp + thời gian embed lại. Sau khi cắt mạng KHÔNG tải được model mới — chỉ đổi được sang model đã warmup.
- Quy tắc quyết định: mở `qa_log.txt` → nếu CONTEXTS lấy ra **lạc đề** so với câu hỏi → mới cân nhắc đổi model (hoặc thử tăng `HYBRID_ALPHA` trước, miễn phí). Nếu context đúng mà đáp án sai → sửa prompt, đừng động vào model.

---

## 7. TIMELINE MẪU HÔM THI

| Thời điểm | Việc |
|---|---|
| 0–3 phút | clone + `pip install` |
| song song | sửa `config.py` (MSSV, IP thầy, IP mình) |
| 3–12 phút | `python warmup.py` (tải model) |
| 12–15 phút | `python server.py` + `python exam.py testllm` + `testlocal` |
| Cắt mạng | `register` → `evaluate` (lần 1, False) → `result` theo dõi |
| Sau lần 1 | đọc `qa_log.txt` → tune prompt/knob → test → `reset` → `evaluate --received` |
| Cuối giờ | chốt cấu hình tốt nhất, còn dư lượt thì nộp chốt lần cuối |
