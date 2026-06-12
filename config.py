# ============================================================
#  CONFIG TRUNG TÂM - MỌI THỨ CẦN SỬA ĐỀU Ở ĐÂY
#  (Sửa xong nhớ RESTART server: Ctrl+C rồi chạy lại)
#  Bản này ĐÃ apply Tầng 1-4 (CoT, neighbor, per-option, rerank)
# ============================================================

# ---------- 1. THÔNG TIN CÁ NHÂN (BẮT BUỘC SỬA) ----------
STUDENT_ID = "B22DCXXXXX"          # MSSV VIẾT HOA của bạn
TEACHER_BASE = "http://192.168.50.218:8000/api/v1"   # CHECK LAI IP tren bang hom thi!
MY_SERVER_URL = "http://192.168.1.15:5000"           # IP may MINH + port (xem NOTE.md)

# ---------- 2. SERVER CỦA MÌNH ----------
HOST = "0.0.0.0"    # giữ nguyên để máy khác gọi vào được
PORT = 5000         # nếu trùng port thì đổi, nhớ đổi cả MY_SERVER_URL

# ---------- 3. LLM (gọi qua proxy của thầy) ----------
# THẦY DẶN: chỉ 4K token + ĐỪNG SPAM kẻo bị rate-limit.
# -> Pipeline này giữ đúng 1 call LLM/câu (tối đa 2 khi lỗi). KHÔNG thêm vote/multi-call.
LLM_MODEL = "gpt-4o-mini"
LLM_BASE_URL = None     # None = TEACHER_BASE + "/proxy" (khi thi). Test nhà: "http://localhost:11434/v1"
LLM_API_KEY = None      # None = dùng STUDENT_ID làm key (khi thi)
LLM_TIMEOUT = 40        # mỗi câu thầy cho 60s, để 40 còn dư retry
LLM_MAX_TOKENS = 160    # Tầng 1 CoT: đủ cho 1-2 câu giải thích + "Đáp án: X"
LLM_TEMPERATURE = 0.0
LLM_RETRY = 1           # CHỈ retry khi LỖI (không phải mọi câu) -> không tính là spam

# ---------- 4. EMBEDDING ----------
EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_MODEL_BACKUP = "bkai-foundation-models/vietnamese-bi-encoder"
MODEL_CACHE_DIR = "./models"
USE_EMBEDDINGS = True   # model loi -> tu fallback BM25-only (hoac set False tay)

# ---------- 5. RERANKER (Tầng 4) ----------
RERANK_MODEL = "BAAI/bge-reranker-base"
USE_RERANK = True        # tắt nhanh nếu máy lab quá chậm (xem elapsed trong qa_log)
RERANK_CANDIDATES = 20   # lấy thô 20 chunk rồi rerank xuống TOP_K

# ---------- 6. CHUNKING + RETRIEVAL ----------
# CHUNK_SIZE 800 cho tài liệu LỚN (5MB ~ 6-7k chunks, embed nhanh hơn 500)
# CHUNK_STRATEGY: "recursive" = tôn trọng ranh giới đoạn văn (tốt cho giáo trình
# có cấu trúc), "sentence" = cắt theo câu (bản gốc). ĐỔI LÀ PHẢI RE-EMBED:
# xóa storage/vectordb.pkl rồi upload lại tài liệu.
CHUNK_STRATEGY = "recursive"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 3               # số chunk CHÍNH sau rerank; mỗi chunk kéo theo hàng xóm
NEIGHBORS = 1           # Tầng 2: số chunk liền kề mỗi bên đưa kèm vào context
HYBRID_ALPHA = 0.6      # 0..1: trọng số embedding vs BM25

# ---------- 7. TOKEN BUDGET (4096 - thầy đã chốt) ----------
# ~120 prompt + ~250 câu hỏi + ~160 output CoT -> còn ~3500 token ≈ 5500 ký tự context
CONTEXT_CHAR_BUDGET = 5500

# ---------- 8. LƯU TRỮ ----------
STORAGE_PATH = "./storage/vectordb.pkl"
LOG_PATH = "./storage/qa_log.txt"