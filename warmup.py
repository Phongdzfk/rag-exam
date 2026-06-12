# ============================================================
#  warmup.py - CHẠY NGAY KHI CÒN MẠNG (10-15 phút đầu giờ thi)
#  Tải embedding model về ./models + kiểm tra import đủ thư viện.
# ============================================================
import time
import config

t0 = time.time()
print("=" * 50)
print("1) Kiểm tra thư viện ...")
import fastapi, uvicorn, numpy, requests          # noqa
from openai import OpenAI                          # noqa
try:
    from rank_bm25 import BM25Okapi                # noqa
    print("   ✓ rank_bm25 OK")
except ImportError:
    print("   ⚠ thiếu rank_bm25 (pip install rank_bm25) - vẫn chạy được nhưng nên có")
print("   ✓ fastapi / uvicorn / numpy / requests / openai OK")

print(f"\n2) Tải embedding model: {config.EMBED_MODEL}")
print("   (lần đầu ~vài trăm MB, các lần sau load từ ./models, KHÔNG cần mạng)")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(config.EMBED_MODEL, cache_folder=config.MODEL_CACHE_DIR)

print("\n3) Encode thử ...")
v = model.encode(["query: xin chào", "passage: hello"], normalize_embeddings=True)
print(f"   ✓ Encode OK, dim = {v.shape[1]}")

backup = getattr(config, "EMBED_MODEL_BACKUP", None)
if backup:
    print(f"\n4) Tải model DỰ PHÒNG: {backup} (lỗi cũng không sao)")
    try:
        SentenceTransformer(backup, cache_folder=config.MODEL_CACHE_DIR)
        print("   ✓ Backup OK - giữa giờ thi có thể đổi EMBED_MODEL sang model này.")
    except Exception as e:
        print(f"   ⚠ Backup tải lỗi ({e}) - vẫn thi bình thường với model chính.")

print(f"\n5) Tải reranker: {getattr(config, 'RERANK_MODEL', None)}")
try:
    import os
    os.environ.setdefault("HF_HOME", config.MODEL_CACHE_DIR)
    from sentence_transformers import CrossEncoder
    CrossEncoder(config.RERANK_MODEL, max_length=512)
    print("   ✓ Reranker đã tải.")
except Exception as e:
    print(f"   ⚠ Reranker tải lỗi: {e} - set USE_RERANK=False là vẫn thi bình thường.")

print("=" * 50)
print(f"✓✓ WARMUP XONG trong {time.time()-t0:.0f}s. Có thể ngắt mạng an toàn.")
print("Tiếp theo: sửa config.py (MSSV, IP) rồi chạy:  python server.py")
