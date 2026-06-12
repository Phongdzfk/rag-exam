# ============================================================
#  rag.py - Chunking + Embedding + VectorDB + Hybrid Retrieval
#  Bình thường KHÔNG cần sửa file này. Knob nằm hết ở config.py
# ============================================================
import os
import re
import pickle
import numpy as np

import config

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

_model = None


def get_model():
    """Lazy-load embedding model. Nếu lỗi (chưa tải/mất mạng) -> tự chuyển BM25-only."""
    global _model
    if not config.USE_EMBEDDINGS:
        return None
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[RAG] Đang load embedding model: {config.EMBED_MODEL} ...")
            _model = SentenceTransformer(config.EMBED_MODEL, cache_folder=config.MODEL_CACHE_DIR)
            print("[RAG] Load model xong.")
        except Exception as e:
            print(f"[RAG] ⚠ Load model THẤT BẠI: {e}")
            print("[RAG] -> TỰ CHUYỂN sang chế độ BM25-only, vẫn thi được bình thường.")
            config.USE_EMBEDDINGS = False
            return None
    return _model


# ------------------------- CHUNKING -------------------------
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\n;])\s+")

# Thứ tự ranh giới ưu tiên cho recursive: đoạn văn -> dòng -> câu -> từ
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, seps):
    """Cắt đệ quy: thử ranh giới to nhất trước, mảnh nào còn dài quá
    CHUNK_SIZE thì đệ quy xuống ranh giới nhỏ hơn."""
    if len(text) <= config.CHUNK_SIZE:
        return [text]
    if not seps:  # hết ranh giới -> cắt cứng
        return [text[i : i + config.CHUNK_SIZE]
                for i in range(0, len(text), config.CHUNK_SIZE)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    if len(parts) == 1:  # ranh giới này không có trong text -> thử cái nhỏ hơn
        return _recursive_split(text, rest)
    pieces = []
    for p in parts:
        if len(p) <= config.CHUNK_SIZE:
            pieces.append(p)
        else:
            pieces.extend(_recursive_split(p, rest))
    return pieces


def _chunk_recursive(text: str):
    """Recursive chunking: tôn trọng cấu trúc đoạn văn tối đa, gộp + overlap."""
    pieces = [p.strip() for p in _recursive_split(text, _SEPARATORS) if p.strip()]
    chunks, cur = [], ""
    for p in pieces:
        if len(cur) + len(p) + 1 <= config.CHUNK_SIZE:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-config.CHUNK_OVERLAP :] if config.CHUNK_OVERLAP > 0 else ""
            if " " in tail:
                tail = tail.split(" ", 1)[1]
            cur = (tail + " " + p).strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) > 20]


def chunk_text(text: str):
    """Dispatch theo config.CHUNK_STRATEGY: "recursive" | "sentence".
    LƯU Ý: đổi strategy là thay đổi INDEX-TIME -> xóa storage/vectordb.pkl + re-embed."""
    text = text.replace("\r\n", "\n").strip()
    if getattr(config, "CHUNK_STRATEGY", "sentence") == "recursive":
        return _chunk_recursive(text)
    return _chunk_sentence(text)


def _chunk_sentence(text: str):
    """Bản cũ: cắt theo câu, gộp tới CHUNK_SIZE ký tự, có overlap."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

    chunks = []
    cur = ""
    for s in sents:
        # câu đơn lẻ quá dài -> cắt cứng
        while len(s) > config.CHUNK_SIZE:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(s[: config.CHUNK_SIZE])
            s = s[config.CHUNK_SIZE - config.CHUNK_OVERLAP :]
        if len(cur) + len(s) + 1 <= config.CHUNK_SIZE:
            cur = (cur + " " + s).strip()
        else:
            chunks.append(cur)
            # overlap: giữ lại đuôi chunk trước (cắt tại ranh giới từ)
            tail = cur[-config.CHUNK_OVERLAP :] if config.CHUNK_OVERLAP > 0 else ""
            if " " in tail:
                tail = tail.split(" ", 1)[1]
            cur = (tail + " " + s).strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) > 20]  # bỏ chunk rác quá ngắn


def _tokenize(s: str):
    return re.findall(r"\w+", s.lower())


# ------------------------- VECTOR STORE -------------------------
class VectorStore:
    def __init__(self):
        self.chunks = []
        self.emb = None      # np.ndarray (n, d), đã normalize
        self.bm25 = None

    @property
    def ready(self):
        return len(self.chunks) > 0

    def build(self, text: str):
        self.chunks = chunk_text(text)
        print(f"[RAG] Chunked thành {len(self.chunks)} chunks.")

        model = get_model()
        if model is not None:
            # model họ e5 yêu cầu prefix "passage: " / "query: ", model khác thì không
            is_e5 = "e5" in config.EMBED_MODEL.lower()
            passages = [("passage: " + c) if is_e5 else c for c in self.chunks]
            self.emb = model.encode(
                passages, batch_size=64, normalize_embeddings=True,
                show_progress_bar=True, convert_to_numpy=True,
            )
        else:
            self.emb = None

        if BM25Okapi is not None:
            self.bm25 = BM25Okapi([_tokenize(c) for c in self.chunks])
        print("[RAG] Build VectorDB xong.")

    # ---------- persistence ----------
    def save(self, path=None):
        path = path or config.STORAGE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "emb": self.emb}, f)
        print(f"[RAG] Đã lưu VectorDB -> {path}")

    def load(self, path=None) -> bool:
        path = path or config.STORAGE_PATH
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.emb = data.get("emb")
        if BM25Okapi is not None and self.chunks:
            self.bm25 = BM25Okapi([_tokenize(c) for c in self.chunks])
        print(f"[RAG] Đã load VectorDB từ {path} ({len(self.chunks)} chunks).")
        return True

    # ---------- retrieval ----------
    def _all_scores(self, query: str):
        """Điểm hybrid cho TẤT CẢ chunks: alpha*cosine + (1-alpha)*BM25."""
        n = len(self.chunks)
        scores = np.zeros(n, dtype=np.float32)
        alpha = config.HYBRID_ALPHA

        model = get_model()
        if model is not None and self.emb is not None:
            is_e5 = "e5" in config.EMBED_MODEL.lower()
            q_text = ("query: " + query) if is_e5 else query
            q = model.encode([q_text], normalize_embeddings=True, convert_to_numpy=True)[0]
            cos = self.emb @ q  # đã normalize -> dot = cosine, range ~[-1,1]
            cos = (cos + 1) / 2
            scores += alpha * cos
        else:
            alpha = 0.0  # không có embedding -> dồn hết cho BM25

        if self.bm25 is not None:
            bm = np.array(self.bm25.get_scores(_tokenize(query)), dtype=np.float32)
            bm = bm - bm.min()          # BM25 có thể âm -> min-max normalize
            if bm.max() > 0:
                bm = bm / bm.max()
            scores += (1 - alpha) * bm

        return scores

    def search(self, query: str, top_k=None):
        """Giữ tương thích cũ: top_k chunk theo điểm hybrid."""
        if not self.ready:
            return []
        scores = self._all_scores(query)
        idx = np.argsort(-scores)[: (top_k or config.TOP_K)]
        return [(self.chunks[i], float(scores[i])) for i in idx]


# store toàn cục dùng chung cho server
STORE = VectorStore()


# ============ TẦNG 4: RERANKER (cross-encoder) ============
_reranker = None


def get_reranker():
    """Lỗi -> tự tắt USE_RERANK, không bao giờ crash giữa giờ thi."""
    global _reranker
    if not getattr(config, "USE_RERANK", False):
        return None
    if _reranker is None:
        try:
            os.environ.setdefault("HF_HOME", config.MODEL_CACHE_DIR)
            from sentence_transformers import CrossEncoder
            print(f"[RAG] Đang load reranker: {config.RERANK_MODEL} ...")
            _reranker = CrossEncoder(config.RERANK_MODEL, max_length=512)
            print("[RAG] Reranker sẵn sàng.")
        except Exception as e:
            print(f"[RAG] ⚠ Reranker lỗi ({e}) -> tắt rerank, chạy bình thường.")
            config.USE_RERANK = False
            return None
    return _reranker


# ============ TẦNG 3: RETRIEVAL THEO TỪNG ĐÁP ÁN ============
def multi_query_scores(store: VectorStore, question: str):
    """Cộng dồn điểm: câu hỏi gốc + (thân câu hỏi + từng option)."""
    stem = re.split(r"\s+(?=A[\.\):])", question, maxsplit=1)[0]
    opts = re.findall(r"([A-D])[\.\):\-]\s*(.+?)(?=\s+[A-D][\.\):\-]\s|$)", question, re.S)
    queries = [question] + [f"{stem} {t.strip()}" for _, t in opts]
    total = None
    for q in queries:
        s = store._all_scores(q)
        total = s if total is None else total + s
    return total


# ============ PIPELINE ĐẦY ĐỦ (server.py gọi hàm này) ============
def retrieve(question: str) -> list[str]:
    """Tầng 3 (per-option) -> top thô -> Tầng 4 (rerank) -> Tầng 2 (neighbors)."""
    if not STORE.ready:
        return []
    scores = multi_query_scores(STORE, question)
    n = len(STORE.chunks)

    rr = get_reranker()
    k_raw = config.RERANK_CANDIDATES if rr is not None else config.TOP_K
    cand = np.argsort(-scores)[:k_raw].tolist()

    if rr is not None:
        rs = rr.predict([(question, STORE.chunks[i]) for i in cand])
        order = sorted(range(len(cand)), key=lambda j: -float(rs[j]))
        cand = [cand[j] for j in order[: config.TOP_K]]

    # Tầng 2: kèm chunk hàng xóm mỗi bên, xếp theo thứ tự văn bản cho mạch lạc
    nb = getattr(config, "NEIGHBORS", 1)
    seen, picked = set(), []
    for i in cand:
        for j in range(max(0, i - nb), min(n, i + nb + 1)):
            if j not in seen:
                seen.add(j)
                picked.append(j)
    picked.sort()
    return [STORE.chunks[j] for j in picked]