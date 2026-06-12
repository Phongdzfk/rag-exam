# ============================================================
#  make_testdoc.py - Tạo test_document.txt ~N MB để test tài liệu lớn
#  Dùng:  python make_testdoc.py        (mặc định 5MB)
#         python make_testdoc.py 10     (10MB)
#
#  Sinh từ chính các fact trong DOCUMENT mẫu của mock_teacher
#  -> bộ QUESTIONS trong mock vẫn trả lời được.
#  LƯU Ý: nội dung lặp lại nên file này CHỈ để đo thời gian
#  embed/upload + smoke test. Muốn đo chất lượng retrieval thật
#  thì dùng tài liệu thật (giáo trình PDF convert, Wikipedia...).
#  Xóa test_document.txt đi là mock quay về tài liệu mẫu nhỏ.
# ============================================================
import sys
import random

SEED = """
RAG (Retrieval-Augmented Generation) là kỹ thuật kết hợp truy xuất thông tin với mô hình sinh văn bản, giúp mô hình trả lời dựa trên tài liệu bên ngoài thay vì chỉ dựa vào kiến thức đã huấn luyện.
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
Mạng nơ-ron tích chập CNN thường dùng cho xử lý ảnh, trong khi Transformer thống trị xử lý ngôn ngữ tự nhiên.
Học sâu là nhánh của học máy sử dụng mạng nơ-ron nhiều tầng để học biểu diễn dữ liệu.
Gradient descent là thuật toán tối ưu cập nhật trọng số theo hướng ngược gradient của hàm mất mát.
Overfitting xảy ra khi mô hình học thuộc dữ liệu huấn luyện và kém tổng quát trên dữ liệu mới.
""".strip()

random.seed(0)
size_mb = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
sents = [s.strip() for s in SEED.split("\n") if s.strip()]

written, i = 0, 0
with open("test_document.txt", "w", encoding="utf-8") as f:
    while written < size_mb * 1_000_000:
        i += 1
        block = f"PHẦN {i}. " + " ".join(random.sample(sents, k=min(6, len(sents)))) + "\n\n"
        f.write(block)
        written += len(block)

print(f"✓ Đã tạo test_document.txt ~{written/1e6:.1f}MB ({i} phần).")
print("Mock teacher sẽ TỰ ĐỘNG dùng file này. Xóa file đi để quay về tài liệu mẫu nhỏ.")
