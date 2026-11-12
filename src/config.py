"""Cấu hình trung tâm của dự án: đường dẫn, tên nhãn, màu sắc, siêu tham số chung.

Mọi module khác đều import từ đây để tránh "hard-code" đường dẫn hay tên nhãn.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Đường dẫn
# --------------------------------------------------------------------------- #
# ROOT: thư mục gốc của dự án (chứa app.py, requirements.txt, ...).
ROOT = Path(__file__).resolve().parents[1]

# BASE: nơi chứa data/, models/, results/, report/.
# Mặc định trùng ROOT. Biến môi trường VSFC_WORKDIR cho phép trỏ sang thư mục khác
# (ví dụ để chạy thử nghiệm nhanh mà không đụng vào dữ liệu thật).
BASE = Path(os.environ.get("VSFC_WORKDIR", ROOT)).resolve()

DATA_DIR = BASE / "data"
RAW_DIR = DATA_DIR / "raw"              # dữ liệu thô tải về (train/validation/test .csv)
PROCESSED_DIR = DATA_DIR / "processed"  # dữ liệu đã tiền xử lý + cache embedding
MODELS_DIR = BASE / "models"            # mô hình đã huấn luyện (.pkl)
RESULTS_DIR = BASE / "results"          # bảng số liệu (.csv/.json) và hình
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = BASE / "report"            # báo cáo LaTeX
REPORT_TABLES_DIR = REPORT_DIR / "tables"
REPORT_FIGURES_DIR = REPORT_DIR / "figures"


def ensure_dirs() -> None:
    """Tạo các thư mục đầu ra nếu chưa tồn tại."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR,
              REPORT_TABLES_DIR, REPORT_FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Bộ dữ liệu
# --------------------------------------------------------------------------- #
HF_DATASET = "uitnlp/vietnamese_students_feedback"
SPLITS = ("train", "validation", "test")
EXPECTED_SIZES = {"train": 11426, "validation": 1583, "test": 3166}  # theo dataset card

TEXT_COL = "sentence"
TASKS = ("sentiment", "topic")

# Thứ tự phải khớp với chỉ số nhãn (0, 1, 2, ...) của dataset.
LABEL_NAMES = {
    "sentiment": ["negative", "neutral", "positive"],
    "topic": ["lecturer", "training_program", "facility", "others"],
}
LABEL_VI = {
    "sentiment": ["Tiêu cực", "Trung tính", "Tích cực"],
    "topic": ["Giảng viên", "Chương trình đào tạo", "Cơ sở vật chất", "Khác"],
}
TASK_VI = {"sentiment": "Cảm xúc", "topic": "Chủ đề"}

# Bảng màu thân thiện với người mù màu (Okabe-Ito) dùng cho biểu đồ.
LABEL_COLORS = {
    "sentiment": ["#D55E00", "#E69F00", "#009E73"],
    "topic": ["#0072B2", "#CC79A7", "#E69F00", "#7F7F7F"],
}

# --------------------------------------------------------------------------- #
# Tiền xử lý & mô hình
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
DEFAULT_TOKENIZER = "pyvi"  # "pyvi" | "underthesea" | "none"

# Mô hình embedding đa ngôn ngữ (hỗ trợ tiếng Việt), nhỏ gọn, chạy được trên CPU.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

FEATURE_KINDS = ("bow", "tfidf", "embedding")
FEATURE_DISPLAY = {"bow": "BoW", "tfidf": "TF-IDF", "embedding": "Embedding"}
MODEL_NAMES = ("NaiveBayes", "LogisticRegression", "LinearSVM", "RandomForest", "KNN")
MODEL_DISPLAY = {
    "NaiveBayes": "Naive Bayes",
    "LogisticRegression": "Logistic Regression",
    "LinearSVM": "Linear SVM",
    "RandomForest": "Random Forest",
    "KNN": "KNN",
}
