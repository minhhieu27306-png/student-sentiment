"""Tiền xử lý văn bản tiếng Việt.

Quy trình (thứ tự quan trọng):
    1. Chuẩn hóa Unicode NFC (tránh cùng một chữ nhưng khác dạng mã hóa dấu).
    2. Giải mã HTML entity (``&quot;``, ``&#93;``), xóa URL.
    3. Chữ thường (lowercasing).
    4. Xóa số, dấu câu, ký tự đặc biệt; rút gọn chữ kéo dài ("quáaaa" -> "quá").
    5. Chuẩn hóa teencode / từ viết tắt ("ko" -> "không", "đc" -> "được", ...).
    6. Gộp khoảng trắng thừa.
    7. Tách từ (word segmentation) bằng pyvi / underthesea: "giảng viên" -> "giảng_viên".

Hai cột đầu ra:
    - ``text_clean``: đã làm sạch nhưng CHƯA tách từ (dùng cho Sentence Embeddings đa ngôn ngữ).
    - ``text_seg``  : đã làm sạch và tách từ (dùng cho BoW / TF-IDF).

Chạy độc lập (tạo data/processed/{train,validation,test}.csv):
    python -m src.preprocess
    python -m src.preprocess --tokenizer underthesea
"""
from __future__ import annotations

import argparse
import html
import re
import unicodedata
import warnings
from functools import lru_cache
from typing import Callable

import pandas as pd
from tqdm import tqdm

from . import config as C
from .data_loader import load_splits

# --------------------------------------------------------------------------- #
# Từ điển teencode / viết tắt phổ biến (chỉ gồm các trường hợp ít gây nhập nhằng).
# Bạn có thể bổ sung thêm sau khi xem lỗi dự đoán của mô hình.
# --------------------------------------------------------------------------- #
TEENCODE: dict[str, str] = {
    # phủ định
    "ko": "không", "k": "không", "kh": "không", "khg": "không", "kg": "không",
    "hok": "không", "hong": "không", "hông": "không", "khong": "không", "kô": "không",
    # từ chức năng
    "dc": "được", "đc": "được", "duoc": "được",
    "vs": "với", "cx": "cũng", "cũg": "cũng", "j": "gì", "ji": "gì",
    "r": "rồi", "rùi": "rồi", "z": "vậy", "zậy": "vậy", "dzậy": "vậy",
    "wa": "quá", "wá": "quá", "qá": "quá", "wuá": "quá",
    "nhìu": "nhiều", "nhiu": "nhiều", "bít": "biết", "bik": "biết",
    "thik": "thích", "thix": "thích", "iu": "yêu", "dzui": "vui", "zui": "vui",
    "mik": "mình", "mn": "mọi người", "mng": "mọi người", "ntn": "như thế nào",
    "bt": "bình thường", "bth": "bình thường", "trc": "trước",
    "oke": "ok", "okie": "ok", "okela": "ok",
    "thanks": "cảm ơn", "thank": "cảm ơn", "thks": "cảm ơn", "tks": "cảm ơn",
    # từ viết tắt trong môi trường đại học
    "sv": "sinh viên", "gv": "giảng viên", "svien": "sinh viên", "gvien": "giảng viên",
    "đh": "đại học", "dh": "đại học", "ctrinh": "chương trình", "ctr": "chương trình",
    "ctđt": "chương trình đào tạo", "ctdt": "chương trình đào tạo",
    "csvc": "cơ sở vật chất", "tg": "thời gian", "tgian": "thời gian",
    "lt": "lý thuyết", "sgk": "sách giáo khoa",
}

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_NUM_UNDERSCORE_RE = re.compile(r"[\d_]+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_REPEAT_RE = re.compile(r"(\w)\1{2,}")   # 3 ký tự giống nhau liên tiếp trở lên
_SPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Các bước tiền xử lý
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Bước 1-4 và 6: chuẩn hóa Unicode, lowercase, xóa số/dấu câu/ký tự đặc biệt."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = html.unescape(text)
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    text = _NUM_UNDERSCORE_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _REPEAT_RE.sub(r"\1", text)
    return _SPACE_RE.sub(" ", text).strip()


def normalize_teencode(text: str, mapping: dict[str, str] | None = None) -> str:
    """Bước 5: thay từng token theo từ điển teencode."""
    mapping = TEENCODE if mapping is None else mapping
    return " ".join(mapping.get(tok, tok) for tok in text.split())


@lru_cache(maxsize=None)
def _get_segmenter(tokenizer: str) -> Callable[[str], str]:
    """Tạo (và cache) hàm tách từ; import trễ để không bắt buộc cài cả hai thư viện."""
    if tokenizer == "none":
        return lambda s: s
    if tokenizer == "pyvi":
        with warnings.catch_warnings():       # pyvi sinh SyntaxWarning vô hại khi biên dịch lần đầu
            warnings.simplefilter("ignore")
            from pyvi import ViTokenizer
        return ViTokenizer.tokenize
    if tokenizer == "underthesea":
        from underthesea import word_tokenize
        return lambda s: word_tokenize(s, format="text")
    raise ValueError(f"Tokenizer không hợp lệ: {tokenizer!r} (chọn 'pyvi', 'underthesea' hoặc 'none').")


def segment_words(text: str, tokenizer: str = C.DEFAULT_TOKENIZER) -> str:
    """Bước 7: tách từ tiếng Việt, nối âm tiết của cùng một từ bằng dấu gạch dưới."""
    if not text.strip():
        return ""
    return _SPACE_RE.sub(" ", _get_segmenter(tokenizer)(text)).strip()


def preprocess_text(text: str, tokenizer: str = C.DEFAULT_TOKENIZER, segment: bool = True) -> str:
    """Tiền xử lý một câu. ``segment=False`` trả về ``text_clean``, ``segment=True`` trả về ``text_seg``."""
    cleaned = normalize_teencode(clean_text(text))
    return segment_words(cleaned, tokenizer) if segment else cleaned


def preprocess_dataframe(df: pd.DataFrame, tokenizer: str = C.DEFAULT_TOKENIZER,
                         desc: str = "") -> pd.DataFrame:
    """Thêm hai cột ``text_clean`` và ``text_seg`` vào bảng dữ liệu."""
    out = df.copy()
    out["text_clean"] = [normalize_teencode(clean_text(t)) for t in out[C.TEXT_COL]]
    out["text_seg"] = [segment_words(t, tokenizer) for t in tqdm(out["text_clean"], desc=desc, leave=False)]
    return out


# --------------------------------------------------------------------------- #
# Lưu / nạp dữ liệu đã xử lý
# --------------------------------------------------------------------------- #
def process_and_save(tokenizer: str = C.DEFAULT_TOKENIZER) -> dict[str, pd.DataFrame]:
    """Nạp dữ liệu thô -> tiền xử lý -> lưu vào ``data/processed/``."""
    C.ensure_dirs()
    raw = load_splits()
    processed = {}
    for split, df in raw.items():
        processed[split] = preprocess_dataframe(df, tokenizer, desc=f"Tiền xử lý {split}")
        processed[split].to_csv(C.PROCESSED_DIR / f"{split}.csv", index=False, encoding="utf-8")
        print(f"  đã lưu {C.PROCESSED_DIR / (split + '.csv')}  ({len(df):,} mẫu)")
    (C.PROCESSED_DIR / "tokenizer.txt").write_text(tokenizer, encoding="utf-8")
    return processed


def load_processed() -> dict[str, pd.DataFrame]:
    """Nạp dữ liệu đã tiền xử lý (chạy ``python -m src.preprocess`` trước)."""
    out = {}
    for split in C.SPLITS:
        path = C.PROCESSED_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Chưa có {path}. Hãy chạy:  python -m src.preprocess")
        out[split] = pd.read_csv(path, keep_default_na=False)   # giữ chuỗi rỗng thay vì NaN
    return out


def saved_tokenizer() -> str:
    """Tên bộ tách từ đã dùng khi tạo dữ liệu processed."""
    path = C.PROCESSED_DIR / "tokenizer.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else C.DEFAULT_TOKENIZER


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lý UIT-VSFC.")
    parser.add_argument("--tokenizer", choices=["pyvi", "underthesea", "none"], default=C.DEFAULT_TOKENIZER)
    args = parser.parse_args()
    print(f"Tiền xử lý với bộ tách từ: {args.tokenizer}")
    processed = process_and_save(args.tokenizer)
    sample = processed["train"].head(5)
    for _, row in sample.iterrows():
        print("\n  gốc   :", row[C.TEXT_COL])
        print("  sạch  :", row["text_clean"])
        print("  tách  :", row["text_seg"])


if __name__ == "__main__":
    main()
