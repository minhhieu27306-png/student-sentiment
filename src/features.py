"""Trích xuất đặc trưng: Bag-of-Words, TF-IDF và Sentence Embeddings.

- ``bow``      : CountVectorizer, n-gram (1, 2)
- ``tfidf``    : TfidfVectorizer, n-gram (1, 2), sublinear_tf
- ``embedding``: sentence-transformers (mặc định ``paraphrase-multilingual-MiniLM-L12-v2``),
                 vector đã chuẩn hóa L2, được cache vào ``data/processed/*.npy``.

BoW và TF-IDF cho ma trận thưa (sparse); embedding cho ma trận dày (dense).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import config as C

# Giữ cả token một ký tự (vd. "à", "ạ"); mặc định của sklearn bỏ token ngắn hơn 2 ký tự.
TOKEN_PATTERN = r"(?u)\b\w+\b"


def make_vectorizer(kind: str):
    """Tạo vectorizer chưa fit cho ``bow`` hoặc ``tfidf``."""
    common = dict(ngram_range=(1, 2), min_df=2, token_pattern=TOKEN_PATTERN, lowercase=False)
    if kind == "bow":
        return CountVectorizer(**common)
    if kind == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **common)
    raise ValueError(f"Loại đặc trưng không hợp lệ cho vectorizer: {kind!r}")


class SentenceEmbedder:
    """Bao gói SentenceTransformer, import trễ vì thư viện này (PyTorch) khá nặng."""

    def __init__(self, model_name: str = C.EMBEDDING_MODEL, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Cần cài sentence-transformers: pip install sentence-transformers") from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Trả về mảng float32 (n, d), đã chuẩn hóa L2."""
        emb = self._load().encode(list(texts), batch_size=self.batch_size, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


def _cache_path(model_name: str, text_col: str, split: str, texts: list[str]) -> Path:
    digest = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:10]   # tránh dùng nhầm cache cũ
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return C.PROCESSED_DIR / f"emb_{safe}_{text_col}_{split}_{digest}.npy"


def embed_split(embedder: SentenceEmbedder, texts: list[str], text_col: str, split: str) -> np.ndarray:
    """Encode một split, dùng cache nếu đã có."""
    path = _cache_path(embedder.model_name, text_col, split, texts)
    if path.exists():
        return np.load(path)
    emb = embedder.encode(texts)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


@dataclass
class FeatureSet:
    """Kết quả trích xuất đặc trưng cho một loại đặc trưng."""
    kind: str
    X_train: object
    X_val: object
    X_test: object
    dense: bool
    text_col: str
    vectorizer: object | None = None      # bow / tfidf: vectorizer đã fit trên tập train
    embedding_model: str | None = None    # embedding: tên mô hình


def build_features(kind: str, data: dict[str, pd.DataFrame], text_col: str,
                   embedding_model: str = C.EMBEDDING_MODEL) -> FeatureSet:
    """Fit trên tập train (tránh rò rỉ dữ liệu) rồi biến đổi cả 3 tập."""
    tr, va, te = (data[s][text_col].tolist() for s in C.SPLITS)
    if kind in ("bow", "tfidf"):
        vec = make_vectorizer(kind)
        return FeatureSet(kind, vec.fit_transform(tr), vec.transform(va), vec.transform(te),
                          dense=False, text_col=text_col, vectorizer=vec)
    if kind == "embedding":
        embedder = SentenceEmbedder(embedding_model)
        return FeatureSet(kind, *(embed_split(embedder, texts, text_col, s)
                                  for texts, s in zip((tr, va, te), C.SPLITS)),
                          dense=True, text_col=text_col, embedding_model=embedding_model)
    raise ValueError(f"Loại đặc trưng không hợp lệ: {kind!r}")
