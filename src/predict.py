"""Nạp mô hình đã huấn luyện và dự đoán cho câu phản hồi mới (dùng bởi app.py và evaluate.py)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from . import config as C
from .features import SentenceEmbedder
from .preprocess import preprocess_text


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TaskModel:
    """Mô hình cho một bài toán (sentiment hoặc topic): vectorizer/embedder + bộ phân loại."""

    def __init__(self, task: str, models_dir: Path = C.MODELS_DIR):
        meta_path = models_dir / f"{task}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Chưa có mô hình '{task}' trong {models_dir}. Hãy chạy:  python -m src.train")
        self.task = task
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.model = joblib.load(models_dir / f"{task}_model.pkl")
        vec_path = models_dir / f"{task}_vectorizer.pkl"
        self.vectorizer = joblib.load(vec_path) if vec_path.exists() else None
        self.embedder = (SentenceEmbedder(self.meta["embedding_model"])
                         if self.meta.get("embedding_model") else None)
        self.label_names: list[str] = self.meta["label_names"]
        self.label_vi: list[str] = self.meta["label_vi"]

    # --- tiền xử lý & biến đổi --- #
    def preprocess(self, text: str) -> str:
        """Tiền xử lý đúng như lúc huấn luyện (cùng cột văn bản và cùng bộ tách từ)."""
        return preprocess_text(text, tokenizer=self.meta["tokenizer"],
                               segment=(self.meta["text_col"] == "text_seg"))

    def _transform(self, processed_texts: list[str]):
        if self.vectorizer is not None:
            return self.vectorizer.transform(processed_texts)
        return self.embedder.encode(processed_texts)

    # --- dự đoán --- #
    def predict_proba_processed(self, processed_texts: list[str]) -> np.ndarray:
        """Xác suất (n, K) cho văn bản ĐÃ tiền xử lý.

        Với mô hình không có predict_proba (Linear SVM) dùng softmax của decision_function;
        khi đó giá trị chỉ mang tính tương đối, không phải xác suất đã hiệu chỉnh.
        """
        X = self._transform(processed_texts)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        scores = np.asarray(self.model.decision_function(X))
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        return _softmax(scores)

    def predict_processed(self, processed_texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.predict(self._transform(processed_texts)))

    def predict_one(self, text: str) -> dict:
        processed = self.preprocess(text)
        probs = self.predict_proba_processed([processed])[0]
        idx = int(np.argmax(probs))
        return {
            "label_id": idx,
            "label": self.label_names[idx],
            "label_vi": self.label_vi[idx],
            "confidence": float(probs[idx]),
            "probabilities": {vi: float(p) for vi, p in zip(self.label_vi, probs)},
            "processed_text": processed,
        }


class FeedbackPredictor:
    """Dự đoán đồng thời cảm xúc (sentiment) và chủ đề (topic) của một câu phản hồi."""

    def __init__(self, models_dir: Path = C.MODELS_DIR):
        self.sentiment = TaskModel("sentiment", models_dir)
        self.topic = TaskModel("topic", models_dir)

    def analyze(self, text: str) -> dict:
        return {"sentiment": self.sentiment.predict_one(text), "topic": self.topic.predict_one(text)}
