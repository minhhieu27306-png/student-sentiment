"""Huấn luyện, tinh chỉnh siêu tham số và so sánh các mô hình.

Lưới thực nghiệm: 3 loại đặc trưng (BoW, TF-IDF, Embedding) x 5 thuật toán
(Naive Bayes, Logistic Regression, Linear SVM, Random Forest, KNN) x 2 bài toán (sentiment, topic).

Giao thức đánh giá (tránh rò rỉ dữ liệu):
    * Vectorizer chỉ fit trên tập train.
    * Siêu tham số của từng mô hình được chọn theo F1-macro trên tập VALIDATION.
    * Mô hình tốt nhất của mỗi bài toán cũng được chọn theo F1-macro trên tập VALIDATION.
    * Tập TEST chỉ dùng một lần để báo cáo kết quả cuối cùng.

Đầu ra:
    results/metrics_all.csv         toàn bộ chỉ số (val + test) của mọi cấu hình
    results/confusion_matrices.json ma trận nhầm lẫn (tập test) của mọi cấu hình
    results/ablation.csv            (tùy chọn --ablation) ảnh hưởng của tách từ và class_weight
    models/{task}_model.pkl, {task}_vectorizer.pkl, {task}_meta.json   mô hình tốt nhất mỗi bài toán

Chạy:
    python -m src.train                     # đầy đủ (cần sentence-transformers cho đặc trưng embedding)
    python -m src.train --skip-embedding    # nhanh: chỉ BoW + TF-IDF
    python -m src.train --ablation          # thêm thí nghiệm đối chứng
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
import warnings
from dataclasses import dataclass
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from . import config as C
from .features import FeatureSet, build_features
from .preprocess import load_processed, saved_tokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

# Cột văn bản dùng cho từng loại đặc trưng:
#   BoW/TF-IDF dùng văn bản đã tách từ; Embedding đa ngôn ngữ dùng văn bản chưa tách từ.
FEATURE_TEXT_COL = {"bow": "text_seg", "tfidf": "text_seg", "embedding": "text_clean"}


# --------------------------------------------------------------------------- #
# Danh mục mô hình + lưới siêu tham số (nhỏ, để chạy được trên máy cá nhân)
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    make: Callable[[dict, bool, str | None], object]   # (params, dense, class_weight) -> estimator
    grid: Callable[[bool], dict[str, list]]            # dense -> lưới siêu tham số


MODEL_ZOO: dict[str, ModelSpec] = {
    # Multinomial NB cho đặc trưng thưa (đếm/TF-IDF); Gaussian NB cho embedding (có giá trị âm).
    "NaiveBayes": ModelSpec(
        make=lambda p, dense, cw: GaussianNB(**p) if dense else MultinomialNB(**p),
        grid=lambda dense: {"var_smoothing": [1e-9, 1e-7, 1e-5]} if dense else {"alpha": [0.1, 0.3, 1.0]},
    ),
    "LogisticRegression": ModelSpec(
        make=lambda p, dense, cw: LogisticRegression(max_iter=3000, class_weight=cw,
                                                     random_state=C.RANDOM_STATE, **p),
        grid=lambda dense: {"C": [0.5, 1.0, 5.0, 20.0]},
    ),
    "LinearSVM": ModelSpec(
        make=lambda p, dense, cw: LinearSVC(max_iter=20000, class_weight=cw,
                                            random_state=C.RANDOM_STATE, **p),
        grid=lambda dense: {"C": [0.05, 0.1, 0.5, 1.0]},
    ),
    "RandomForest": ModelSpec(
        make=lambda p, dense, cw: RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=C.RANDOM_STATE,
            class_weight="balanced_subsample" if cw else None, **p),
        grid=lambda dense: {"min_samples_leaf": [1, 2]},
    ),
    "KNN": ModelSpec(
        make=lambda p, dense, cw: KNeighborsClassifier(weights="distance", metric="cosine",
                                                       algorithm="brute", n_jobs=-1, **p),
        grid=lambda dense: {"n_neighbors": [5, 11, 21]},
    ),
}


def _expand(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


# --------------------------------------------------------------------------- #
# Chỉ số đánh giá
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, n_classes: int, prefix: str) -> dict[str, float]:
    """Accuracy, Precision/Recall/F1 (macro & weighted) và F1 theo từng lớp."""
    labels = list(range(n_classes))
    kw = dict(labels=labels, zero_division=0)
    out = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision_macro": precision_score(y_true, y_pred, average="macro", **kw),
        f"{prefix}_recall_macro": recall_score(y_true, y_pred, average="macro", **kw),
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average="macro", **kw),
        f"{prefix}_precision_weighted": precision_score(y_true, y_pred, average="weighted", **kw),
        f"{prefix}_recall_weighted": recall_score(y_true, y_pred, average="weighted", **kw),
        f"{prefix}_f1_weighted": f1_score(y_true, y_pred, average="weighted", **kw),
    }
    for i, f1 in enumerate(f1_score(y_true, y_pred, average=None, **kw)):
        out[f"{prefix}_f1_c{i}"] = f1
    return out


@dataclass
class RunResult:
    row: dict
    estimator: object
    cm: list


def tune_and_evaluate(model_name: str, fs: FeatureSet, y_tr, y_va, y_te, task: str,
                      class_weight: str | None) -> RunResult:
    """Chọn siêu tham số tốt nhất theo F1-macro trên validation, rồi đánh giá trên test."""
    spec = MODEL_ZOO[model_name]
    n_classes = len(C.LABEL_NAMES[task])
    best = None   # (val_f1, params, estimator, fit_seconds)
    for params in _expand(spec.grid(fs.dense)):
        clf = spec.make(params, fs.dense, class_weight)
        t0 = time.perf_counter()
        clf.fit(fs.X_train, y_tr)
        fit_s = time.perf_counter() - t0
        val_f1 = f1_score(y_va, clf.predict(fs.X_val), average="macro",
                          labels=list(range(n_classes)), zero_division=0)
        if best is None or val_f1 > best[0]:
            best = (val_f1, params, clf, fit_s)

    _, params, clf, fit_s = best
    row = {"task": task, "feature": fs.kind, "model": model_name,
           "params": json.dumps(params), "class_weight": class_weight or "none",
           "text_col": fs.text_col, "fit_seconds": round(fit_s, 2)}
    row.update(compute_metrics(y_va, clf.predict(fs.X_val), n_classes, "val"))
    y_pred = clf.predict(fs.X_test)
    row.update(compute_metrics(y_te, y_pred, n_classes, "test"))
    cm = confusion_matrix(y_te, y_pred, labels=list(range(n_classes))).tolist()
    return RunResult(row, clf, cm)


# --------------------------------------------------------------------------- #
# Lưu mô hình tốt nhất
# --------------------------------------------------------------------------- #
def save_best(task: str, result: RunResult, fs: FeatureSet, tokenizer: str) -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.estimator, C.MODELS_DIR / f"{task}_model.pkl", compress=3)
    vec_path = C.MODELS_DIR / f"{task}_vectorizer.pkl"
    if fs.vectorizer is not None:
        joblib.dump(fs.vectorizer, vec_path, compress=3)
    elif vec_path.exists():
        vec_path.unlink()    # tránh để sót vectorizer cũ khi mô hình tốt nhất là embedding
    row = result.row
    meta = {
        "task": task, "feature": fs.kind, "model": row["model"], "params": json.loads(row["params"]),
        "class_weight": row["class_weight"], "text_col": fs.text_col, "tokenizer": tokenizer,
        "embedding_model": fs.embedding_model, "label_names": C.LABEL_NAMES[task],
        "label_vi": C.LABEL_VI[task],
        "val_f1_macro": row["val_f1_macro"], "test_accuracy": row["test_accuracy"],
        "test_f1_macro": row["test_f1_macro"], "test_f1_weighted": row["test_f1_weighted"],
    }
    (C.MODELS_DIR / f"{task}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")


# --------------------------------------------------------------------------- #
# Thí nghiệm chính
# --------------------------------------------------------------------------- #
def run_experiments(features: list[str], models: list[str], tasks: list[str], class_weight: str | None,
                    embedding_model: str) -> pd.DataFrame:
    data = load_processed()
    tokenizer = saved_tokenizer()
    labels = {t: {s: data[s][t].to_numpy() for s in C.SPLITS} for t in tasks}

    rows: list[dict] = []
    cms: dict[str, list] = {}
    best: dict[str, tuple[float, RunResult, FeatureSet]] = {}
    total = len(features) * len(models) * len(tasks)
    done = 0

    for kind in features:
        print(f"\n=== Đặc trưng: {C.FEATURE_DISPLAY[kind]} ({FEATURE_TEXT_COL[kind]}) ===")
        t0 = time.perf_counter()
        fs = build_features(kind, data, FEATURE_TEXT_COL[kind], embedding_model)
        shape = fs.X_train.shape
        print(f"  Ma trận train: {shape[0]:,} x {shape[1]:,}  ({time.perf_counter() - t0:.1f}s)")
        for task in tasks:
            y = labels[task]
            for model_name in models:
                done += 1
                res = tune_and_evaluate(model_name, fs, y["train"], y["validation"], y["test"], task, class_weight)
                r = res.row
                print(f"  [{done:>2}/{total}] {task:<9} {model_name:<19} "
                      f"val F1={r['val_f1_macro']:.4f} | test acc={r['test_accuracy']:.4f} "
                      f"F1-macro={r['test_f1_macro']:.4f}  {r['params']}  ({r['fit_seconds']}s)")
                rows.append(r)
                cms[f"{task}|{kind}|{model_name}"] = res.cm
                # Chọn mô hình tốt nhất theo F1-macro trên VALIDATION (không dùng test).
                if task not in best or r["val_f1_macro"] > best[task][0]:
                    best[task] = (r["val_f1_macro"], res, fs)
        # Giải phóng bộ nhớ của ma trận đặc trưng nếu không phải mô hình tốt nhất đang giữ tham chiếu.
        del fs

    df = pd.DataFrame(rows)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTS_DIR / "metrics_all.csv", index=False, encoding="utf-8")
    (C.RESULTS_DIR / "confusion_matrices.json").write_text(json.dumps(cms), encoding="utf-8")

    print("\n=== Mô hình tốt nhất (chọn theo F1-macro trên validation) ===")
    for task, (val_f1, res, fs_best) in best.items():
        save_best(task, res, fs_best, tokenizer)
        r = res.row
        print(f"  {task:<9}: {C.FEATURE_DISPLAY[r['feature']]} + {C.MODEL_DISPLAY[r['model']]}  "
              f"val F1={val_f1:.4f} | test acc={r['test_accuracy']:.4f} F1-macro={r['test_f1_macro']:.4f}")
    print(f"\nĐã lưu: {C.RESULTS_DIR / 'metrics_all.csv'} và mô hình trong {C.MODELS_DIR}")
    return df


# --------------------------------------------------------------------------- #
# Thí nghiệm đối chứng (ablation)
# --------------------------------------------------------------------------- #
def run_ablation(tasks: list[str]) -> pd.DataFrame:
    """Đo ảnh hưởng của (1) tách từ và (2) trọng số lớp, trên TF-IDF với LR và Linear SVM."""
    data = load_processed()
    labels = {t: {s: data[s][t].to_numpy() for s in C.SPLITS} for t in tasks}
    studies = [
        # (tên nghiên cứu, nhãn biến thể, cột văn bản, class_weight)
        ("Tách từ", "Có tách từ (pyvi)", "text_seg", "balanced"),
        ("Tách từ", "Không tách từ", "text_clean", "balanced"),
        ("Trọng số lớp", "class_weight = balanced", "text_seg", "balanced"),
        ("Trọng số lớp", "Không trọng số", "text_seg", None),
    ]
    rows = []
    print("\n=== Thí nghiệm đối chứng (TF-IDF) ===")
    cache: dict[str, FeatureSet] = {}
    for study, variant, text_col, cw in studies:
        if text_col not in cache:
            cache[text_col] = build_features("tfidf", data, text_col)
        fs = cache[text_col]
        for task in tasks:
            y = labels[task]
            for model_name in ("LogisticRegression", "LinearSVM"):
                r = tune_and_evaluate(model_name, fs, y["train"], y["validation"], y["test"], task, cw).row
                rows.append({"study": study, "variant": variant, "task": task, "model": model_name,
                             "val_f1_macro": r["val_f1_macro"], "test_accuracy": r["test_accuracy"],
                             "test_f1_macro": r["test_f1_macro"], "test_f1_weighted": r["test_f1_weighted"]})
                print(f"  {study:<13} | {variant:<24} | {task:<9} {model_name:<19} "
                      f"acc={r['test_accuracy']:.4f} F1-macro={r['test_f1_macro']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "ablation.csv", index=False, encoding="utf-8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện & so sánh mô hình phân loại phản hồi sinh viên.")
    parser.add_argument("--features", nargs="+", choices=C.FEATURE_KINDS, default=list(C.FEATURE_KINDS))
    parser.add_argument("--skip-embedding", action="store_true", help="Bỏ qua đặc trưng Sentence Embedding.")
    parser.add_argument("--models", nargs="+", choices=C.MODEL_NAMES, default=list(C.MODEL_NAMES))
    parser.add_argument("--tasks", nargs="+", choices=C.TASKS, default=list(C.TASKS))
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced",
                        help="Trọng số lớp cho LR/SVM/RF (mặc định: balanced, phù hợp dữ liệu mất cân bằng).")
    parser.add_argument("--embedding-model", default=C.EMBEDDING_MODEL)
    parser.add_argument("--ablation", action="store_true", help="Chạy thêm thí nghiệm đối chứng.")
    args = parser.parse_args()

    features = [f for f in args.features if not (args.skip_embedding and f == "embedding")]
    cw = None if args.class_weight == "none" else "balanced"
    np.random.seed(C.RANDOM_STATE)

    run_experiments(features, args.models, args.tasks, cw, args.embedding_model)
    if args.ablation:
        run_ablation(args.tasks)


if __name__ == "__main__":
    main()
