"""Đánh giá mô hình tốt nhất trên tập test và vẽ các hình phục vụ báo cáo.

Đầu ra (trong ``results/``):
    classification_report_{task}.txt / .json   Precision/Recall/F1 từng lớp
    errors_{task}.csv                          các câu dự đoán sai (phục vụ phân tích lỗi)
    figures/confusion_matrix_{task}.png        ma trận nhầm lẫn (số lượng + chuẩn hóa theo hàng)
    figures/model_comparison.png               so sánh F1-macro của toàn bộ cấu hình

Chạy:
    python -m src.evaluate
"""
from __future__ import annotations

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from . import config as C
from .predict import TaskModel
from .preprocess import load_processed

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def _wrap(labels: list[str], width: int = 11) -> list[str]:
    return [textwrap.fill(label, width) for label in labels]


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, path) -> None:
    """Hai ma trận cạnh nhau: số lượng và tỉ lệ chuẩn hóa theo hàng (recall của từng lớp)."""
    cm = np.asarray(cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    wrapped = _wrap(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, data, fmt, subtitle in ((axes[0], cm, "d", "Số lượng"), (axes[1], norm, ".1%", "Chuẩn hóa theo hàng")):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", cbar=False, ax=ax, linewidths=0.5,
                    xticklabels=wrapped, yticklabels=wrapped, annot_kws={"fontsize": 11})
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế", labelpad=8)
        ax.set_title(subtitle)
        plt.setp(ax.get_yticklabels(), rotation=0, va="center")   # nhãn ngang, không đè lên tiêu đề trục
        plt.setp(ax.get_xticklabels(), rotation=0)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame, path) -> None:
    """Biểu đồ cột F1-macro (tập test) cho mọi tổ hợp đặc trưng x thuật toán."""
    feat_palette = dict(zip(C.FEATURE_KINDS, ["#9AA5B1", "#0072B2", "#E69F00"]))
    tasks = [t for t in C.TASKS if t in set(metrics["task"])]
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.8), sharey=True, squeeze=False)
    handles = legend_labels = None
    for ax, task in zip(axes[0], tasks):
        sub = metrics[metrics["task"] == task].copy()
        sub["model_name"] = sub["model"].map(C.MODEL_DISPLAY)
        sub["feature_name"] = sub["feature"].map(C.FEATURE_DISPLAY)
        order = [C.MODEL_DISPLAY[m] for m in C.MODEL_NAMES if m in set(sub["model"])]
        hue_order = [C.FEATURE_DISPLAY[f] for f in C.FEATURE_KINDS if f in set(sub["feature"])]
        sns.barplot(data=sub, x="model_name", y="test_f1_macro", hue="feature_name", order=order,
                    hue_order=hue_order, palette={C.FEATURE_DISPLAY[k]: v for k, v in feat_palette.items()}, ax=ax)
        ax.set_title(f"Bài toán {C.TASK_VI[task].lower()} ({task})")
        ax.set_xlabel("")
        ax.set_ylabel("F1-macro (tập test)" if ax is axes[0][0] else "")
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0.0, 1.08)                       # trục y bắt đầu từ 0 để không phóng đại chênh lệch
        for container in ax.containers:              # ghi giá trị lên đầu mỗi cột
            ax.bar_label(container, fmt="%.2f", fontsize=7, rotation=90, padding=2)
        if handles is None:
            handles, legend_labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
    # Một chú giải chung đặt phía dưới, không che các cột.
    fig.legend(handles, legend_labels, title="Đặc trưng", loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate_best_models() -> None:
    C.ensure_dirs()
    data = load_processed()
    test = data["test"]
    metrics_path = C.RESULTS_DIR / "metrics_all.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None

    for task in C.TASKS:
        tm = TaskModel(task)
        texts = test[tm.meta["text_col"]].tolist()
        y_true = test[task].to_numpy()
        y_pred = tm.predict_processed(texts)
        probs = tm.predict_proba_processed(texts)
        labels_vi = tm.label_vi

        acc = accuracy_score(y_true, y_pred)
        # Kiểm tra nhất quán với số liệu lúc huấn luyện.
        if abs(acc - tm.meta["test_accuracy"]) > 1e-6:
            print(f"  [!] {task}: accuracy khi nạp lại ({acc:.4f}) khác lúc huấn luyện ({tm.meta['test_accuracy']:.4f})")

        report_txt = classification_report(y_true, y_pred, labels=list(range(len(labels_vi))),
                                          target_names=labels_vi, digits=4, zero_division=0)
        report_dict = classification_report(y_true, y_pred, labels=list(range(len(labels_vi))),
                                            target_names=labels_vi, output_dict=True, zero_division=0)
        header = f"Bài toán: {task} | {C.FEATURE_DISPLAY[tm.meta['feature']]} + {C.MODEL_DISPLAY[tm.meta['model']]}\n"
        (C.RESULTS_DIR / f"classification_report_{task}.txt").write_text(header + report_txt, encoding="utf-8")
        (C.RESULTS_DIR / f"classification_report_{task}.json").write_text(
            json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_vi))))
        title = f"Ma trận nhầm lẫn - {C.TASK_VI[task]} ({C.FEATURE_DISPLAY[tm.meta['feature']]} + {C.MODEL_DISPLAY[tm.meta['model']]})"
        plot_confusion_matrix(cm, labels_vi, title, C.FIGURES_DIR / f"confusion_matrix_{task}.png")

        wrong = test.loc[y_true != y_pred, [C.TEXT_COL]].copy()
        wrong["thuc_te"] = [labels_vi[i] for i in y_true[y_true != y_pred]]
        wrong["du_doan"] = [labels_vi[i] for i in y_pred[y_true != y_pred]]
        wrong["do_tin_cay"] = probs[y_true != y_pred].max(axis=1).round(3)
        wrong.sort_values("do_tin_cay", ascending=False).to_csv(
            C.RESULTS_DIR / f"errors_{task}.csv", index=False, encoding="utf-8")

        print(f"\n{header}{report_txt}  -> {len(wrong):,} câu sai / {len(test):,}")

    if metrics is not None:
        plot_model_comparison(metrics, C.FIGURES_DIR / "model_comparison.png")
        print(f"\nĐã lưu hình trong {C.FIGURES_DIR}")


if __name__ == "__main__":
    evaluate_best_models()
