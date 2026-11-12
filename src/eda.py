"""Khám phá dữ liệu (EDA): thống kê, độ mất cân bằng lớp, độ dài câu, từ khóa nổi bật.

Đầu ra:
    results/eda_summary.json                   số liệu thống kê (dùng cho báo cáo LaTeX)
    results/figures/class_distribution_{task}.png
    results/figures/length_distribution.png
    results/figures/sentiment_by_topic.png

Chạy:
    python -m src.eda            (cần dữ liệu thô trong data/raw/ hoặc kết nối Hugging Face)
"""
from __future__ import annotations

import json
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C
from .data_loader import load_splits

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# Một số từ chức năng phổ biến, loại khi thống kê từ khóa nổi bật.
STOPWORDS = set("""và là của có được cho các này với một những để trong khi thì mà cũng rất nhiều
    ở đã sẽ bị như nên cần còn hơn lại ra vào lên không em thầy cô các nhưng nếu vì do tại từ
    đến theo về trên dưới sau trước nữa luôn quá lắm ạ à ơi nhé nha""".split())


def word_count(text: str) -> int:
    return len(str(text).split())


def class_counts(df: pd.DataFrame, task: str) -> list[int]:
    n = len(C.LABEL_NAMES[task])
    return df[task].value_counts().reindex(range(n), fill_value=0).tolist()


def top_words(df: pd.DataFrame, task: str, k: int = 10) -> dict[str, list]:
    """k từ (đơn âm tiết/âm tiết) xuất hiện nhiều nhất trong mỗi lớp, sau khi bỏ từ chức năng."""
    out = {}
    for i, name in enumerate(C.LABEL_NAMES[task]):
        counter: Counter = Counter()
        for s in df.loc[df[task] == i, C.TEXT_COL]:
            counter.update(w for w in re.findall(r"\w+", str(s).lower())
                           if w not in STOPWORDS and not w.isdigit() and len(w) > 1)
        out[name] = [[w, c] for w, c in counter.most_common(k)]
    return out


def build_summary(splits: dict[str, pd.DataFrame]) -> dict:
    train, val, test = (splits[s] for s in C.SPLITS)
    allv = pd.concat(splits.values(), ignore_index=True)
    lengths = allv[C.TEXT_COL].map(word_count)
    train_set = set(train[C.TEXT_COL].str.lower())
    summary = {
        "sizes": {**{s: len(d) for s, d in splits.items()}, "total": len(allv)},
        "class_counts": {t: {s: class_counts(d, t) for s, d in splits.items()} for t in C.TASKS},
        "imbalance_ratio_train": {
            t: round(max(class_counts(train, t)) / max(1, min(class_counts(train, t))), 2) for t in C.TASKS},
        "length_words": {
            "mean": round(float(lengths.mean()), 2), "median": float(lengths.median()),
            "p95": float(lengths.quantile(0.95)), "min": int(lengths.min()), "max": int(lengths.max()),
        },
        "duplicates_in_train": int(train[C.TEXT_COL].str.lower().duplicated().sum()),
        "test_sentences_seen_in_train": int(test[C.TEXT_COL].str.lower().isin(train_set).sum()),
        "empty_sentences": int((allv[C.TEXT_COL].str.strip() == "").sum()),
        "top_words": {t: top_words(train, t) for t in C.TASKS},
    }
    return summary


# --------------------------------------------------------------------------- #
# Biểu đồ
# --------------------------------------------------------------------------- #
def plot_class_distribution(splits: dict[str, pd.DataFrame], task: str, path) -> None:
    labels = C.LABEL_VI[task]
    rows = [{"Tập": s, "Lớp": labels[i], "Tỉ lệ (%)": 100 * c / max(1, len(d))}
            for s, d in splits.items() for i, c in enumerate(class_counts(d, task))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    sns.barplot(data=pd.DataFrame(rows), x="Lớp", y="Tỉ lệ (%)", hue="Tập", ax=axes[0],
                palette=["#0072B2", "#E69F00", "#009E73"])
    axes[0].set_title(f"Tỉ lệ các lớp theo từng tập ({task})")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)
    train_counts = class_counts(splits["train"], task)
    bars = axes[1].bar(labels, train_counts, color=C.LABEL_COLORS[task])
    axes[1].bar_label(bars, fmt="{:,.0f}", padding=2)
    axes[1].set_title("Số mẫu mỗi lớp (tập train)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_ylim(0, max(train_counts) * 1.12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_length_distribution(df: pd.DataFrame, path) -> None:
    d = df.assign(n_words=df[C.TEXT_COL].map(word_count))
    cap = int(d["n_words"].quantile(0.99))
    d = d[d["n_words"] <= cap]
    d["Cảm xúc"] = d["sentiment"].map(dict(enumerate(C.LABEL_VI["sentiment"])))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.histplot(d, x="n_words", bins=min(40, cap), color="#0072B2", ax=axes[0])
    axes[0].set_title("Phân bố độ dài câu (số từ, tập train)")
    axes[0].set_xlabel("Số từ")
    axes[0].set_ylabel("Số câu")
    sns.boxplot(data=d, x="Cảm xúc", y="n_words", order=C.LABEL_VI["sentiment"],
                palette=C.LABEL_COLORS["sentiment"], hue="Cảm xúc", legend=False, ax=axes[1])
    axes[1].set_title("Độ dài câu theo cảm xúc")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Số từ")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_sentiment_by_topic(df: pd.DataFrame, path) -> None:
    ct = pd.crosstab(df["topic"], df["sentiment"]).reindex(
        index=range(len(C.LABEL_VI["topic"])), columns=range(len(C.LABEL_VI["sentiment"])), fill_value=0)
    ct.index = C.LABEL_VI["topic"]
    ct.columns = C.LABEL_VI["sentiment"]
    pct = ct.div(ct.sum(axis=1).replace(0, 1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sns.heatmap(pct, annot=ct.map(lambda v: f"{v:,}") if hasattr(ct, "map") else ct.applymap(lambda v: f"{v:,}"),
                fmt="", cmap="Blues", cbar_kws={"label": "% theo chủ đề"}, ax=ax, linewidths=0.5)
    ax.set_title("Cảm xúc theo chủ đề (tập train, số lượng)")
    ax.set_xlabel("Cảm xúc")
    ax.set_ylabel("Chủ đề")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_eda() -> dict:
    C.ensure_dirs()
    splits = load_splits()
    summary = build_summary(splits)
    (C.RESULTS_DIR / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    for task in C.TASKS:
        plot_class_distribution(splits, task, C.FIGURES_DIR / f"class_distribution_{task}.png")
    plot_length_distribution(splits["train"], C.FIGURES_DIR / "length_distribution.png")
    plot_sentiment_by_topic(splits["train"], C.FIGURES_DIR / "sentiment_by_topic.png")

    print("Kích thước:", summary["sizes"])
    for t in C.TASKS:
        print(f"Phân bố lớp {t} (train):", dict(zip(C.LABEL_NAMES[t], summary["class_counts"][t]["train"])),
              "| tỉ lệ lớp lớn nhất/nhỏ nhất =", summary["imbalance_ratio_train"][t])
    print("Độ dài câu (số từ):", summary["length_words"])
    print("Câu trùng trong train:", summary["duplicates_in_train"],
          "| câu test đã xuất hiện trong train:", summary["test_sentences_seen_in_train"])
    print(f"Đã lưu số liệu và hình vào {C.RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    run_eda()
