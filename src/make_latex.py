"""Chuyển kết quả thực nghiệm (results/*.csv|json) thành bảng LaTeX, macro số liệu và bảng Markdown.

Mọi con số trong báo cáo đều được sinh tự động từ kết quả thật, không gõ tay, nên không thể lệch với
những gì mô hình thực sự đạt được.

Đầu ra:
    report/tables/dataset_stats.tex, results_sentiment.tex, results_topic.tex, best_models.tex,
                  per_class_sentiment.tex, per_class_topic.tex, ablation.tex, macros.tex, findings.tex
    report/figures/*.png        (sao chép từ results/figures)
    results/summary.md          (bảng Markdown để dán vào README)

Chạy (sau khi đã chạy eda, train, evaluate):
    python -m src.make_latex
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from . import config as C

METRIC_COLS = [("test_accuracy", "Acc"), ("test_precision_macro", "P-macro"), ("test_recall_macro", "R-macro"),
               ("test_f1_macro", "F1-macro"), ("test_f1_weighted", "F1-w")]
FIGURES = ["class_distribution_sentiment.png", "class_distribution_topic.png", "length_distribution.png",
           "sentiment_by_topic.png", "model_comparison.png",
           "confusion_matrix_sentiment.png", "confusion_matrix_topic.png"]


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def tex_escape(s) -> str:
    for ch, rep in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = str(s).replace(ch, rep)
    return s


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", r"\,")


def write_table(name: str, content: str) -> None:
    C.REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORT_TABLES_DIR / name).write_text(content, encoding="utf-8")


def params_str(params_json: str) -> str:
    return ", ".join(f"{k}={v}" for k, v in json.loads(params_json).items())


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- #
# Các bảng LaTeX
# --------------------------------------------------------------------------- #
def dataset_table(eda: dict) -> str:
    rows = []
    for split in C.SPLITS:
        s = eda["class_counts"]["sentiment"][split]
        t = eda["class_counts"]["topic"][split]
        n = eda["sizes"][split]
        cells = [tex_escape(split), thousands(n)] + [f"{100 * c / sum(s):.1f}" for c in s] + \
                [f"{100 * c / sum(t):.1f}" for c in t]
        rows.append(" & ".join(cells) + r" \\")
    rows.append(r"\midrule")
    rows.append(" & ".join(["Tổng", thousands(eda["sizes"]["total"])] + [""] * 7) + r" \\")
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Số mẫu và phân bố lớp (\%) của UIT-VSFC. CTĐT: chương trình đào tạo; CSVC: cơ sở vật chất.}",
        r"\label{tab:dataset}", r"\fitbox{\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r" & & \multicolumn{3}{c}{Cảm xúc} & \multicolumn{4}{c}{Chủ đề} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-9}",
        r"Tập & Số mẫu & Tiêu cực & Trung tính & Tích cực & Giảng viên & CTĐT & CSVC & Khác \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


def results_table(metrics: pd.DataFrame, task: str) -> str:
    sub = metrics[metrics["task"] == task]
    best = {col: sub[col].max() for col, _ in METRIC_COLS}
    lines = []
    feats = [f for f in C.FEATURE_KINDS if f in set(sub["feature"])]
    for fi, feat in enumerate(feats):
        block = sub[sub["feature"] == feat]
        models = [m for m in C.MODEL_NAMES if m in set(block["model"])]
        for mi, model in enumerate(models):
            r = block[block["model"] == model].iloc[0]
            cells = [C.FEATURE_DISPLAY[feat] if mi == 0 else "", C.MODEL_DISPLAY[model]]
            for col, _ in METRIC_COLS:
                v = pct(r[col])
                cells.append(rf"\textbf{{{v}}}" if abs(r[col] - best[col]) < 1e-12 else v)
            lines.append(" & ".join(cells) + r" \\")
        if fi < len(feats) - 1:
            lines.append(r"\midrule")
    header = " & ".join(["Đặc trưng", "Mô hình"] + [h for _, h in METRIC_COLS]) + r" \\"
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering\small",
        rf"\caption{{Kết quả trên tập kiểm thử cho bài toán {C.TASK_VI[task].lower()} (\%). "
        r"In đậm: giá trị cao nhất mỗi cột. P/R/F1-macro: trung bình không trọng số các lớp; F1-w: trung bình có trọng số.}",
        rf"\label{{tab:results-{task}}}", r"\fitbox{\begin{tabular}{llrrrrr}", r"\toprule", header, r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


def best_models_table(metrics: pd.DataFrame) -> str:
    lines = []
    for task in C.TASKS:
        sub = metrics[metrics["task"] == task]
        if sub.empty:
            continue
        r = sub.loc[sub["val_f1_macro"].idxmax()]     # chọn theo validation, không dùng test
        lines.append(" & ".join([
            C.TASK_VI[task], C.FEATURE_DISPLAY[r["feature"]], C.MODEL_DISPLAY[r["model"]],
            tex_escape(params_str(r["params"])), pct(r["val_f1_macro"]), pct(r["test_accuracy"]),
            pct(r["test_f1_macro"])]) + r" \\")
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering\small",
        r"\caption{Mô hình tốt nhất cho mỗi bài toán (chọn theo F1-macro trên tập kiểm định; số liệu test (\%) chỉ để báo cáo).}",
        r"\label{tab:best}", r"\fitbox{\begin{tabular}{lllcrrr}", r"\toprule",
        r"Bài toán & Đặc trưng & Mô hình & Siêu tham số & F1-macro (val) & Acc (test) & F1-macro (test) \\",
        r"\midrule", *lines, r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


def per_class_table(task: str, report: dict, meta_text: str) -> str:
    labels = C.LABEL_VI[task]
    lines = []
    for lab in labels:
        r = report[lab]
        lines.append(" & ".join([tex_escape(lab), pct(r["precision"]), pct(r["recall"]), pct(r["f1-score"]),
                                 thousands(int(r["support"]))]) + r" \\")
    lines.append(r"\midrule")
    for key, name in (("macro avg", "Macro avg"), ("weighted avg", "Weighted avg")):
        r = report[key]
        lines.append(" & ".join([name, pct(r["precision"]), pct(r["recall"]), pct(r["f1-score"]),
                                 thousands(int(r["support"]))]) + r" \\")
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering\small",
        rf"\caption{{Precision, Recall, F1 (\%) theo từng lớp của mô hình tốt nhất cho bài toán {C.TASK_VI[task].lower()} ({meta_text}), tập kiểm thử.}}",
        rf"\label{{tab:perclass-{task}}}", r"\fitbox{\begin{tabular}{lrrrr}", r"\toprule",
        r"Lớp & Precision & Recall & F1 & Số mẫu \\", r"\midrule", *lines, r"\bottomrule",
        r"\end{tabular}}", r"\end{table}", ""])


def ablation_table(abl: pd.DataFrame) -> str:
    piv = abl.pivot_table(index=["study", "variant"], columns=["task", "model"], values="test_f1_macro", sort=False)
    cols = [(t, m) for t in C.TASKS for m in ("LogisticRegression", "LinearSVM") if (t, m) in piv.columns]
    head1 = " & ".join(["", ""] + [rf"\multicolumn{{2}}{{c}}{{{C.TASK_VI[t]}}}" for t in C.TASKS
                                   if any(c[0] == t for c in cols)]) + r" \\"
    head2 = " & ".join(["Thí nghiệm", "Biến thể"] + [C.MODEL_DISPLAY[m] for _, m in cols]) + r" \\"
    lines, last_study = [], None
    for (study, variant), row in piv.iterrows():
        if last_study is not None and study != last_study:
            lines.append(r"\midrule")
        label = tex_escape(study) if study != last_study else ""
        lines.append(" & ".join([label, tex_escape(variant)] + [pct(row[c]) for c in cols]) + r" \\")
        last_study = study
    spec = "ll" + "rr" * (len(cols) // 2)
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering\small",
        r"\caption{Thí nghiệm đối chứng với đặc trưng TF-IDF: F1-macro (\%) trên tập kiểm thử.}",
        r"\label{tab:ablation}", rf"\fitbox{{\begin{{tabular}}{{{spec}}}", r"\toprule", head1,
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}", head2, r"\midrule", *lines, r"\bottomrule",
        r"\end{tabular}}", r"\end{table}", ""])


# --------------------------------------------------------------------------- #
# Macro số liệu + nhận xét tự động
# --------------------------------------------------------------------------- #
def best_row(metrics: pd.DataFrame, task: str):
    sub = metrics[metrics["task"] == task]
    return sub.loc[sub["val_f1_macro"].idxmax()] if not sub.empty else None


def macros_tex(metrics: pd.DataFrame, eda: dict | None) -> str:
    out = ["% Tệp sinh tự động bởi src/make_latex.py - KHÔNG chỉnh tay."]
    if eda:
        for key, split in (("NTrain", "train"), ("NVal", "validation"), ("NTest", "test"), ("NTotal", "total")):
            out.append(rf"\newcommand{{\{key}}}{{{thousands(eda['sizes'][split])}}}")
        out.append(rf"\newcommand{{\MeanLen}}{{{eda['length_words']['mean']:.1f}}}")
        out.append(rf"\newcommand{{\MaxLen}}{{{int(eda['length_words']['max'])}}}")
        out.append(rf"\newcommand{{\ImbSent}}{{{eda['imbalance_ratio_train']['sentiment']:.1f}}}")
        out.append(rf"\newcommand{{\ImbTopic}}{{{eda['imbalance_ratio_train']['topic']:.1f}}}")
        out.append(rf"\newcommand{{\DupTrain}}{{{thousands(eda['duplicates_in_train'])}}}")
        out.append(rf"\newcommand{{\OverlapTest}}{{{thousands(eda['test_sentences_seen_in_train'])}}}")
    for task, tag in (("sentiment", "Sent"), ("topic", "Topic")):
        r = best_row(metrics, task)
        if r is None:
            continue
        out += [
            rf"\newcommand{{\Best{tag}Feature}}{{{C.FEATURE_DISPLAY[r['feature']]}}}",
            rf"\newcommand{{\Best{tag}Model}}{{{C.MODEL_DISPLAY[r['model']]}}}",
            rf"\newcommand{{\Best{tag}Acc}}{{{pct(r['test_accuracy'])}}}",
            rf"\newcommand{{\Best{tag}FOne}}{{{pct(r['test_f1_macro'])}}}",
            rf"\newcommand{{\Best{tag}FW}}{{{pct(r['test_f1_weighted'])}}}",
        ]
    return "\n".join(out) + "\n"


def findings_tex(metrics: pd.DataFrame) -> str:
    """Các câu nhận xét thuần mô tả, sinh từ số liệu (phần diễn giải sâu hơn do người viết bổ sung)."""
    paras = []
    for task in C.TASKS:
        sub = metrics[metrics["task"] == task]
        if sub.empty:
            continue
        r = best_row(metrics, task)
        lo, hi = sub.loc[sub["test_f1_macro"].idxmin()], sub.loc[sub["test_f1_macro"].idxmax()]
        name = lambda x: f"{C.FEATURE_DISPLAY[x['feature']]} + {C.MODEL_DISPLAY[x['model']]}"
        by_feat = sub.groupby("feature")["test_f1_macro"].mean()
        by_model = sub.groupby("model")["test_f1_macro"].mean()
        feat_txt = ", ".join(f"{C.FEATURE_DISPLAY[f]} {pct(by_feat[f])}\\%" for f in C.FEATURE_KINDS if f in by_feat)
        model_txt = ", ".join(f"{C.MODEL_DISPLAY[m]} {pct(by_model[m])}\\%" for m in C.MODEL_NAMES if m in by_model)
        sentences = [
            rf"\textbf{{{C.TASK_VI[task]}.}} Cấu hình được chọn theo F1-macro trên tập kiểm định là "
            rf"{name(r)}; trên tập kiểm thử cấu hình này đạt Accuracy {pct(r['test_accuracy'])}\% và "
            rf"F1-macro {pct(r['test_f1_macro'])}\%.",
            rf"Trên {len(sub)} cấu hình, F1-macro kiểm thử dao động từ {pct(lo['test_f1_macro'])}\% ({name(lo)}) "
            rf"đến {pct(hi['test_f1_macro'])}\% ({name(hi)}).",
            rf"F1-macro kiểm thử trung bình theo loại đặc trưng: {feat_txt}; theo thuật toán: {model_txt}.",
        ]
        report = _read_json(C.RESULTS_DIR / f"classification_report_{task}.json")
        if report:
            per = {lab: report[lab]["f1-score"] for lab in C.LABEL_VI[task]}
            worst = min(per, key=per.get)
            sentences.append(rf"Với mô hình được chọn, lớp có F1 thấp nhất là \textit{{{tex_escape(worst)}}} "
                             rf"(F1 = {pct(per[worst])}\%, {int(report[worst]['support'])} mẫu kiểm thử).")
        paras.append(" ".join(sentences))
    return "\n\n".join(paras) + "\n"


# --------------------------------------------------------------------------- #
# Markdown cho README
# --------------------------------------------------------------------------- #
def summary_md(metrics: pd.DataFrame) -> str:
    out = ["# Tóm tắt kết quả (tập test)", "", "Sinh tự động bởi `python -m src.make_latex`.", ""]
    for task in C.TASKS:
        sub = metrics[metrics["task"] == task]
        if sub.empty:
            continue
        r = best_row(metrics, task)
        out += [f"## {C.TASK_VI[task]} ({task})", "",
                f"Mô hình được chọn (theo F1-macro trên validation): **{C.FEATURE_DISPLAY[r['feature']]} + "
                f"{C.MODEL_DISPLAY[r['model']]}** - Accuracy {pct(r['test_accuracy'])}%, F1-macro {pct(r['test_f1_macro'])}%.", "",
                "| Đặc trưng | Mô hình | Acc | P-macro | R-macro | F1-macro | F1-weighted |",
                "|---|---|---:|---:|---:|---:|---:|"]
        for feat in C.FEATURE_KINDS:
            for model in C.MODEL_NAMES:
                m = sub[(sub["feature"] == feat) & (sub["model"] == model)]
                if m.empty:
                    continue
                m = m.iloc[0]
                out.append(f"| {C.FEATURE_DISPLAY[feat]} | {C.MODEL_DISPLAY[model]} | "
                           + " | ".join(pct(m[c]) for c, _ in METRIC_COLS) + " |")
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main() -> None:
    C.ensure_dirs()
    metrics_path = C.RESULTS_DIR / "metrics_all.csv"
    if not metrics_path.exists():
        raise SystemExit("Chưa có results/metrics_all.csv. Hãy chạy:  python -m src.train")
    metrics = pd.read_csv(metrics_path)
    eda = _read_json(C.RESULTS_DIR / "eda_summary.json")

    if eda:
        write_table("dataset_stats.tex", dataset_table(eda))
    for task in C.TASKS:
        if (metrics["task"] == task).any():
            write_table(f"results_{task}.tex", results_table(metrics, task))
        report = _read_json(C.RESULTS_DIR / f"classification_report_{task}.json")
        r = best_row(metrics, task)
        if report and r is not None:
            write_table(f"per_class_{task}.tex",
                        per_class_table(task, report, f"{C.FEATURE_DISPLAY[r['feature']]} + {C.MODEL_DISPLAY[r['model']]}"))
    write_table("best_models.tex", best_models_table(metrics))
    abl_path = C.RESULTS_DIR / "ablation.csv"
    if abl_path.exists():
        write_table("ablation.tex", ablation_table(pd.read_csv(abl_path)))
    write_table("macros.tex", macros_tex(metrics, eda))
    write_table("findings.tex", findings_tex(metrics))
    (C.RESULTS_DIR / "summary.md").write_text(summary_md(metrics), encoding="utf-8")

    copied = 0
    for name in FIGURES:
        src = C.FIGURES_DIR / name
        if src.exists():
            shutil.copy2(src, C.REPORT_FIGURES_DIR / name)
            copied += 1
    print(f"Đã sinh bảng LaTeX trong {C.REPORT_TABLES_DIR} và chép {copied} hình vào {C.REPORT_FIGURES_DIR}")
    print(f"Bảng Markdown: {C.RESULTS_DIR / 'summary.md'}")
    print("Biên dịch báo cáo: xelatex (hoặc latexmk -xelatex) report/main.tex")


if __name__ == "__main__":
    main()
