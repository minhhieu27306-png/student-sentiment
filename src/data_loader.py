"""Tải và nạp bộ dữ liệu UIT-VSFC (uitnlp/vietnamese_students_feedback).

Thứ tự ưu tiên khi nạp dữ liệu:
    1. File cục bộ trong ``data/raw/`` (train/validation/test dạng .csv/.tsv/.parquet/.json,
       hoặc định dạng gốc của UIT: thư mục train|dev|test chứa sents.txt, sentiments.txt, topics.txt).
    2. Tải từ Hugging Face Hub (``datasets.load_dataset``), rồi lưu lại thành CSV trong ``data/raw/``.

Nhờ vậy dự án vẫn chạy được khi không có Internet, miễn là bạn đặt sẵn file dữ liệu vào ``data/raw/``.

Chạy độc lập:
    python -m src.data_loader                 # nạp (hoặc tải) và in thống kê
    python -m src.data_loader --force-download
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config as C

# Tên file/thư mục được chấp nhận cho từng split.
_SPLIT_ALIASES = {
    "train": ("train",),
    "validation": ("validation", "valid", "val", "dev"),
    "test": ("test",),
}
# Tên cột văn bản được chấp nhận (dataset gốc dùng "sentence").
_TEXT_ALIASES = ("sentence", "text", "content", "feedback", "comment")
_SUFFIXES = {".csv", ".tsv", ".parquet", ".json", ".jsonl"}


# --------------------------------------------------------------------------- #
# Chuẩn hóa bảng dữ liệu
# --------------------------------------------------------------------------- #
def _to_label_id(series: pd.Series, task: str) -> pd.Series:
    """Chuyển cột nhãn (số hoặc chữ) về chỉ số nguyên 0..K-1."""
    names = C.LABEL_NAMES[task]
    if pd.api.types.is_numeric_dtype(series):
        out = series
    else:
        norm = (series.astype(str).str.strip().str.lower()
                .str.replace(r"[\s\-]+", "_", regex=True))
        if norm.str.fullmatch(r"\d+").all():
            out = norm.astype(int)
        else:
            out = norm.map({n: i for i, n in enumerate(names)})
    if out.isna().any():
        raise ValueError(f"Cột nhãn '{task}' có giá trị không nhận diện được: "
                         f"{sorted(series[out.isna()].astype(str).unique())[:5]}")
    out = out.astype(int)
    if not out.between(0, len(names) - 1).all():
        raise ValueError(f"Cột nhãn '{task}' có chỉ số nằm ngoài khoảng 0..{len(names) - 1}.")
    return out


def _standardize(df: pd.DataFrame, where: str) -> pd.DataFrame:
    """Đưa bảng về đúng 3 cột: sentence, sentiment, topic."""
    cols = {c.lower(): c for c in df.columns}
    text_col = next((cols[a] for a in _TEXT_ALIASES if a in cols), None)
    missing = [t for t in C.TASKS if t not in cols]
    if text_col is None or missing:
        raise ValueError(f"{where}: thiếu cột (cần 'sentence', 'sentiment', 'topic'); "
                         f"hiện có {list(df.columns)}")
    out = pd.DataFrame({
        C.TEXT_COL: df[text_col].fillna("").astype(str).str.strip(),
        "sentiment": _to_label_id(df[cols["sentiment"]], "sentiment"),
        "topic": _to_label_id(df[cols["topic"]], "topic"),
    })
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Đọc file cục bộ
# --------------------------------------------------------------------------- #
def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_json(path)


def _find_split_files(raw_dir: Path, split: str) -> list[Path]:
    aliases = _SPLIT_ALIASES[split]
    files = [p for p in sorted(raw_dir.rglob("*")) if p.is_file() and p.suffix.lower() in _SUFFIXES]
    exact = [p for p in files if p.stem.lower() in aliases]          # train.csv
    if exact:
        return exact[:1]
    shards = [p for p in files if any(p.stem.lower().startswith(a) for a in aliases)]  # train-0000-of-0001.parquet
    if shards:
        return shards
    return [p for p in files if p.parent.name.lower() in aliases]    # default/train/0000.parquet


def _read_uit_original(raw_dir: Path, split: str) -> pd.DataFrame | None:
    """Định dạng gốc của UIT: <split>/sents.txt, sentiments.txt, topics.txt (mỗi dòng một mẫu)."""
    for sents in sorted(raw_dir.rglob("sents.txt")):
        if sents.parent.name.lower() not in _SPLIT_ALIASES[split]:
            continue
        folder = sents.parent
        try:
            sentences = sents.read_text(encoding="utf-8").splitlines()
            sentiments = (folder / "sentiments.txt").read_text(encoding="utf-8").split()
            topics = (folder / "topics.txt").read_text(encoding="utf-8").split()
        except FileNotFoundError:
            continue
        if not (len(sentences) == len(sentiments) == len(topics)):
            raise ValueError(f"{folder}: số dòng của sents/sentiments/topics không khớp "
                             f"({len(sentences)}/{len(sentiments)}/{len(topics)}).")
        return _standardize(pd.DataFrame({C.TEXT_COL: sentences, "sentiment": sentiments,
                                          "topic": topics}), str(folder))
    return None


def load_local(raw_dir: Path = C.RAW_DIR) -> dict[str, pd.DataFrame] | None:
    """Nạp đủ 3 split từ ``raw_dir``; trả về None nếu thiếu bất kỳ split nào."""
    if not raw_dir.exists():
        return None
    splits: dict[str, pd.DataFrame] = {}
    for split in C.SPLITS:
        files = _find_split_files(raw_dir, split)
        if files:
            df = pd.concat([_read_table(p) for p in files], ignore_index=True)
            splits[split] = _standardize(df, ", ".join(p.name for p in files))
            continue
        df_uit = _read_uit_original(raw_dir, split)
        if df_uit is None:
            return None
        splits[split] = df_uit
    return splits


# --------------------------------------------------------------------------- #
# Tải từ Hugging Face
# --------------------------------------------------------------------------- #
def download_from_hf(raw_dir: Path = C.RAW_DIR) -> dict[str, pd.DataFrame]:
    """Tải UIT-VSFC từ Hugging Face Hub và lưu thành CSV trong ``raw_dir``."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Chưa cài thư viện 'datasets'. Chạy: pip install -r requirements.txt") from exc

    errors: list[str] = []
    ds = None
    # Cách 1: nạp bình thường. Cách 2: nhánh Parquet tự động chuyển đổi của Hub
    # (dùng khi phiên bản `datasets` mới không còn hỗ trợ dataset dạng script).
    for kwargs in ({}, {"revision": "refs/convert/parquet"}):
        try:
            ds = load_dataset(C.HF_DATASET, **kwargs)
            break
        except Exception as exc:  # noqa: BLE001 - muốn thử phương án kế tiếp
            errors.append(f"  - {kwargs or 'mặc định'}: {type(exc).__name__}: {exc}")
    if ds is None:
        raise RuntimeError(
            "Không tải được dataset từ Hugging Face:\n" + "\n".join(errors) + "\n"
            "Cách khắc phục: mở https://huggingface.co/datasets/" + C.HF_DATASET +
            " -> tab 'Files and versions', tải file train/validation/test (.parquet hoặc .csv) "
            f"và đặt vào thư mục '{raw_dir}'."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, pd.DataFrame] = {}
    for split in C.SPLITS:
        if split not in ds:
            raise KeyError(f"Dataset không có split '{split}'. Các split hiện có: {list(ds.keys())}")
        df = _standardize(ds[split].to_pandas(), f"HF:{split}")
        df.to_csv(raw_dir / f"{split}.csv", index=False, encoding="utf-8")
        splits[split] = df
    return splits


# --------------------------------------------------------------------------- #
# API chính
# --------------------------------------------------------------------------- #
def load_splits(raw_dir: Path = C.RAW_DIR, force_download: bool = False) -> dict[str, pd.DataFrame]:
    """Trả về ``{"train": df, "validation": df, "test": df}`` với 3 cột sentence/sentiment/topic."""
    if not force_download:
        local = load_local(raw_dir)
        if local is not None:
            return local
    return download_from_hf(raw_dir)


def check_sizes(splits: dict[str, pd.DataFrame]) -> None:
    """In kích thước và cảnh báo nếu lệch so với dataset card."""
    for name, df in splits.items():
        expected = C.EXPECTED_SIZES.get(name)
        note = "" if expected == len(df) else f"  [!] khác kỳ vọng ({expected:,})"
        print(f"  {name:<11}: {len(df):>6,} mẫu{note}")
    print(f"  {'tổng':<11}: {sum(len(d) for d in splits.values()):>6,} mẫu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải/nạp bộ dữ liệu UIT-VSFC.")
    parser.add_argument("--raw-dir", type=Path, default=C.RAW_DIR, help="Thư mục chứa dữ liệu thô.")
    parser.add_argument("--force-download", action="store_true", help="Bỏ qua file cục bộ, tải lại từ Hugging Face.")
    args = parser.parse_args()

    splits = load_splits(args.raw_dir, args.force_download)
    print("Đã nạp dữ liệu UIT-VSFC:")
    check_sizes(splits)
    print(splits["train"].head(3).to_string())


if __name__ == "__main__":
    main()
