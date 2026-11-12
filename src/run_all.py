"""Chạy toàn bộ pipeline bằng một lệnh.

    python -m src.run_all                     # đầy đủ: preprocess -> eda -> train -> evaluate -> make_latex
    python -m src.run_all --skip-embedding    # nhanh, không cần PyTorch/sentence-transformers
    python -m src.run_all --ablation          # thêm thí nghiệm đối chứng
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from . import config as C


def run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n{'=' * 78}\n>>> {' '.join(cmd[1:])}\n{'=' * 78}", flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=C.ROOT)
    print(f"<<< xong trong {time.perf_counter() - t0:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy toàn bộ pipeline UIT-VSFC.")
    parser.add_argument("--tokenizer", choices=["pyvi", "underthesea", "none"], default=C.DEFAULT_TOKENIZER)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    train_args = (["--skip-embedding"] if args.skip_embedding else []) + (["--ablation"] if args.ablation else [])
    run("src.data_loader")
    run("src.preprocess", "--tokenizer", args.tokenizer)
    run("src.eda")
    run("src.train", *train_args)
    run("src.evaluate")
    run("src.make_latex")
    print("\nHoàn tất. Chạy giao diện demo:  streamlit run app.py")


if __name__ == "__main__":
    main()
