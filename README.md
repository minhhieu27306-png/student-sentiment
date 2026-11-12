# 🎓 Student Sentiment Classification (UIT-VSFC)

Phân loại tự động **cảm xúc** và **chủ đề** của ý kiến phản hồi sinh viên bằng tiếng Việt, trên bộ dữ liệu
[UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) (`uitnlp/vietnamese_students_feedback`).

| Bài toán | Số lớp | Các lớp (chỉ số) |
|---|:---:|---|
| `sentiment` (cảm xúc) | 3 | 0 Negative · 1 Neutral · 2 Positive |
| `topic` (chủ đề) | 4 | 0 Lecturer · 1 Training program · 2 Facility · 3 Others |

Dữ liệu: 16.175 câu, chia sẵn Train 11.426 / Validation 1.583 / Test 3.166.

Dự án gồm: tiền xử lý tiếng Việt → so sánh **3 loại đặc trưng** (BoW, TF-IDF, Sentence Embeddings) × **5 thuật toán**
(Naive Bayes, Logistic Regression, Linear SVM, Random Forest, KNN) × 2 bài toán → ứng dụng **Streamlit** → báo cáo **LaTeX**.

## Cấu trúc thư mục

```
├── data/
│   ├── raw/             # train/validation/test .csv (tải về, không đưa lên Git)
│   └── processed/       # dữ liệu đã tiền xử lý + cache embedding
├── notebooks/
│   ├── eda.ipynb        # khám phá dữ liệu
│   └── train.ipynb      # huấn luyện từng bước
├── src/
│   ├── config.py        # đường dẫn, tên nhãn, hằng số
│   ├── data_loader.py   # tải/nạp dữ liệu (Hugging Face hoặc file cục bộ)
│   ├── preprocess.py    # làm sạch, teencode, tách từ
│   ├── features.py      # BoW, TF-IDF, Sentence Embeddings
│   ├── eda.py           # thống kê + biểu đồ
│   ├── train.py         # huấn luyện, chọn siêu tham số, lưu mô hình tốt nhất
│   ├── evaluate.py      # báo cáo phân loại, ma trận nhầm lẫn, phân tích lỗi
│   ├── predict.py       # nạp mô hình để dự đoán (dùng cho app)
│   ├── make_latex.py    # kết quả → bảng/macro LaTeX + bảng Markdown
│   └── run_all.py       # chạy toàn bộ pipeline bằng một lệnh
├── models/              # *_model.pkl, *_vectorizer.pkl, *_meta.json
├── results/             # metrics_all.csv, ablation.csv, hình, báo cáo phân loại
├── report/              # main.tex (+ tables/, figures/ do make_latex sinh ra)
├── app.py               # giao diện Streamlit
├── requirements.txt
└── README.md
```

## Cài đặt (VS Code)

Mở thư mục dự án trong VS Code (`File → Open Folder`), mở Terminal (`Ctrl+``), rồi:

```bash
# 1. Tạo môi trường ảo (khuyến nghị Python 3.10 - 3.12)
python -m venv .venv

# 2. Kích hoạt
#    Windows (PowerShell):  .venv\Scripts\Activate.ps1
#       (nếu bị chặn: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
#    Windows (cmd):         .venv\Scripts\activate.bat
#    macOS / Linux:         source .venv/bin/activate
source .venv/bin/activate

# 3. Cài thư viện
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sau đó nhấn `Ctrl+Shift+P` → **Python: Select Interpreter** → chọn `.venv`. Với notebook, chọn kernel `.venv` ở góc trên bên phải.

> `sentence-transformers` kéo theo PyTorch (nặng, vài trăm MB đến vài GB). Nếu chỉ muốn chạy nhanh với BoW/TF-IDF,
> thêm cờ `--skip-embedding` ở các lệnh bên dưới. Lần đầu dùng embedding, mô hình
> `paraphrase-multilingual-MiniLM-L12-v2` sẽ được tải từ Hugging Face.

## Chạy pipeline

Chạy tất cả bằng một lệnh:

```bash
python -m src.run_all                    # đầy đủ
python -m src.run_all --skip-embedding   # nhanh (không cần PyTorch)
python -m src.run_all --ablation         # thêm thí nghiệm đối chứng (tách từ, class_weight)
```

Hoặc từng bước (trong `.vscode/launch.json` đã có sẵn cấu hình chạy/debug cho từng bước):

| Bước | Lệnh | Đầu ra |
|---|---|---|
| 1. Tải dữ liệu | `python -m src.data_loader` | `data/raw/*.csv` |
| 2. Tiền xử lý | `python -m src.preprocess` | `data/processed/*.csv` |
| 3. EDA | `python -m src.eda` | `results/eda_summary.json`, `results/figures/*.png` |
| 4. Huấn luyện | `python -m src.train [--skip-embedding] [--ablation]` | `results/metrics_all.csv`, `models/*` |
| 5. Đánh giá | `python -m src.evaluate` | báo cáo phân loại, ma trận nhầm lẫn, `results/errors_*.csv` |
| 6. Bảng LaTeX | `python -m src.make_latex` | `report/tables/*.tex`, `results/summary.md` |
| 7. Demo | `streamlit run app.py` | giao diện tại http://localhost:8501 |

**Không có Internet / không tải được từ Hugging Face?** Tải thủ công các file train/validation/test (`.parquet` hoặc `.csv`)
từ tab *Files and versions* của dataset và đặt vào `data/raw/`. `data_loader` tự nhận diện tên file
(`train.*`, `validation.*`/`val.*`/`dev.*`, `test.*`) và cả định dạng gốc của UIT (thư mục `train|dev|test` chứa `sents.txt`, `sentiments.txt`, `topics.txt`).

## Phương pháp

**Tiền xử lý** (`src/preprocess.py`): chuẩn hóa Unicode NFC → giải mã HTML entity, xóa URL → chữ thường → xóa số/dấu câu/ký tự đặc biệt,
rút gọn chữ kéo dài → chuẩn hóa teencode (`ko → không`, `đc → được`, …) → tách từ bằng `pyvi` (`giảng viên → giảng_viên`).
Có thể đổi sang `underthesea` bằng `--tokenizer underthesea`.

**Đặc trưng** (`src/features.py`): BoW và TF-IDF với n-gram (1, 2), `min_df=2`; Sentence Embeddings
(`paraphrase-multilingual-MiniLM-L12-v2`, chuẩn hóa L2, cache vào `data/processed/`).

**Giao thức đánh giá** (tránh rò rỉ dữ liệu):
vectorizer chỉ fit trên train; siêu tham số của từng mô hình **và** mô hình tốt nhất đều được chọn theo **F1-macro trên validation**;
tập test chỉ dùng để báo cáo. Logistic Regression / Linear SVM / Random Forest dùng `class_weight="balanced"` vì dữ liệu mất cân bằng lớp.
Chỉ số: Accuracy, Precision, Recall, F1 (macro và weighted), ma trận nhầm lẫn.

## Kết quả

Kết quả đầy đủ được sinh ra sau khi chạy pipeline, **không ghi tay** để tránh sai lệch:

- `results/metrics_all.csv`: toàn bộ chỉ số (validation + test) của mọi cấu hình;
- `results/summary.md`: bảng Markdown, dán trực tiếp vào mục này;
- `results/figures/model_comparison.png`, `confusion_matrix_*.png`.

<!-- Sau khi chạy `python -m src.run_all`, dán nội dung results/summary.md vào đây. -->

## Giao diện demo

```bash
streamlit run app.py
```

Nhập một câu phản hồi tiếng Việt (hoặc chọn ví dụ có sẵn) → hiển thị **Cảm xúc** và **Chủ đề** với màu sắc tương ứng
(xanh: tích cực, đỏ: tiêu cực, vàng: trung tính) cùng điểm của từng lớp. Với Linear SVM, điểm là softmax của `decision_function`,
chỉ mang tính tương đối, không phải xác suất đã hiệu chỉnh.

## Báo cáo LaTeX

`report/main.tex` (XeLaTeX, tiếng Việt) lấy toàn bộ số liệu từ `report/tables/*.tex` do `make_latex` sinh ra. Các chỗ cần tự viết nhận xét
được đánh dấu bằng `\todo{...}` (chữ đỏ).

```bash
python -m src.make_latex
cd report && latexmk -xelatex main.tex
```

Trong VS Code dùng extension *LaTeX Workshop* (đã cấu hình sẵn recipe XeLaTeX), hoặc tải thư mục `report/` lên Overleaf và chọn compiler **XeLaTeX**.
Muốn chèn ảnh giao diện: lưu ảnh chụp màn hình thành `report/figures/app_screenshot.png`.

## Đưa mã nguồn lên GitHub

```bash
git init
git add .
git commit -m "Student sentiment classification on UIT-VSFC"
git branch -M main
git remote add origin https://github.com/<tai-khoan>/<ten-repo>.git   # tạo repo rỗng trên github.com trước
git push -u origin main
```

`.gitignore` đã bỏ qua `.venv/`, `__pycache__/`, dữ liệu trong `data/` và tệp tạm của LaTeX. Các file `models/*.pkl` mặc định **được giữ**
để app chạy ngay sau khi clone; nếu mô hình > 50 MB (ví dụ Random Forest), bỏ comment dòng `models/*.pkl` trong `.gitignore`
(GitHub giới hạn 100 MB mỗi file).

## Hạn chế & hướng phát triển

- Chỉ dùng mô hình học máy cổ điển và embedding đóng băng; có thể tinh chỉnh PhoBERT để cải thiện thêm.
- Có thể có câu trùng giữa các tập (xem `results/eda_summary.json`), làm điểm test cao hơn dữ liệu hoàn toàn mới.
- Từ điển teencode nhỏ, nên bổ sung sau khi xem `results/errors_*.csv`.

## Trích dẫn

Dữ liệu: K. V. Nguyen, V. D. Nguyen, P. X. V. Nguyen, T. T. H. Truong, N. L.-T. Nguyen.
*UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis.* KSE 2018. Xem dataset card trên Hugging Face để biết điều khoản sử dụng dữ liệu.
