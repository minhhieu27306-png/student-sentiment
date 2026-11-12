# Tóm tắt kết quả (tập test)

Sinh tự động bởi `python -m src.make_latex`.

## Cảm xúc (sentiment)

Mô hình được chọn (theo F1-macro trên validation): **BoW + Linear SVM** - Accuracy 89.01%, F1-macro 75.71%.

| Đặc trưng | Mô hình | Acc | P-macro | R-macro | F1-macro | F1-weighted |
|---|---|---:|---:|---:|---:|---:|
| BoW | Naive Bayes | 85.98 | 67.42 | 65.90 | 66.37 | 85.48 |
| BoW | Logistic Regression | 87.84 | 73.82 | 76.91 | 75.06 | 88.24 |
| BoW | Linear SVM | 89.01 | 75.32 | 76.16 | 75.71 | 89.10 |
| BoW | Random Forest | 87.24 | 72.91 | 70.71 | 71.57 | 86.92 |
| BoW | KNN | 83.26 | 68.92 | 65.71 | 66.87 | 82.73 |
| TF-IDF | Naive Bayes | 87.40 | 76.95 | 64.50 | 65.22 | 85.82 |
| TF-IDF | Logistic Regression | 88.25 | 72.62 | 73.36 | 72.93 | 88.38 |
| TF-IDF | Linear SVM | 89.39 | 75.65 | 72.21 | 73.47 | 88.95 |
| TF-IDF | Random Forest | 87.43 | 75.58 | 68.54 | 70.38 | 86.63 |
| TF-IDF | KNN | 82.82 | 65.97 | 63.43 | 64.26 | 82.15 |

## Chủ đề (topic)

Mô hình được chọn (theo F1-macro trên validation): **TF-IDF + Linear SVM** - Accuracy 85.69%, F1-macro 75.33%.

| Đặc trưng | Mô hình | Acc | P-macro | R-macro | F1-macro | F1-weighted |
|---|---|---:|---:|---:|---:|---:|
| BoW | Naive Bayes | 82.06 | 67.46 | 70.97 | 69.06 | 82.29 |
| BoW | Logistic Regression | 81.65 | 69.83 | 78.16 | 72.51 | 83.14 |
| BoW | Linear SVM | 84.59 | 73.29 | 77.63 | 74.93 | 85.30 |
| BoW | Random Forest | 85.79 | 76.48 | 70.72 | 73.31 | 85.25 |
| BoW | KNN | 82.94 | 73.11 | 62.68 | 66.61 | 81.68 |
| TF-IDF | Naive Bayes | 84.14 | 74.57 | 66.57 | 68.42 | 83.00 |
| TF-IDF | Logistic Regression | 83.39 | 71.62 | 77.19 | 73.95 | 84.21 |
| TF-IDF | Linear SVM | 85.69 | 74.64 | 76.20 | 75.33 | 85.86 |
| TF-IDF | Random Forest | 81.74 | 69.12 | 77.01 | 72.18 | 82.90 |
| TF-IDF | KNN | 82.91 | 76.76 | 62.97 | 67.35 | 81.18 |
