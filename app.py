"""Giao diện demo Streamlit: nhập một câu phản hồi tiếng Việt -> dự đoán Cảm xúc + Chủ đề.

Chạy:
    streamlit run app.py

Yêu cầu: đã huấn luyện mô hình (python -m src.train) để có file trong thư mục models/.
"""
from __future__ import annotations

import streamlit as st

from src import config as C
from src.predict import FeedbackPredictor

st.set_page_config(page_title="Phân loại phản hồi sinh viên", page_icon="🎓", layout="centered")

# Màu huy hiệu (nền, chữ) - độ tương phản chữ/nền >= 4.5:1.
SENTIMENT_STYLE = {
    "negative": ("#C0392B", "#FFFFFF", "😞"),
    "neutral": ("#F4B400", "#1A1A1A", "😐"),
    "positive": ("#1E8449", "#FFFFFF", "😊"),
}
TOPIC_STYLE = {
    "lecturer": ("#2E6DB4", "#FFFFFF", "👩‍🏫"),
    "training_program": ("#7D3C98", "#FFFFFF", "📚"),
    "facility": ("#9C640C", "#FFFFFF", "🏫"),
    "others": ("#566573", "#FFFFFF", "💬"),
}
EXAMPLES = {
    "Khen giảng viên": "Thầy dạy rất dễ hiểu và nhiệt tình, em rất hài lòng.",
    "Chê cơ sở vật chất": "Phòng học nóng, máy chiếu hay hỏng và wifi quá yếu.",
    "Nhận xét chương trình": "Nội dung môn học còn nặng lý thuyết, cần thêm thực hành.",
    "Ý kiến trung tính": "Môn học này bình thường, không có ý kiến gì thêm.",
}
MAX_CHARS = 1000


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def load_predictor() -> FeedbackPredictor:
    return FeedbackPredictor()


def badge(text: str, bg: str, fg: str, emoji: str) -> str:
    return (f'<div style="display:inline-block;padding:.45rem 1rem;border-radius:999px;'
            f'background:{bg};color:{fg};font-weight:700;font-size:1.15rem;">{emoji} {text}</div>')


def prob_bars(probs: dict[str, float], top_label: str) -> str:
    rows = []
    for label, p in probs.items():
        weight = "700" if label == top_label else "400"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:.6rem;margin:.25rem 0;font-size:.9rem;">'
            f'<div style="width:9.5rem;font-weight:{weight};">{label}</div>'
            f'<div style="flex:1;background:rgba(128,128,128,.2);border-radius:6px;height:.7rem;">'
            f'<div style="width:{p * 100:.1f}%;background:#4C78A8;height:100%;border-radius:6px;"></div></div>'
            f'<div style="width:3.6rem;text-align:right;font-variant-numeric:tabular-nums;">{p * 100:.1f}%</div></div>')
    return "".join(rows)


def render_result(title: str, res: dict, style: dict, model_meta: dict) -> None:
    bg, fg, emoji = style[res["label"]]
    with st.container(border=True):
        st.caption(title)
        st.markdown(badge(res["label_vi"], bg, fg, emoji), unsafe_allow_html=True)
        st.markdown(prob_bars(res["probabilities"], res["label_vi"]), unsafe_allow_html=True)
        st.caption(f"Mô hình: {C.FEATURE_DISPLAY[model_meta['feature']]} + {C.MODEL_DISPLAY[model_meta['model']]}")


def set_example(text: str) -> None:
    st.session_state["feedback_text"] = text


# --------------------------------------------------------------------------- #
st.title("🎓 Phân loại phản hồi sinh viên")
st.write("Nhập một câu phản hồi bằng tiếng Việt, hệ thống sẽ dự đoán **cảm xúc** và **chủ đề** "
         "(bộ dữ liệu UIT-VSFC).")

try:
    predictor = load_predictor()
except FileNotFoundError as err:
    st.error(f"{err}")
    st.info("Chạy lần lượt các lệnh sau trong Terminal của VS Code rồi tải lại trang:\n\n"
            "```\npython -m src.preprocess\npython -m src.train\n```")
    st.stop()

if "history" not in st.session_state:
    st.session_state["history"] = []

st.caption("Thử nhanh một ví dụ:")
cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES.items()):
    col.button(label, on_click=set_example, args=(text,), width="stretch")

with st.form("predict_form"):
    text = st.text_area("Câu phản hồi", key="feedback_text", height=120, max_chars=MAX_CHARS,
                        placeholder="Ví dụ: Giảng viên dạy rất nhiệt tình nhưng phòng học hơi nóng...")
    submitted = st.form_submit_button("Phân tích", type="primary")

if submitted:
    if not text.strip():
        st.warning("Vui lòng nhập một câu phản hồi.")
    else:
        result = predictor.analyze(text)
        left, right = st.columns(2)
        with left:
            render_result("Cảm xúc (sentiment)", result["sentiment"], SENTIMENT_STYLE, predictor.sentiment.meta)
        with right:
            render_result("Chủ đề (topic)", result["topic"], TOPIC_STYLE, predictor.topic.meta)
        with st.expander("Văn bản sau tiền xử lý"):
            st.code(result["sentiment"]["processed_text"] or "(rỗng)", language=None)
            if predictor.sentiment.meta["text_col"] != predictor.topic.meta["text_col"]:
                st.code(result["topic"]["processed_text"] or "(rỗng)", language=None)
        st.session_state["history"].insert(0, {
            "Câu phản hồi": text.strip()[:80] + ("…" if len(text.strip()) > 80 else ""),
            "Cảm xúc": result["sentiment"]["label_vi"],
            "Chủ đề": result["topic"]["label_vi"],
        })

if st.session_state["history"]:
    st.subheader("Lịch sử dự đoán")
    st.dataframe(st.session_state["history"][:10], width="stretch", hide_index=True)

with st.sidebar:
    st.header("Thông tin mô hình")
    for task, tm in (("Cảm xúc", predictor.sentiment), ("Chủ đề", predictor.topic)):
        m = tm.meta
        st.markdown(f"**{task}**: {C.FEATURE_DISPLAY[m['feature']]} + {C.MODEL_DISPLAY[m['model']]}")
        st.caption(f"F1-macro (test): {m['test_f1_macro'] * 100:.2f}% · Accuracy: {m['test_accuracy'] * 100:.2f}%")
    st.divider()
    st.caption("Dữ liệu: UIT-VSFC (uitnlp/vietnamese_students_feedback). "
               "Điểm hiển thị của Linear SVM là softmax của khoảng cách đến siêu phẳng, chỉ mang tính tương đối.")
