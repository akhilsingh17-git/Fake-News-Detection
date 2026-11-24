import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

@st.cache_resource
def load_model():
    model_path = "Akhilsingh17/fake-news-bert"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

st.title("📰 AI Fake News Detection")
st.write("Paste any news article below and click **Check News**.")

col1, col2 = st.columns(2)
if col1.button("Load Sample Real News"):
    st.session_state["news_input"] = (
        "The Reserve Bank of India kept the repo rate unchanged on Friday, "
        "citing steady GDP growth and moderating inflation across major sectors."
    )
if col2.button("Load Sample Fake News"):
    st.session_state["news_input"] = (
        "SHOCKING: Scientists confirm drinking saltwater every morning cures "
        "all diseases instantly and governments are hiding the secret!"
    )

news = st.text_area("Enter News", key="news_input", height=180)

def predict_news(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
    label = "🟢 True News" if prediction.item() == 1 else "🔴 Fake News"
    return label, confidence.item() * 100

if st.button("Check News", type="primary"):
    if news.strip() == "":
        st.warning("Please enter some news.")
    else:
        with st.spinner("Analyzing article..."):
            label, conf = predict_news(news)
        if "True" in label:
            st.success(f"{label} — **{conf:.2f}% confidence**")
        else:
            st.error(f"{label} — **{conf:.2f}% confidence**")
