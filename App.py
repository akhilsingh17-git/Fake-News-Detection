import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for cleaner look
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

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

# Sidebar with project info
with st.sidebar:
    st.header("About This Project")
    st.write(
        "This app uses a fine-tuned **BERT** transformer model to classify "
        "news articles as **Real** or **Fake**."
    )
    st.markdown("---")
    st.markdown("**Model:** `Akhilsingh17/fake-news-bert`")
    st.markdown("**Framework:** PyTorch + Transformers + Streamlit")
    st.markdown("---")
    st.info("Tip: Paste a full paragraph rather than a short headline for better accuracy.")

# Initialize session history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Main Header
st.markdown('<div class="main-title">📰 AI Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Verify news articles instantly using a fine-tuned BERT model.</div>', unsafe_allow_html=True)

# Sample buttons
col1, col2, col3 = st.columns([1, 1, 2])
if col1.button("📄 Load Sample Real News", use_container_width=True):
    st.session_state["news_input"] = (
        "The Reserve Bank of India kept the repo rate unchanged on Friday, "
        "citing steady GDP growth and moderating inflation across major sectors."
    )
if col2.button("🚨 Load Sample Fake News", use_container_width=True):
    st.session_state["news_input"] = (
        "SHOCKING: Scientists confirm drinking saltwater every morning cures "
        "all diseases instantly and governments are hiding the secret!"
    )

news = st.text_area("Paste News Article Here", key="news_input", height=180)

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
    is_real = prediction.item() == 1
    return is_real, confidence.item() * 100

if st.button("🔍 Analyze News", type="primary"):
    if news.strip() == "":
        st.warning("Please enter some news text first.")
    else:
        with st.spinner("Analyzing article with BERT..."):
            is_real, conf = predict_news(news)

        label = "🟢 True News" if is_real else "🔴 Fake News"
        if is_real:
            st.success(f"### {label}")
        else:
            st.error(f"### {label}")

        st.write(f"**Confidence Score:** {conf:.2f}%")
        st.progress(int(conf))

        # Save to history
        st.session_state["history"].insert(0, {
            "snippet": news[:100] + ("..." if len(news) > 100 else ""),
            "result": label,
            "confidence": f"{conf:.2f}%"
        })

# Display Check History
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("🕒 Recent Checks")
    for item in st.session_state["history"][:5]:
        st.write(f"**{item['result']}** ({item['confidence']}) — *{item['snippet']}*")
