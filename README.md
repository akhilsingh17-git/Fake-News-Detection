# 📰 AI Fake News Detection 

An end-to-end NLP web application that classifies news articles as **Real** or **Fake** using a fine-tuned **BERT** model. The application provides real-time predictions with confidence scores through an interactive Streamlit interface.

## 🚀 Live Demo

🔗 https://fake-news-detection-123456.streamlit.app/

---

## 📌 Project Overview

Fake news spreads rapidly across digital platforms, making manual verification difficult. This project leverages **BERT (Bidirectional Encoder Representations from Transformers)** to classify news articles based on patterns learned from labeled datasets.

> **Note:** This project performs text classification and does **not** verify news against live internet sources.

---

## ✨ Features

- 🤖 Fine-tuned BERT model for fake news detection
- 📊 Confidence score for every prediction
- 📰 Sample Real and Fake news examples
- 🌐 Interactive Streamlit web application
- ☁️ Model hosted on Hugging Face
- 🚀 Deployed on Streamlit Community Cloud

---

## 🧠 Model

- **Base Model:** `bert-base-uncased`
- **Task:** Binary Text Classification
- **Framework:** PyTorch + Hugging Face Transformers
- **Hosted Model:** https://huggingface.co/Akhilsingh17/fake-news-bert

---

## ⚙️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
- Google Colab
- Hugging Face Hub
- Git & GitHub

---

## 🔄 Workflow

```text
News Article
      │
      ▼
BERT Tokenizer
      │
      ▼
Fine-Tuned BERT Model
      │
      ▼
Prediction
      │
      ▼
Real / Fake + Confidence Score
```

---

## 📂 Repository Structure

```
Fake-News-Detection/
│
├── app.py
├── Fake_News_Detection.ipynb
├── requirements.txt
└── README.md
```

---

## 💻 Run Locally

```bash
git clone https://github.com/akhilsingh17-git/Fake-News-Detection.git

cd Fake-News-Detection

pip install -r requirements.txt

streamlit run app.py
```

---

## 🌐 Deployment

- **Code Repository:** GitHub
- **Model Hosting:** Hugging Face Hub
- **Application Hosting:** Streamlit Community Cloud

Deployment Flow:

```text
Google Colab → GitHub → Streamlit Cloud
                    │
                    ▼
              Hugging Face (BERT Model)
```

---

## 🔮 Future Improvements

- Live fact-checking using News APIs
- Browser Extension
- Explainable AI (XAI)
- Multilingual Fake News Detection

---

## 👨‍💻 Author

**Akhil Singh**

- GitHub: https://github.com/akhilsingh17-git
- Hugging Face: https://huggingface.co/Akhilsingh17

⭐ If you found this project useful, consider giving it a star!
