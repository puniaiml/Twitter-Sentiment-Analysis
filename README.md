# 🧠 SentimentAI · Twitter Sentiment Analysis

A deep learning web app that classifies tweets into 4 sentiment categories
using a **BERT + BiLSTM** architecture, achieving **~90% accuracy**.

## 🚀 Live Demo
[Click here to try the app](https://your-app.streamlit.app)

---

## 🏗️ Model Architecture

| Layer | Details |
|---|---|
| BERT Encoder | bert-base-uncased |
| Dropout | p = 0.10 |
| BiLSTM | 2 layers · hidden size = 128 |
| Linear | 256 → 4 classes |
| Log-Softmax | output activation |

---

## 🏷️ Sentiment Classes

| Class | Description |
|---|---|
| 🟢 Positive | Happiness, excitement, satisfaction, praise |
| 🔴 Negative | Anger, sadness, frustration, criticism |
| 🟡 Neutral | Factual, informational, no strong emotion |
| ⚪ Irrelevant | Unrelated, spam, or off-topic content |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 90.0% |
| Precision | 90.0% |
| Recall | 90.0% |
| F1 Score | 90.0% |

- **Training set:** 74,681 tweets  
- **Test set:** 7,200 tweets  

---

## 🛠️ Tech Stack

- **App:** Streamlit  
- **Model:** PyTorch + HuggingFace Transformers  
- **Visualization:** Plotly  
- **Model Hosting:** HuggingFace Hub  

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/puniaiml/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then click **🚀 Load / Reload Model** in the sidebar.  
The model downloads automatically from HuggingFace on first run — no manual setup needed.

---

## 📂 Project Structure