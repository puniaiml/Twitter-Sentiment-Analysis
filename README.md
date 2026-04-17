# 🧠 SentimentAI · Twitter Sentiment Analysis

A deep learning web app that classifies tweets into 4 sentiment categories using a **BERT + BiLSTM** architecture, achieving **~90% accuracy** on 74,681 training tweets.

## 🚀 Live Demo
[Click here to try the app](#) <!-- replace with your Streamlit Cloud URL after deploying -->

## 🏗️ Model Architecture
| Layer | Details |
|---|---|
| BERT Encoder | bert-base-uncased |
| Dropout | p = 0.10 |
| BiLSTM | 2 layers · hidden size = 128 |
| Linear | 256 → 4 classes |
| Log-Softmax | output activation |

## 🏷️ Sentiment Classes
| Class | Description |
|---|---|
| 🟢 Positive | Happiness, excitement, satisfaction, praise |
| 🔴 Negative | Anger, sadness, frustration, criticism |
| 🟡 Neutral | Factual, informational, no strong emotion |
| ⚪ Irrelevant | Unrelated, spam, or off-topic content |

## 📊 Performance
| Metric | Score |
|---|---|
| Accuracy | 90.0% |
| Precision | 90.0% |
| Recall | 90.0% |
| F1 Score | 90.0% |

## 📁 Dataset
- **Training set:** 74,681 tweets
- **Source:** Twitter Sentiment Analysis dataset
- **Classes:** 4 (Positive, Negative, Neutral, Irrelevant)

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Model:** PyTorch + HuggingFace Transformers
- **Visualization:** Plotly
- **Model Hosting:** HuggingFace Hub

## ⚙️ Run Locally

1. Clone the repository:
```bash
git clone https://github.com/puneethas26/sentiment-bert-bilstm.git
cd sentiment-bert-bilstm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

4. Click **🚀 Load / Reload Model** in the sidebar — the model will be downloaded automatically from HuggingFace on first run.
