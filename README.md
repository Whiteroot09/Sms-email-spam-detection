# 📩 SMS Spam Detection using Machine Learning

A machine learning project to automatically detect spam SMS messages using natural language processing (NLP) and classification algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-green)
![NLP](https://img.shields.io/badge/NLP-TFIDF-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🔍 Project Overview

This project builds an intelligent SMS spam filter using a dataset of real SMS messages labeled as *spam* or *ham (not spam)*. It uses text preprocessing techniques and a machine learning model to classify incoming messages in real time.

---

## 📊 Dataset

- **Source**: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Size**: ~5,500 messages
- **Labels**: `spam`, `ham`

---

## 🧠 Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – Data analysis
  - `sklearn` – ML models and preprocessing
  - `nltk` – Text cleaning
  - `matplotlib`, `seaborn` – Visualizations
- **Model**: Multinomial Naive Bayes

---

## 🔧 Features

- Clean and preprocess SMS text using NLP techniques
- Convert text to numerical features using TF-IDF Vectorization
- Train and evaluate an ML classifier (Naive Bayes)
- Predict whether a new SMS is spam or not
- Visualize performance metrics like confusion matrix & accuracy

---

## 🖼️ Screenshots

![Confusion Matrix](assets/confusion_matrix.png)
> Confusion matrix showing classifier performance

---

## 🚀 Getting Started

### ✅ Requirements

- Python 3.8+
- `pip install -r requirements.txt`

### ⚙️ How to Run

```bash
# Clone the repository
git clone https://github.com/Whiteroot09/sms-spam-detection.git
cd sms-spam-detection

# Install dependencies
pip install -r requirements.txt

# Run the script
python sms_spam_detector.py
