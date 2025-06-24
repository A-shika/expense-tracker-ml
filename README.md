# 💸 ML-Based Personal Expense Tracker App

A smart expense tracker built using **Streamlit and Machine Learning** that classifies your expenses based on their description. Users can log daily expenses, view categorized history, visualize spending trends, and even edit the predicted categories manually.

---

## 🚀 Tech Stack

### 🎯 Core Stack:
- Streamlit (Frontend + App Framework)
- Python
- Scikit-learn (for ML model)
- Pandas
- Altair (for data visualization)

### 🔍 ML Model:
- Text Classification using Logistic Regression / Naive Bayes
- TF-IDF Vectorizer / Count Vectorizer
- Trained on custom labeled expense descriptions

---

## ✨ Features

- 🤖 Predicts expense category using ML
- 📝 Add, view, and delete expenses
- ✏️ Manually edit predicted categories
- 📊 Total and category-wise spending summary
- 📅 Filter by date and category
- 📈 Visualize spending via bar charts
- 💾 CSV-based data storage (auto-managed)
- 📦 Easy to deploy via Streamlit Cloud

---

## 📂 Folder Structure

```bash
expense-tracker-ml/
├── app.py                   # Main Streamlit app
├── train_model.py           # Model training script
├── model/
│   └── expense_model.pkl    # Trained ML model
├── data/
│   └── expense.csv          # Stored expenses
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
