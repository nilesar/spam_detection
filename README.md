# 📩 Email/SMS Spam Classifier

A Streamlit-based web application that classifies text messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and Machine Learning.

---

## 🚀 Features

- Text preprocessing: lowercasing, tokenization, stopword removal, punctuation removal, stemming.
- TF-IDF vectorization for text representation.
- Trained using multiple ML models; best performer: **Multinomial Naive Bayes (MNB)**.
- Real-time prediction using a simple Streamlit interface.

---

## 📁 Project Structure

📦spam_classifier_app
├── app.py # Main Streamlit app

├── model.pkl # Trained ML model (MNB)

├── vectorizer.pkl # TF-IDF Vectorizer

├── README.md # Project documentation

├── requirements.txt # Required dependencies




## 🔍 Model Selection  
Tested multiple classifiers:

| Model                    | Accuracy | Comments                         |
|--------------------------|----------|----------------------------------|
| Multinomial Naive Bayes | ✅ Best  | Fast, simple, accurate           |
| Logistic Regression      | Good     | Slightly less accurate           |
| SVM                      | Avg      | Slow training                    |
| Random Forest            | Avg      | Overfitting on small data        |
| XGBoost                  | Avg      | Slow, little accuracy gain       |




---

## 📚 Dataset Used

SMS Spam Collection Dataset:  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset


