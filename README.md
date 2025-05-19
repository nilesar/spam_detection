📩 Email/SMS Spam Classifier
A simple and efficient Streamlit web application that detects whether a given text message is Spam or Not Spam, using Natural Language Processing (NLP) and a machine learning model trained on real-world SMS/email data.

🚀 Demo
<!-- (Optional: Replace with your screenshot link or remove this section) -->

🧠 Features
Text preprocessing with tokenization, stemming, punctuation removal, and stopword filtering.

Built using multiple machine learning algorithms including:

Multinomial Naive Bayes (best performer)

Logistic Regression

Support Vector Machine

Random Forest

XGBoost

Integrated with TF-IDF Vectorization.

Real-time prediction on user input using a trained model and Streamlit interface.

📦 Project Structure
bash
Copy
Edit
.
├── app.py                # Streamlit app code
├── model.pkl             # Trained machine learning model (MNB)
├── vectorizer.pkl        # TF-IDF vectorizer used for feature extraction
├── README.md             # Project documentation
├── requirements.txt      # Required dependencies
└── data/                 # (Optional) Dataset used for training
📋 Requirements
Install the required Python packages using:

bash
Copy
Edit
pip install -r requirements.txt
Your requirements.txt should include:

txt
Copy
Edit
streamlit
nltk
scikit-learn
xgboost
pandas
matplotlib
seaborn
🧪 How It Works
Preprocessing
Raw input is cleaned by:

Converting to lowercase

Removing non-alphanumeric characters

Removing English stopwords

Applying stemming (PorterStemmer)

Vectorization
Processed text is transformed into numerical format using TF-IDF Vectorizer.

Prediction
The vector is passed to a trained Multinomial Naive Bayes model for classification.

Output
The model returns whether the message is Spam or Not Spam, displayed on the UI.

📈 Model Selection
Several algorithms were tested:

✅ Multinomial Naive Bayes – Best balance of performance and simplicity

❌ Logistic Regression – Good, but slightly less accurate

❌ SVM – High training time with marginal improvement

❌ Random Forest – Prone to overfitting

❌ XGBoost – Complex and slower without much accuracy gain

🖥️ Run the App Locally
bash
Copy
Edit
streamlit run app.py
Then go to http://localhost:8501 in your browser to try out the app.

📚 Dataset
You can use datasets such as the SMS Spam Collection Dataset for training.

✨ Future Improvements
Add support for other languages

Deploy via Streamlit Cloud or Heroku

Improve preprocessing with lemmatization

Add visualization for prediction confidence

📜 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Nilendu Adhikary
GitHub • LinkedIn

