# from transformers import BertTokenizer, BertForSequenceClassification
# import torch

# # Load pre-trained BERT model and tokenizer
# model_name = 'bert-base-uncased'  # Example of a pre-trained BERT model
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Assuming 3 classes: Positive, Negative, Neutral

# # Function for sentiment prediction
# def predict_sentiment(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     outputs = model(**inputs)
#     logits = outputs.logits
#     prediction = torch.argmax(logits, dim=1).item()
#     sentiment = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}[prediction]
#     return sentiment

# # User input for prediction
# while True:
#     user_input = input("Enter employee feedback (or type 'exit' to quit): ")
#     if user_input.lower() == 'exit':
#         break
#     predicted_sentiment = predict_sentiment(user_input)
#     print(f"Predicted Sentiment: {predicted_sentiment}")

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pandas as pd

# Load the trained SVM model
svm_model = joblib.load(r"D:\Downloads\UTP\FYP\sa_model.pkl")

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load(r"D:\Downloads\UTP\FYP\tfidf_vectorizer_sa.pkl")

def preprocess_text(text):
    """Preprocess the input text using the same TF-IDF vectorizer."""
    return tfidf_vectorizer.transform([text])

def predict_sentiment(text):
    """Predict the sentiment polarity (label) of the input text."""
    preprocessed_text = preprocess_text(text)
    prediction = svm_model.predict(preprocessed_text)[0]
    return prediction

# User input for prediction
while True:
    user_input = input("Enter employee feedback (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    prediction = predict_sentiment(user_input)
    sentiment = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}[prediction]
    print(f"Sentiment: {sentiment}")
