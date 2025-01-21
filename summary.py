import pandas as pd
from transformers import pipeline
import joblib
import spacy
import re

# Load the SVM model and TF-IDF vectorizer for sentiment analysis
sa_model = joblib.load(r"D:\Downloads\UTP\FYP\sa_model.pkl")
tfidf_vectorizer_sa = joblib.load(r"D:\Downloads\UTP\FYP\tfidf_vectorizer_sa.pkl")

# Load the Excel file
file_path = r'D:\Downloads\UTP\FYP\indeed_employee_feedbacks.xlsx'
df = pd.read_excel(file_path)

# Fill NaN values in the 'feedback' column with empty strings
df['feedback'] = df['feedback'].fillna('')

# Load the English language model for preprocessing
nlp = spacy.load("en_core_web_sm")

# Load the T5 summarization model
summarizer = pipeline("summarization", model="t5-small")

def preprocess(text):
    """Preprocess the input text: lemmatization, removing stop words and punctuation."""
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(filtered_tokens)

def analyze_sentiment(feedback):
    """Analyze the sentiment of the feedback using SVM model."""
    preprocessed_input = preprocess(feedback)
    features = tfidf_vectorizer_sa.transform([preprocessed_input])
    predicted_label_num = sa_model.predict(features)[0]
    sentiment = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}[predicted_label_num]
    return sentiment

# Apply the sentiment analysis function to the feedback column
df['sentiment'] = df['feedback'].apply(analyze_sentiment)

# Separate feedback by sentiment
positive_feedback = " ".join(df[df['sentiment'] == 'Positive']['feedback'].tolist())
negative_feedback = " ".join(df[df['sentiment'] == 'Negative']['feedback'].tolist())
neutral_feedback = " ".join(df[df['sentiment'] == 'Neutral']['feedback'].tolist())

def split_text_into_chunks(text, max_chunk_size=500):
    """Split text into chunks of max_chunk_size while maintaining sentence boundaries."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def summarize_text(text, max_length=20, min_length=10):
    """Summarize the text in chunks and combine the summaries."""
    chunks = split_text_into_chunks(text)
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)

def clean_summary(summary):
    """Ensure sentences start with a capital letter and proper punctuation."""
    sentences = re.split(r'(?<=[.!?]) +', summary.strip())
    cleaned_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            sentence = ' '.join([word if i == 0 or word.isupper() else word.lower() for i, word in enumerate(sentence.split())])
            cleaned_sentences.append(sentence)

    cleaned_summary = '. '.join(cleaned_sentences)
    if not cleaned_summary.endswith('.'):
        cleaned_summary += '.'
    return cleaned_summary

# Summarize the feedback for each sentiment category
positive_summary = clean_summary(summarize_text(positive_feedback))
negative_summary = clean_summary(summarize_text(negative_feedback))
neutral_summary = clean_summary(summarize_text(neutral_feedback))

# Output the summaries
print("Positive Summary:\n", positive_summary)
print("Negative Summary:\n", negative_summary)
print("Neutral Summary:\n", neutral_summary)
