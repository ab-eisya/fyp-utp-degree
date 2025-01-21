import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained Logistic Regression model and TF-IDF vectorizer
model = joblib.load(r"D:\Downloads\UTP\FYP\logistic_regression_model.pkl")
tfidf_vectorizer = joblib.load(r"D:\Downloads\UTP\FYP\tfidf_vectorizer_lr.pkl")

# Load the English language model for preprocessing
nlp = spacy.load("en_core_web_sm")

# Define the label mapping
label_mapping = {
    0: 'High Workload',
    1: 'Lack of Autonomy',
    2: 'Poor Location',
    3: 'Poor Work-Life Balance',
    4: 'Poor Career Growth',
    5: 'Poor Work Environment',
    6: 'Poor Job Security',
    7: 'Lack of Benefits',
    8: 'Poor Management System',
    9: 'Poor Work Culture',
    10: 'Poor Salary',
    11: 'Poor Appraisal System',
    12: 'Poor Leadership'
}

def preprocess(text):
    """Preprocess the input text: lemmatization, removing stop words and punctuation."""
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(filtered_tokens)

def predict_dissatisfaction_factor(user_input):
    """Predict the dissatisfaction factor from user input."""
    preprocessed_input = preprocess(user_input)
    features = tfidf_vectorizer.transform([preprocessed_input])
    predicted_label_num = model.predict(features)[0]
    predicted_label = label_mapping.get(predicted_label_num, "Unknown")
    
    return predicted_label

def main():
    while True:
        user_input = input("Enter employee feedback (or type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        
        predicted_label = predict_dissatisfaction_factor(user_input)
        
        print(f"The predicted dissatisfaction factor is: {predicted_label}")
        print("\n")

if __name__ == "__main__":
    main()
