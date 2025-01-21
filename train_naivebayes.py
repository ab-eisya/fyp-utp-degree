import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the preprocessed dataset
file_path = r"D:\Downloads\UTP\FYP\generated_employee_feedback.xlsx"
df = pd.read_excel(file_path)

# Remove rows with NaN values in 'cons' or 'label_num'
df = df.dropna(subset=['feedback', 'label'])

# Define features and target
X = df['feedback']
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred_nb = nb_model.predict(X_test_tfidf)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb)}")
print(classification_report(y_test, y_pred_nb))

# Save the trained Naive Bayes model
joblib.dump(nb_model, r"D:\Downloads\UTP\FYP\sa_model_nb.pkl")

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, r"D:\Downloads\UTP\FYP\tfidf_vectorizer_sa_nb.pkl")

print("Training complete and Naive Bayes model saved.")
