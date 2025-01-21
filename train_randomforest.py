import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the preprocessed dataset
file_path = r"D:\Downloads\UTP\FYP\generated_employee_feedback.xlsx"
df = pd.read_excel(file_path)

# Remove rows with NaN values in 'preprocessed_cons' or 'label_num'
df = df.dropna(subset=['feedback', 'label'])

# Define features and target
X = df['feedback']
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to top 5000 features for simplicity
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the trained model and the TF-IDF vectorizer
joblib.dump(model, r"D:\Downloads\UTP\FYP\sa_model_rf.pkl")
joblib.dump(tfidf_vectorizer, r"D:\Downloads\UTP\FYP\tfidf_vectorizer_sa_rf.pkl")

print("Training complete and model saved.")