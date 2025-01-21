import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred_lr = lr_model.predict(X_test_tfidf)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}")
print(classification_report(y_test, y_pred_lr))

# Save the trained Logistic Regression model
joblib.dump(lr_model, r"D:\Downloads\UTP\FYP\sa_model_lr.pkl")

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, r"D:\Downloads\UTP\FYP\tfidf_vectorizer_sa_lr.pkl")

print("Training complete and Logistic Regression model saved.")
