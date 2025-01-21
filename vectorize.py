import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the preprocessed dataset
file_path = r"D:\Downloads\UTP\FYP\employee_feedback_unique_cleaned.xlsx"
df = pd.read_excel(file_path)

# Remove rows with NaN values in 'preprocessed_cons' or 'label_num'
df = df.dropna(subset=['cons', 'label_num'])

# Define features (preprocessed text) and target (labels)
X = df['cons']  # cleaned text data
y = df['label_num']  # numerical labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features for simplicity
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Save the TF-IDF vectorizer for consistency across models
import joblib
joblib.dump(tfidf_vectorizer, r"D:\Downloads\UTP\FYP\tfidf_vectorizer.pkl")