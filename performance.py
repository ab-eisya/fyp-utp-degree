import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the preprocessed dataset
file_path = r"D:\Downloads\UTP\FYP\generated_employee_feedback.xlsx"
df = pd.read_excel(file_path)

# Remove rows with NaN values in 'feedback' or 'label'
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

# Define a function to evaluate a model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Train and evaluate different models
models = {
    "SVM": SVC(probability=True, kernel='linear'),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier()
}

metrics = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, X_test_tfidf, y_test)
    metrics["Model"].append(model_name)
    metrics["Accuracy"].append(accuracy)
    metrics["Precision"].append(precision)
    metrics["Recall"].append(recall)
    metrics["F1 Score"].append(f1)
    
    # Save the model
    joblib.dump(model, f"D:\\Downloads\\UTP\\FYP\\sa_model_{model_name.replace(' ', '_').lower()}.pkl")

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, r"D:\Downloads\UTP\FYP\tfidf_vectorizer_sa.pkl")

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics)

# Print the table
print(metrics_df)

# Plot the performance metrics
plt.figure(figsize=(12, 8))
x = np.arange(len(metrics_df["Model"]))
width = 0.2

plt.bar(x - 2*width, metrics_df["Accuracy"], width, label='Accuracy')
plt.bar(x - width, metrics_df["Precision"], width, label='Precision')
plt.bar(x, metrics_df["Recall"], width, label='Recall')
plt.bar(x + width, metrics_df["F1 Score"], width, label='F1 Score')

plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Model Performance Comparison')
plt.xticks(x, metrics_df["Model"])
plt.legend()

plt.show()
