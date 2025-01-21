import pandas as pd
import spacy

# Load the cleaned dataset
file_path = r"D:\Downloads\UTP\FYP\employee_feedback_unique.xlsx"
df = pd.read_excel(file_path)

# Remove rows with NaN values in the 'cons' column
df.dropna(subset=['cons'], inplace=True)

# Load the English language model and create an nlp object from it
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    # Check if the input is a string, if not return an empty string
    if not isinstance(text, str):
        return ""
    
    # Remove stop words and lemmatize the text
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    return " ".join(filtered_tokens)

# Preprocess the 'cons' column and store the result in 'preprocessed_cons'
df['preprocessed_cons'] = df['cons'].apply(preprocess)

# Example output: Display the original and preprocessed 'cons' columns
print(df[['cons', 'preprocessed_cons']].head())

# Save the preprocessed dataframe to a new Excel file
df.to_excel(r"D:\Downloads\UTP\FYP\employee_feedback_unique_cleaned.xlsx", index=False)