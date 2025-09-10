# -*- coding: utf-8 -*-
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Downloads the necessary NLTK resources.
# The download function automatically checks if the resources already exist.
print("Downloading necessary NLTK resources...")
nltk.download('stopwords')
print("Stopwords download completed.")

# 1. Load the data
# Assuming the 'spam_ham_dataset.csv' file is in the same directory
try:
    df = pd.read_csv('spam_ham_dataset.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'spam_ham_dataset.csv' was not found.")
    print("Please make sure the file is in the same directory as the script.")
    exit()

# Display basic information about the dataset
print("\nDataset information:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())

# 2. Text preprocessing
# Converts the text label ('spam'/'ham') to numeric (1/0)
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

def preprocess_text(text):
    """
    Function to clean and preprocess email text.
    - Converts to lowercase
    - Removes special characters and numbers
    - Removes stopwords (common words like 'a', 'an', 'the')
    - Applies stemming (reduces the word to its root, e.g., 'running' -> 'run')
    """
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
        return ' '.join(words)
    else:
        return ""

# Applies the preprocessing function to the 'text' column
df['preprocessed_text'] = df['text'].apply(preprocess_text)

print("\nData after preprocessing:")
print(df[['text', 'preprocessed_text', 'label_num']].head())

# 3. Text vectorization using TF-IDF
# TF-IDF converts text into numerical vectors of word importance
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['preprocessed_text'])
y = df['label_num']

print("\nTF-IDF matrix dimensions (samples x words):", X.shape)

# 4. Splitting the data into training and testing sets
# 75% for training and 25% for testing to evaluate performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("\nTraining set: {} samples".format(X_train.shape[0]))
print("Testing set: {} samples".format(X_test.shape[0]))

# 5. Model training
# We use the Naive Bayes classifier, which is very effective for text classification tasks
model = MultinomialNB()
print("\nTraining the model...")
model.fit(X_train, y_train)
print("Training completed!")

# 6. Model evaluation
y_pred = model.predict(X_test)

print("\n--- Model Evaluation Report ---")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Saving the model and vectorizer
# This allows you to use the trained model without needing to retrain it
joblib.dump(model, 'spam_detector_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\nModel and vectorizer saved as 'spam_detector_model.pkl' and 'tfidf_vectorizer.pkl'.")

# 8. Prediction function for new emails
def predict_new_email(email_text):
    """
    Function to predict whether a new email is spam or not.
    Loads the saved model and vectorizer.
    """
    try:
        loaded_model = joblib.load('spam_detector_model.pkl')
        loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except FileNotFoundError:
        print("Error: Model and vectorizer files not found. Please run the script first.")
        return

    # Preprocesses the new email
    preprocessed_email = preprocess_text(email_text)
    
    # Vectorizes the text using the trained vectorizer
    vectorized_email = loaded_vectorizer.transform([preprocessed_email])
    
    # Makes the prediction
    prediction = loaded_model.predict(vectorized_email)[0]
    
    # Returns the result
    return "SPAM" if prediction == 1 else "Not SPAM"

# Testing prediction on new emails
print("\n--- Testing prediction on new emails ---")
test_email1 = "Congratulations! You've won a free prize. Click the link to claim it now."
print(f"The email '{test_email1}' is classified as: {predict_new_email(test_email1)}")

test_email2 = "Hi team, please find the updated report attached to this email. Thanks."
print(f"The email '{test_email2}' is classified as: {predict_new_email(test_email2)}")

test_email3 = "URGENT! Your account has been compromised. Log in immediately to secure it."
print(f"The email '{test_email3}' is classified as: {predict_new_email(test_email3)}")
