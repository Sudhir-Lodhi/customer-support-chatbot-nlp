import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from preprocess import preprocess_text


# Load training dataset
train_data = pd.read_csv(
    "dataset/banking77/Bitext_Sample_Customer_Service_Training_Dataset.csv"
)

# Get customer messages and intents
X_train = train_data["utterance"]
y_train = train_data["intent"]

# Preprocess customer messages
X_train_clean = X_train.apply(preprocess_text)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train_clean)

# Create and train Naive Bayes model
model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

# Save the trained model
joblib.dump(model, "models/intent_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model training completed successfully!")
print("Number of training examples:", len(X_train))
print("Number of intents:", y_train.nunique())
print("Model saved successfully.")