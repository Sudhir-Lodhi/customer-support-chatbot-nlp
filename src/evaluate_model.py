import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report

from preprocess import preprocess_text


# Load testing dataset
test_data = pd.read_csv(
    "dataset/banking77/Bitext_Sample_Customer_Service_Testing_Dataset.csv"
)

# Load trained model and TF-IDF vectorizer
model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Get test messages and actual intents
X_test = test_data["utterance"]
y_test = test_data["intent"]

# Preprocess test messages
X_test_clean = X_test.apply(preprocess_text)

# Convert test messages into TF-IDF features
X_test_tfidf = vectorizer.transform(X_test_clean)

# Predict intents
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("===== MODEL EVALUATION =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

# Classification report
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))