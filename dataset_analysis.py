import pandas as pd

# Load datasets
train_data = pd.read_csv(
    "dataset/banking77/Bitext_Sample_Customer_Service_Training_Dataset.csv"
)

test_data = pd.read_csv(
    "dataset/banking77/Bitext_Sample_Customer_Service_Testing_Dataset.csv"
)

validation_data = pd.read_csv(
    "dataset/banking77/Bitext_Sample_Customer_Service_Validation_Dataset.csv"
)

# Display basic information
print("===== DATASET SIZE =====")
print("Training data:", train_data.shape)
print("Testing data:", test_data.shape)
print("Validation data:", validation_data.shape)

# Display column names
print("\n===== COLUMNS =====")
print(train_data.columns.tolist())

# Display first 5 rows
print("\n===== FIRST 5 TRAINING RECORDS =====")
print(train_data.head())

# Display number of intents
print("\n===== NUMBER OF INTENTS =====")
print(train_data["intent"].nunique())

# Display all intents
print("\n===== INTENTS =====")
print(train_data["intent"].unique())

# Display number of examples for each intent
print("\n===== INTENT DISTRIBUTION =====")
print(train_data["intent"].value_counts())

# Check missing values
print("\n===== MISSING VALUES =====")
print(train_data.isnull().sum())