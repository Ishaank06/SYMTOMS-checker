import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# -----------------------
# 1. Load the Dataset
# -----------------------
# Use the correct path based on your folder structure
file_path = "datasets/survey_lung_cancer.csv"
data = pd.read_csv(file_path)

# -----------------------
# 2. Data Preprocessing
# -----------------------
# Map 'GENDER' column to numerical values: M -> 1, F -> 0
data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})

# Map target column 'LUNG_CANCER' to numerical values: YES -> 1, NO -> 0
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Handle missing values (if any) by filling with median
data.fillna(data.median(numeric_only=True), inplace=True)

# Features and target separation
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# 3. Model Training
# -----------------------
print("\nTraining Lung Cancer Model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------
# 4. Model Evaluation
# -----------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------
# 5. Save the Model
# -----------------------
# Save the model in the "models" directory
output_path = "models/lung_cancer_model.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists

joblib.dump(model, output_path)

print(f"\n✅ Model saved as {output_path}")
