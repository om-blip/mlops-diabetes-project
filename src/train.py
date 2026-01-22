import pandas as pd
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("data/raw/diabetes.csv")

X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==========================
# APPLY SMOTE (ONLY TRAIN)
# ==========================
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_resampled).value_counts())

# ==========================
# Train model
# ==========================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = model.predict(X_test)

# Metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_macro": f1_score(y_test, y_pred, average="macro"),
    "f1_weighted": f1_score(y_test, y_pred, average="weighted")
}

# Save metrics
os.makedirs("metrics", exist_ok=True)
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Training complete with SMOTE")
print(metrics)
