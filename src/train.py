import pandas as pd
import mlflow
import mlflow.sklearn
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load data
# =========================
df = pd.read_csv("data/raw/diabetes.csv")

X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# Model
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

mlflow.set_experiment("Diabetes_MLOps_Pipeline")

with mlflow.start_run():

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    os.makedirs("models", exist_ok=True)

    model_path = "models/model.pkl"
    joblib.dump(model, model_path)

    mlflow.log_artifact(model_path)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    f1_weighted = f1_score(y_test, preds, average="weighted")

    # Save metrics for DVC
    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()

    # Log to MLflow
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("artifacts/confusion_matrix.png")
    mlflow.sklearn.log_model(model, "model")

print("Training complete.")
