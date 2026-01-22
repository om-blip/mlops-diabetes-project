import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model
model = joblib.load("models/model.pkl")

app = FastAPI(title="Diabetes Prediction API")

# Input schema
class DiabetesInput(BaseModel):
    HighBP: float
    HighChol: float
    CholCheck: float
    BMI: float
    Smoker: float
    Stroke: float
    HeartDiseaseorAttack: float
    PhysActivity: float
    Fruits: float
    Veggies: float
    HvyAlcoholConsump: float
    AnyHealthcare: float
    NoDocbcCost: float
    GenHlth: float
    MentHlth: float
    PhysHlth: float
    DiffWalk: float
    Sex: float
    Age: float
    Education: float
    Income: float


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}


@app.post("/predict")
def predict(data: DiabetesInput):

    input_data = np.array([[ 
        data.HighBP,
        data.HighChol,
        data.CholCheck,
        data.BMI,
        data.Smoker,
        data.Stroke,
        data.HeartDiseaseorAttack,
        data.PhysActivity,
        data.Fruits,
        data.Veggies,
        data.HvyAlcoholConsump,
        data.AnyHealthcare,
        data.NoDocbcCost,
        data.GenHlth,
        data.MentHlth,
        data.PhysHlth,
        data.DiffWalk,
        data.Sex,
        data.Age,
        data.Education,
        data.Income
    ]])

    prediction = model.predict(input_data)[0]

    label_map = {
        0: "No Diabetes",
        1: "Prediabetes",
        2: "Diabetes"
    }

    return {
        "prediction_class": int(prediction),
        "prediction_label": label_map[int(prediction)]
    }
