from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from RainInAustralia.load import get_model
import os
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return "Root - Rain in Australia prediction"

# define a root `/` endpoint
@app.get("/predict_rain")
def predict_rain(Humidity3pm, WindGustSpeed, Location, Pressure9am, MinTemp):
    model = get_model("model")
    X_pred = pd.DataFrame({
            "Humidity3pm": [float(Humidity3pm)],
            "WindGustSpeed": [float(WindGustSpeed)],
            "Location": [Location],
            "Pressure9am": [float(Pressure9am)],
            "MinTemp": [float(MinTemp)]
        })

    y_pred = model.best_estimator_.predict(X_pred).tolist()
    y_pred_proba = model.best_estimator_.predict_proba(X_pred).tolist()
    return {
             "RainProba": round(float(y_pred_proba[0][1]), 3),
             "RainPrediction": int(y_pred[0])
           }
