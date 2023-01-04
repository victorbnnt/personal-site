from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from NYCTaxiFare.predict import get_model
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
    return "Root - NYC Taxi Fare prediction"

# define a root `/` endpoint
@app.get("/predict_fare")
def predict_fare(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):
    model = get_model("model")
    X_pred = pd.DataFrame({
            "pickup_datetime": [pickup_datetime],
            "pickup_longitude": [float(pickup_longitude)],
            "pickup_latitude": [float(pickup_latitude)],
            "dropoff_longitude": [float(dropoff_longitude)],
            "dropoff_latitude": [float(dropoff_latitude)],
            "passenger_count": [int(passenger_count)]
        })

    y_pred = model.predict(X_pred)
    return {
             "FarePrediction": y_pred[0]
           }
