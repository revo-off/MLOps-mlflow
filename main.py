from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Loading Model
MODEL_PATH = Path(__file__).parent / "regression.joblib"
model = joblib.load(MODEL_PATH)


class HouseFeatures(BaseModel):
    size: float
    nb_rooms: int
    garden: int


def make_prediction(size: float, nb_rooms: int, garden: int) -> float:
    features = pd.DataFrame([[size, nb_rooms, garden]], columns=["size", "nb_rooms", "garden"])
    prediction = model.predict(features)
    return float(prediction[0])


@app.get("/predict")
def predict_get(size: float = 100.0, nb_rooms: int = 2, garden: int = 0):
    y_pred = make_prediction(size, nb_rooms, garden)
    return {"y_pred": y_pred}


@app.post("/predict")
def predict_post(features: HouseFeatures):
    y_pred = make_prediction(features.size, features.nb_rooms, features.garden)
    return {"y_pred": y_pred}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)