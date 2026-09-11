import os
import random
from contextlib import asynccontextmanager
import mlflow
from mlflow.client import MlflowClient
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

FEATURE_NAMES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Modèles et versions en mémoire
current_model = None
current_version = "1"
next_model = None
next_version = "1"

# Probabilité d'utiliser current_model (p) vs next_model (1 - p)
canary_p = float(os.getenv("CANARY_P", "0.8"))


class PredictRequest(BaseModel):
    features: list[list[float]]


class UpdateModelRequest(BaseModel):
    version: str | None = None


class SetCanaryPRequest(BaseModel):
    p: float = Field(..., ge=0.0, le=1.0, description="Probabilité de routage vers le modèle actuel (0.0 à 1.0)")


def get_next_version_to_load() -> str:
    """
    Détermine automatiquement la prochaine version du modèle à charger.
    Cherche la version suivante enregistrée dans MLflow (> current_version).
    Si aucune version supérieure n'existe, tente (current_version + 1).
    """
    global current_version
    try:
        client = MlflowClient(TRACKING_URI)
        mvs = client.search_model_versions("name='iris-model'")
        available_versions = sorted([int(v.version) for v in mvs if str(v.version).isdigit()])
        curr_int = int(current_version) if str(current_version).isdigit() else 1

        # Cherche la plus petite version supérieure à l'actuelle
        higher_versions = [v for v in available_versions if v > curr_int]
        if higher_versions:
            return str(min(higher_versions))

        # Si aucune version supérieure, tente le numéro suivant
        return str(curr_int + 1)
    except Exception:
        try:
            return str(int(current_version) + 1)
        except ValueError:
            return "2"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global current_model, next_model, current_version, next_version
    mlflow.set_tracking_uri(TRACKING_URI)

    # Au démarrage, current_model et next_model sont identiques (version 1)
    print(f"Chargement initial du modèle version {current_version}...")
    loaded_model = mlflow.pyfunc.load_model(f"models:/iris-model/{current_version}")
    current_model = loaded_model
    next_model = loaded_model
    next_version = current_version
    print(f"Démarrage terminé : current_version={current_version}, next_version={next_version}, canary_p={canary_p}")
    yield


app = FastAPI(
    title="MLOps Canary Deployment Webservice",
    description="Webservice FastAPI avec routage de trafic Canary (modèle current et next)",
    lifespan=lifespan,
)


@app.get("/model-status")
def get_model_status():
    """Retourne l'état actuel des modèles et la configuration canary."""
    return {
        "current_version": current_version,
        "next_version": next_version,
        "canary_p": canary_p,
        "canary_active": current_version != next_version,
    }


@app.post("/predict")
def predict(data: PredictRequest):
    """
    Effectue une prédiction en routant le trafic avec probabilité p vers current
    et probabilité (1 - p) vers next.
    """
    global current_model, next_model, current_version, next_version, canary_p

    # Tirage aléatoire pour le routage canary
    r = random.random()
    if r < canary_p:
        chosen_model = current_model
        chosen_version = current_version
        chosen_role = "current"
    else:
        chosen_model = next_model
        chosen_version = next_version
        chosen_role = "next"

    df = pd.DataFrame(data.features, columns=FEATURE_NAMES)
    preds = chosen_model.predict(df)

    return {
        "predictions": preds.tolist(),
        "version": chosen_version,
        "role": chosen_role,
    }


@app.post("/update-model")
def update_model(data: UpdateModelRequest | None = None):
    """
    Met à jour le modèle 'next' avec la version suivante automatiquement
    (ou avec la version spécifiée si fournie).
    """
    global next_model, next_version

    if data and data.version:
        target_version = data.version
    else:
        target_version = get_next_version_to_load()

    try:
        loaded_model = mlflow.pyfunc.load_model(f"models:/iris-model/{target_version}")
        next_model = loaded_model
        next_version = target_version
        return {
            "status": "success",
            "message": f"Modèle next mis à jour automatiquement vers la version {next_version}.",
            "current_version": current_version,
            "next_version": next_version,
            "canary_p": canary_p,
        }
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Impossible de charger la version '{target_version}' depuis MLflow : {str(e)}. "
                f"Entraînez d'abord une nouvelle version avec 'py train.py'."
            ),
        )


@app.post("/accept-next-model")
def accept_next_model():
    """
    Valide et promeut 'next_model' comme nouveau 'current_model'.
    Désormais, current et next utilisent tous les deux la nouvelle version.
    """
    global current_model, current_version, next_model, next_version

    current_model = next_model
    current_version = next_version

    return {
        "status": "success",
        "message": f"Modèle v{next_version} accepté et promu comme modèle courant.",
        "current_version": current_version,
        "next_version": next_version,
    }


@app.post("/set-canary-p")
def set_canary_p(data: SetCanaryPRequest):
    """Permet d'ajuster dynamiquement la probabilité de routage p."""
    global canary_p
    canary_p = data.p
    return {"status": "success", "canary_p": canary_p}