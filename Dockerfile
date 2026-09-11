FROM python:3.11-slim

WORKDIR /app

# Copie et installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source et du modèle
COPY . .

# Port exposé par le conteneur
EXPOSE 8000

# Lancement de l'API avec Uvicorn sur 0.0.0.0 pour être accessible hors du conteneur
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
