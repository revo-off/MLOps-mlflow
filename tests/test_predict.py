import sys
import requests

# Fix encodage console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "http://localhost:8000"
DEFAULT_FEATURES = [5.1, 3.5, 1.4, 0.2]


def main():
    print("\n" + "=" * 60)
    print("  Test de prédiction sur le Webservice FastAPI")
    print("=" * 60)

    try:
        user_input = input(
            f"Entrez 4 caractéristiques séparées par des virgules [défaut: 5.1, 3.5, 1.4, 0.2] : "
        ).strip()
        if user_input:
            features = [[float(x.strip()) for x in user_input.split(",")]]
        else:
            features = [DEFAULT_FEATURES]
    except (KeyboardInterrupt, EOFError):
        print("\nUtilisation des valeurs par défaut.")
        features = [DEFAULT_FEATURES]
    except ValueError:
        print("\nFormat invalide. Utilisation des valeurs par défaut.")
        features = [DEFAULT_FEATURES]

    print(f"\nEnvoi de la requête /predict avec les données : {features}")

    try:
        response = requests.post(f"{URL}/predict", json={"features": features})
        assert response.status_code == 200, f"Erreur {response.status_code}: {response.text}"
        data = response.json()
        assert "predictions" in data, "Réponse invalide : clé 'predictions' absente"

        print(f"\n✓ Réponse reçue :")
        print(f"  - Prédiction(s) : {data.get('predictions')}")
        print(f"  - Modèle utilisé: version {data.get('version')}")
        print("=" * 60 + "\n")
    except requests.exceptions.ConnectionError:
        print(f"\n[ERREUR] Impossible de joindre le webservice à l'adresse {URL}")
        print("Vérifiez que le conteneur Docker est actif : 'docker compose up -d'\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
