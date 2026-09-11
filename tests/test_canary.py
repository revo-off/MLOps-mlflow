import sys
from collections import Counter
import requests

# Fix encodage console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "http://localhost:8000"


def main():
    print("\n" + "=" * 65)
    print("  Test de Routage Canary (/predict)")
    print("=" * 65)

    # 1. Vérification de l'état du webservice
    try:
        status_res = requests.get(f"{URL}/model-status")
        status = status_res.json()
        print("État actuel du webservice :")
        print(f"  - Modèle current : version {status['current_version']}")
        print(f"  - Modèle next    : version {status['next_version']}")
        print(f"  - Probabilité p  : {status['canary_p']} ({status['canary_p'] * 100:.0f}% current / {(1 - status['canary_p']) * 100:.0f}% next)")
        print(f"  - Canary actif   : {'OUI' if status['canary_active'] else 'NON (les deux modèles sont identiques)'}")
    except requests.exceptions.ConnectionError:
        print(f"[ERREUR] Impossible de joindre le webservice à l'adresse {URL}")
        sys.exit(1)

    # 2. Demande du nombre de requêtes à simuler
    try:
        nb_req_input = input("\nNombre de requêtes à envoyer pour tester la distribution [défaut: 50] : ").strip()
        nb_requests = int(nb_req_input) if nb_req_input else 50
    except (KeyboardInterrupt, EOFError):
        print("\nSortie.")
        sys.exit(0)

    print(f"\nEnvoi de {nb_requests} requêtes vers {URL}/predict...")

    version_counts = Counter()
    role_counts = Counter()
    payload = {"features": [[5.1, 3.5, 1.4, 0.2]]}

    for i in range(1, nb_requests + 1):
        res = requests.post(f"{URL}/predict", json=payload)
        data = res.json()
        ver = data.get("version", "inconnue")
        role = data.get("role", "inconnu")
        version_counts[ver] += 1
        role_counts[role] += 1

    print("\n" + "-" * 65)
    print("  Résultats de la répartition du trafic :")
    print("-" * 65)
    for role, count in role_counts.items():
        pct = (count / nb_requests) * 100
        print(f"  Rôle '{role:<7}' : {count:>3}/{nb_requests} requêtes ({pct:>5.1f}%)")

    print("\n  Détail par version :")
    for ver, count in version_counts.items():
        pct = (count / nb_requests) * 100
        print(f"  Version v{ver:<5} : {count:>3}/{nb_requests} requêtes ({pct:>5.1f}%)")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
