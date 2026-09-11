import sys
import requests

# Fix encodage console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "http://localhost:8000"


def main():
    print("\n" + "=" * 65)
    print("  Mise à jour Canary automatique vers la version suivante")
    print("=" * 65)

    # 1. Vérification de l'état actuel
    try:
        r_status = requests.get(f"{URL}/model-status")
        status = r_status.json()
        curr_ver = status["current_version"]
        next_ver = status["next_version"]
        canary_p = status["canary_p"]
        print("État actuel des modèles :")
        print(f"  - Modèle current : version {curr_ver}")
        print(f"  - Modèle next    : version {next_ver}")
        print(f"  - Probabilité p  : {canary_p} ({canary_p*100:.0f}% current / {(1-canary_p)*100:.0f}% next)")
    except requests.exceptions.ConnectionError:
        print(f"[ERREUR] Impossible de joindre le webservice à l'adresse {URL}")
        sys.exit(1)

    # 2. Mise à jour automatique vers la version suivante (aucun choix demandé)
    print("\nRecherche et chargement de la version suivante du modèle dans MLflow...")
    try:
        r_update = requests.post(f"{URL}/update-model", json={})
        if r_update.status_code != 200:
            print(f"\n[INFO] {r_update.json().get('detail', r_update.text)}")
            print("=" * 65 + "\n")
            return
        update_data = r_update.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERREUR] Échec de la requête /update-model : {e}")
        return

    new_next = update_data["next_version"]
    print(f"✓ Succès : Modèle 'next' mis à jour vers la version {new_next}.")
    print(f"  Nouvel état : current=v{update_data['current_version']}, next=v{new_next}")
    print(f"  Trafic /predict partagé : {canary_p*100:.0f}% sur v{update_data['current_version']} / {(1-canary_p)*100:.0f}% sur v{new_next}")

    # 3. Demande de confirmation pour promouvoir le modèle 'next' en modèle 'current'
    try:
        accept_input = input(
            f"\nPromouvoir le modèle v{new_next} comme modèle courant (/accept-next-model) ? (O/n) [défaut: O] : "
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        accept_input = "n"

    if accept_input in ("o", "oui", "y", "yes", ""):
        print("\nPromotion en cours...")
        r_accept = requests.post(f"{URL}/accept-next-model")
        assert r_accept.status_code == 200, f"Erreur /accept-next-model : {r_accept.text}"
        accept_data = r_accept.json()
        print(f"✓ Succès : Modèle v{new_next} accepté !")
        print(f"  Les modèles current et next utilisent désormais tous les deux la version {accept_data['current_version']}.")
        print("  100% du trafic est désormais servi par cette version.")
    else:
        print(f"\nCanary maintenu : le modèle 'next' v{new_next} reste en test canary ({(1-canary_p)*100:.0f}% du trafic).")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
