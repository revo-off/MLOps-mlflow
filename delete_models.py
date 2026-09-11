#!/usr/bin/env python3
"""
Script interactif pour supprimer des modèles MLflow du Model Registry.
Toutes les actions sont guidées via des questions dans la console.
"""

import os
import sys
from datetime import datetime

# Fix encodage console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Supprimer le bandeau hint mlflow
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def format_timestamp(ts):
    """Formate un timestamp millisecondes en date lisible."""
    if not ts:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def get_client() -> MlflowClient:
    """Initialise le client MLflow et vérifie la connectivité."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(TRACKING_URI)
    try:
        client.search_registered_models(max_results=1)
    except Exception as e:
        print(f"\n[ERREUR] Impossible de se connecter au serveur MLflow ({TRACKING_URI}) :")
        print(f"  {e}")
        print("\nVérifiez que le service Docker est bien actif : 'docker compose up -d'\n")
        sys.exit(1)
    return client


def get_all_versions(client: MlflowClient, model_name: str):
    """Récupère toutes les versions d'un modèle enregistré."""
    try:
        return client.search_model_versions(f"name='{model_name}'")
    except Exception:
        m = client.get_registered_model(model_name)
        return m.latest_versions or []


def display_registered_models(client: MlflowClient):
    """Affiche la liste actuelle des modèles et versions."""
    models = client.search_registered_models()
    if not models:
        print("\nAucun modèle enregistré dans MLflow Model Registry.\n")
        return []

    print("\n" + "=" * 70)
    print(f"  Modèles enregistrés dans MLflow ({len(models)} au total) :")
    print("=" * 70)
    for idx, m in enumerate(models, start=1):
        versions = get_all_versions(client, m.name)
        ver_nums = [str(v.version) for v in sorted(versions, key=lambda x: int(x.version) if str(x.version).isdigit() else str(x.version))]
        ver_display = ", ".join(f"v{v}" for v in ver_nums) if ver_nums else "aucune version"
        print(f"  [{idx}] Modèle : {m.name} (versions disponibles : {ver_display})")
    print("=" * 70)
    return models


def delete_single_version(client: MlflowClient, model_name: str, version: str, delete_runs: bool):
    """Supprime une seule version d'un modèle."""
    try:
        mv = client.get_model_version(model_name, str(version))
        run_id = mv.run_id
    except MlflowException as e:
        print(f"[ERREUR] Version {version} du modèle '{model_name}' introuvable : {e}")
        return

    try:
        client.delete_model_version(model_name, str(version))
        print(f"  ✓ Version v{version} du modèle '{model_name}' supprimée.")
    except Exception as e:
        print(f"  ✗ Échec de la suppression de la version : {e}")
        return

    if delete_runs and run_id:
        try:
            client.delete_run(run_id)
            print(f"  ✓ Run associé '{run_id}' supprimé.")
        except Exception as e:
            print(f"  ✗ Impossible de supprimer le run associé : {e}")


def delete_full_model(client: MlflowClient, model_name: str, delete_runs: bool):
    """Supprime un modèle et toutes ses versions (réinitialise le compteur à v1)."""
    versions = get_all_versions(client, model_name)
    run_ids = set()

    for v in versions:
        if v.run_id:
            run_ids.add(v.run_id)
        try:
            client.delete_model_version(model_name, str(v.version))
            print(f"  ✓ Version v{v.version} supprimée.")
        except Exception:
            pass

    try:
        client.delete_registered_model(model_name)
        print(f"  ✓ Entrée du modèle '{model_name}' supprimée du registre (le prochain entraînement démarrera à v1).")
    except Exception as e:
        print(f"  ✗ Erreur lors de la suppression du modèle : {e}")
        return

    if delete_runs and run_ids:
        for rid in run_ids:
            try:
                client.delete_run(rid)
                print(f"  ✓ Run d'entraînement '{rid}' supprimé.")
            except Exception:
                pass


def main():
    client = get_client()

    while True:
        models = display_registered_models(client)
        if not models:
            break

        print("\nOptions de suppression :")
        print("  [1] Supprimer un modèle complet (et toutes ses versions -> reset à v1)")
        print("  [2] Supprimer une version spécifique")
        print("  [3] Supprimer TOUS les modèles enregistrés")
        print("  [q] Quitter")

        try:
            choice = input("\nVotre choix (1/2/3/q) [défaut: q] : ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nSortie.")
            break

        if choice in ("q", ""):
            print("Au revoir !")
            break

        # Option 1 : Supprimer un modèle complet
        if choice == "1":
            default_model = models[0].name
            m_input = input(f"Nom du modèle à supprimer [défaut: {default_model}] : ").strip()
            model_name = m_input if m_input else default_model

            del_runs = input("Supprimer aussi les runs d'expériences associés ? (o/N) [défaut: N] : ").strip().lower() in ("o", "oui", "y", "yes")
            confirm = input(f"Confirmer la suppression du modèle '{model_name}' ? (o/N) : ").strip().lower()

            if confirm in ("o", "oui", "y", "yes"):
                delete_full_model(client, model_name, delete_runs=del_runs)
            else:
                print("Suppression annulée.")

        # Option 2 : Supprimer une version spécifique
        elif choice == "2":
            default_model = models[0].name
            m_input = input(f"Nom du modèle [défaut: {default_model}] : ").strip()
            model_name = m_input if m_input else default_model

            versions = get_all_versions(client, model_name)
            if not versions:
                print(f"Aucune version disponible pour '{model_name}'.")
                continue

            available_versions = [str(v.version) for v in versions]
            default_ver = available_versions[-1]
            v_input = input(f"Numéro de version à supprimer ({', '.join(available_versions)}) [défaut: {default_ver}] : ").strip()
            version_to_delete = v_input if v_input else default_ver

            del_runs = input("Supprimer aussi le run d'entraînement associé ? (o/N) [défaut: N] : ").strip().lower() in ("o", "oui", "y", "yes")
            confirm = input(f"Confirmer la suppression de la version v{version_to_delete} de '{model_name}' ? (o/N) : ").strip().lower()

            if confirm in ("o", "oui", "y", "yes"):
                delete_single_version(client, model_name, version_to_delete, delete_runs=del_runs)
            else:
                print("Suppression annulée.")

        # Option 3 : Supprimer tous les modèles
        elif choice == "3":
            del_runs = input("Supprimer aussi les runs d'expériences associés ? (o/N) [défaut: N] : ").strip().lower() in ("o", "oui", "y", "yes")
            confirm = input("⚠️  Confirmer la suppression de TOUS les modèles ? (o/N) : ").strip().lower()

            if confirm in ("o", "oui", "y", "yes"):
                for m in models:
                    delete_full_model(client, m.name, delete_runs=del_runs)
                print("\nTous les modèles ont été supprimés.")
            else:
                print("Suppression annulée.")
        else:
            print("Choix invalide.")


if __name__ == "__main__":
    main()
