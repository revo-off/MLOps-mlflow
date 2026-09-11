import os
import sys
import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Fix encodage console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Configuration MLflow
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "iris-class-exp"
client = mlflow.MlflowClient(tracking_uri)
exp = client.get_experiment_by_name(experiment_name)
if exp and exp.lifecycle_stage == "deleted":
    client.restore_experiment(exp.experiment_id)

mlflow.set_experiment(experiment_name)


def get_hyperparameters():
    """Demande interactive des hyperparamètres dans la console."""
    print("\n" + "=" * 60)
    print("  Configuration de l'entraînement RandomForest (Iris)")
    print("=" * 60)

    try:
        # Hyperparamètre principal : n_estimators
        n_est_input = input("Nombre d'arbres (n_estimators) [défaut: 100] : ").strip()
        n_estimators = int(n_est_input) if n_est_input else 100

        # Option de personnalisation avancée
        more = input("Personnaliser d'autres hyperparamètres (max_depth, etc.) ? (o/N) [défaut: N] : ").strip().lower()

        if more in ("o", "oui", "y", "yes"):
            d_input = input("  Profondeur maximale (max_depth) [défaut: None / illimitée] : ").strip()
            max_depth = int(d_input) if d_input else None

            s_input = input("  Min samples split [défaut: 2] : ").strip()
            min_samples_split = int(s_input) if s_input else 2

            l_input = input("  Min samples leaf [défaut: 1] : ").strip()
            min_samples_leaf = int(l_input) if l_input else 1

            crit_input = input("  Critère ('gini', 'entropy', 'log_loss') [défaut: gini] : ").strip().lower()
            criterion = crit_input if crit_input in ("gini", "entropy", "log_loss") else "gini"

            feat_input = input("  Max features ('sqrt', 'log2', 'all') [défaut: sqrt] : ").strip().lower()
            max_features = feat_input if feat_input in ("sqrt", "log2", "all") else "sqrt"

            test_input = input("  Taille du jeu de test (test_size) [défaut: 0.2] : ").strip()
            test_size = float(test_input) if test_input else 0.2
        else:
            max_depth = None
            min_samples_split = 2
            min_samples_leaf = 1
            criterion = "gini"
            max_features = "sqrt"
            test_size = 0.2

    except (KeyboardInterrupt, EOFError):
        print("\nUtilisation des paramètres par défaut.")
        n_estimators = 100
        max_depth = None
        min_samples_split = 2
        min_samples_leaf = 1
        criterion = "gini"
        max_features = "sqrt"
        test_size = 0.2

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "criterion": criterion,
        "max_features": max_features,
        "test_size": test_size,
        "random_state": 42,
    }


def main():
    params = get_hyperparameters()

    # Chargement des données
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )

    max_features_val = None if params["max_features"] == "all" else params["max_features"]

    print("\nEntraînement en cours...")
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        criterion=params["criterion"],
        max_features=max_features_val,
        random_state=params["random_state"],
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    # Enregistrement dans MLflow
    with mlflow.start_run():
        mlflow.log_params(
            {
                "n_estimators": params["n_estimators"],
                "max_depth": str(params["max_depth"]),
                "min_samples_split": params["min_samples_split"],
                "min_samples_leaf": params["min_samples_leaf"],
                "criterion": params["criterion"],
                "max_features": params["max_features"],
                "test_size": params["test_size"],
                "random_state": params["random_state"],
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": test_acc,
                "train_accuracy": train_acc,
            }
        )
        mlflow.sklearn.log_model(model, name="model", registered_model_name="iris-model")

    print("\n" + "=" * 60)
    print("  Résultats de l'entraînement")
    print("=" * 60)
    print(f"  Hyperparamètres : n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
          f"min_samples_split={params['min_samples_split']}, min_samples_leaf={params['min_samples_leaf']}, "
          f"criterion='{params['criterion']}'")
    print(f"  Métriques       : test_accuracy={test_acc:.4f}, train_accuracy={train_acc:.4f}")
    print("  Enregistré dans : MLflow Model Registry ('iris-model')")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
