from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import Config

# ── Choix du modèle ───────────────────────────────────────────────────────────
#
#  On propose trois classifieurs. RandomForest est le défaut :
#  rapide à entraîner, robuste, et donne de bonnes probabilités via predict_proba.
#
#  SVM (RBF) est souvent le meilleur sur peu de données mais plus lent.
#  GradientBoosting est plus précis mais nettement plus lent à entraîner.
# ─────────────────────────────────────────────────────────────────────────────


class Classification:
    def __init__(self, model_name: str = "random_forest"):
        self.pipeline = None
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=Config.RANDOM_STATE,
                n_jobs=-1,
            ),
            "svm": SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=Config.RANDOM_STATE,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=Config.RANDOM_STATE,
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=5,
                algorithm="auto",
                leaf_size=30,
                metric="euclidean",
                n_jobs=-1,
            ),
        }

        self.build_pipeline(model_name)

    def build_pipeline(self, model_name: str) -> None:
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", self.models[model_name]),
            ]
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Entraîne le pipeline et retourne le pipeline fitted."""
        self.pipeline.fit(X_train, y_train)

    @staticmethod
    def save_model(pipeline: Pipeline, name: str = "model") -> None:
        path = Config.PATH_MODEL + f"{name}.joblib"
        joblib.dump(pipeline, path)

    @staticmethod
    def load_model(name: str) -> Pipeline:
        path = Config.PATH_MODEL + f"{name}.joblib"
        return joblib.load(path)
