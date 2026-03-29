from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import Config
from src.utils.audio import AudioUtils

# ── Choix du modèle ───────────────────────────────────────────────────────────
#
#  On propose trois classifieurs. RandomForest est le défaut :
#  rapide à entraîner, robuste, et donne de bonnes probabilités via predict_proba.
#
#  SVM (RBF) est souvent le meilleur sur peu de données mais plus lent.
#  GradientBoosting est plus précis mais nettement plus lent à entraîner.
# ─────────────────────────────────────────────────────────────────────────────


class Classification:
    def __init__(self, model_name: str = "random_forest", models: dict = None):
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
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=Config.RANDOM_STATE
            ),
            "knn": KNeighborsClassifier(
                n_neighbors=5,
                algorithm="auto",
                n_jobs=-1,
            ) if None else models
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
    def save_model(pipeline: Pipeline, path: str, name: str = "model") -> None:
        path = path + f"{name}.joblib"
        joblib.dump(pipeline, path)

    @staticmethod
    def load_model(path: str, name: str) -> Pipeline:
        path = path + f"{name}.joblib"
        return joblib.load(path)

    @staticmethod
    def predict(pipeline: Pipeline, audio_path: str) -> dict:
        signal = AudioUtils.load_audio(audio_path)
        features = AudioUtils.extract_features(signal)
        # (1, n_features) — batch de 1
        X = features.reshape(1, -1)

        label = pipeline.predict(X)[0]

        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            probas = pipeline.predict_proba(X)[0]
            confidence = {
                name: round(float(p), 4) for name, p in zip(Config.CLASS_NAMES, probas)
            }
        else:
            confidence = None

        return {"label": label, "confidence": confidence}
