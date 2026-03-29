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
#  Support Vector Machine est souvent le meilleur sur peu de données mais plus lent.
#  GradientBoosting est plus précis mais nettement plus lent à entraîner.
# ─────────────────────────────────────────────────────────────────────────────


class Classification:
    def __init__(self, model_name: str = "Random Forest"):
        self.pipeline = None
        self.models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=Config.RANDOM_STATE,
                n_jobs=-1,
            ),
            "Support Vector Machine": SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=Config.RANDOM_STATE,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=Config.RANDOM_STATE,
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=Config.RANDOM_STATE,
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=5,
                algorithm="auto",
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
        signal = AudioUtils.load_audio_full(audio_path)
        features = AudioUtils.extract_features(signal)
        X = features.reshape(1, -1)

        label = pipeline.predict(X)[0]

        # Utiliser pipeline.classes_ pour supporter P1 (5 classes) et P2 (6 classes)
        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            probas = pipeline.predict_proba(X)[0]
            classes = pipeline.classes_
            confidence = {name: round(float(p), 4) for name, p in zip(classes, probas)}
        else:
            confidence = None

        return {"label": label, "confidence": confidence}

    @staticmethod
    def predict_all(audio_path: str, model_dir: str, audio_duration: float | None = None) -> dict:
        Config.AUDIO_DURATION = audio_duration if audio_duration is not None else 5.0

        temp = Classification()
        results = {}

        for model_name in temp.models:
            pipeline = Classification.load_model(path=model_dir, name=model_name)
            results[model_name] = Classification.predict(pipeline, audio_path)

        return results