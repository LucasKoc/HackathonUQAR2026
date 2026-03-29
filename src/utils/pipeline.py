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


class Classification:
    def __init__(self, model_name: str = "Random Forest"):
        self.pipeline = None
        self.models = self._build_models()
        self.build_pipeline(model_name)

    @staticmethod
    def _none_if_zero(value):
        return None if value in (0, None) else value

    def _build_models(self) -> dict:
        rf = Config.get_model_params("Random Forest")
        svm = Config.get_model_params("Support Vector Machine")
        gb = Config.get_model_params("Gradient Boosting")
        lr = Config.get_model_params("Logistic Regression")
        knn = Config.get_model_params("KNN")

        return {
            "Random Forest": RandomForestClassifier(
                n_estimators=rf["n_estimators"],
                max_depth=self._none_if_zero(rf["max_depth"]),
                min_samples_leaf=rf["min_samples_leaf"],
                class_weight=rf["class_weight"],
                random_state=Config.RANDOM_STATE,
                n_jobs=rf["n_jobs"],
            ),
            "Support Vector Machine": SVC(
                kernel=svm["kernel"],
                C=svm["C"],
                gamma=svm["gamma"],
                probability=svm["probability"],
                class_weight=svm["class_weight"],
                random_state=Config.RANDOM_STATE,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=gb["n_estimators"],
                learning_rate=gb["learning_rate"],
                max_depth=gb["max_depth"],
                random_state=Config.RANDOM_STATE,
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=lr["max_iter"],
                class_weight=lr["class_weight"],
                random_state=Config.RANDOM_STATE,
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=knn["n_neighbors"],
                algorithm=knn["algorithm"],
                n_jobs=knn["n_jobs"],
            ),
        }

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
    def save_model(pipeline: Pipeline, path: str | Path, name: str = "model") -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / f"{name}.joblib"
        joblib.dump(pipeline, model_path)

    @staticmethod
    def load_model(path: str | Path, name: str) -> Pipeline:
        path = Path(path)
        model_path = path / f"{name}.joblib"
        return joblib.load(model_path)

    @staticmethod
    def predict(pipeline: Pipeline, audio_path: str) -> dict:
        signal = AudioUtils.load_audio_full(audio_path)
        features = AudioUtils.extract_features(signal)
        X = features.reshape(1, -1)

        label = pipeline.predict(X)[0]

        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            probas = pipeline.predict_proba(X)[0]
            classes = pipeline.classes_
            confidence = {name: round(float(p), 4) for name, p in zip(classes, probas)}
        else:
            confidence = None

        return {"label": label, "confidence": confidence}

    @staticmethod
    def predict_all(
        audio_path: str, model_dir: str, audio_duration: float | None = None
    ) -> dict:
        Config.AUDIO_DURATION = (
            audio_duration if audio_duration is not None else Config.P1_AUDIO_DURATION
        )

        temp = Classification()
        results = {}

        for model_name in temp.models:
            pipeline = Classification.load_model(path=model_dir, name=model_name)
            results[model_name] = Classification.predict(pipeline, audio_path)

        return results
