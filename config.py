from copy import deepcopy


class Config:
    # Files Path
    DATASET_PATH_P1 = "Data/Partie #1/Base de données/"
    DATASET_PATH_P2 = "Data/Partie #2/Complément de base de données/"
    DATASET_PATH_P2_LS = "Data/Partie #2/Longues séquences/"

    PATH_TRAIN = "train/"
    PATH_TEST = "test/"
    PATH_MODEL = "model/"
    PATH_ANNOTATIONS = "annotations/"
    PATH_AUDIO = "audio/"

    # Classes
    CLASS_NAMES = [
        "Beluga_WhiteWhale",
        "Fin_FinbackWhale",
        "HumpbackWhale",
        "SpermWhale",
        "White_sidedDolphin",
    ]

    CLASS_NAMES_P2 = CLASS_NAMES + ["noise"]

    # Audio / features
    AUDIO_SAMPLE_RATE = 44100
    AUDIO_N_MFCC = 13
    P1_AUDIO_DURATION = 5.0

    # P2
    WINDOW_SIZE_SEC = 0.5
    HOP_SIZE_SEC = 0.25

    # AUDIO_DURATION DOIT correspondre à WINDOW_SIZE_SEC
    # pour que les features d'entraînement et de détection soient compatibles.
    # Pour la P1, on le redéfinit à 5.0 dans main.py avant l'extraction.
    AUDIO_DURATION = WINDOW_SIZE_SEC

    # Global
    RANDOM_STATE = 42

    # Detection
    CONFIDENCE_THRESHOLD = 0.5
    ENERGY_THRESHOLD = 0.001
    MIN_DETECTION_SEC = 0.8

    # Model hyperparameters
    MODEL_PARAMS = {
        "Random Forest": {
            "n_estimators": 300,
            "max_depth": 0,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "n_jobs": -1,
        },
        "Support Vector Machine": {
            "C": 10.0,
            "gamma": "scale",
            "kernel": "rbf",
            "probability": True,
            "class_weight": "balanced",
        },
        "Gradient Boosting": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5,
        },
        "Logistic Regression": {
            "max_iter": 1000,
            "class_weight": "balanced",
        },
        "KNN": {
            "n_neighbors": 5,
            "algorithm": "auto",
            "n_jobs": -1,
        },
    }

    @classmethod
    def get_model_params(cls, model_name: str) -> dict:
        return deepcopy(cls.MODEL_PARAMS[model_name])

    @classmethod
    def apply_runtime_overrides(cls, payload: dict) -> None:
        general = payload.get("general", {})
        for key, value in general.items():
            setattr(cls, key, value)

        models = payload.get("models", {})
        for model_name, params in models.items():
            if model_name in cls.MODEL_PARAMS:
                cls.MODEL_PARAMS[model_name].update(params)
