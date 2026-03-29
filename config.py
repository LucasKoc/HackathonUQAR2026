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

    # Audio settings
    AUDIO_SAMPLE_RATE = 44100
    AUDIO_N_MFCC = 13
    AUDIO_DURATION = 5.0

    # Classification settings
    RANDOM_STATE = 42

    # P2 Audio settings
    WINDOW_SIZE_SEC = 2.0
    HOP_SIZE_SEC = 0.5

    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
        # RMS min (filtre les fenêtres silencieuses)
    ENERGY_THRESHOLD = 0.005
    MIN_DETECTION_SEC = 0.8
