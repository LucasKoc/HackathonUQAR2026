class Config:
    # Files Path
    DATASET_PATH_P1 = "Data/Partie #1/Base de données/"
    DATASET_PATH_P2 = "Data/Partie #2/Complément de base de données/"

    PATH_TRAIN = "train/"
    PATH_TEST = "test/"
    PATH_MODEL = DATASET_PATH_P1 + "model/"

    # Classes
    CLASS_NAMES = [
        "Beluga_WhiteWhale",
        "Fin_FinbackWhale",
        "HumpbackWhale",
        "SpermWhale",
        "White_sidedDolphin",
    ]

    # Audio settings
    AUDIO_SAMPLE_RATE = 22050
    AUDIO_N_MFCC = 13
    AUDIO_DURATION = 5.0

    # Classification settings
    RANDOM_STATE = 42
