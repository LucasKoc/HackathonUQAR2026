class Config:
    # Files Path
    DATASET_PATH = "Data/Partie #1/Base de données/"
    PATH_TRAIN = "train/"
    PATH_TEST = "test/"
    PATH_MODEL = DATASET_PATH + "model/"

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
