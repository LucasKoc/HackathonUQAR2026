class Config:
    DATASET_PATH = "Data/Part 1/"
    DATASET_PATH_AUDIO = DATASET_PATH + "audio/"
    DATASET_PATH_SPECTROGRAM = DATASET_PATH + "spectrogram/"
    PATH_TRAIN = "train/"
    PATH_TEST = "test/"
    PATH_MODEL = DATASET_PATH + "model/"
    CLASS_NAMES = [
        "Beluga_WhiteWhale",
        "Fin_FinbackWhale",
        "HumpbackWhale",
        "SpermWhale",
        "White_sidedDolphin",
    ]
    AUDIO_SAMPLE_RATE = 22050
    AUDIO_N_MFCC = 13
    AUDIO_DURATION = 5
    RANDOM_STATE = 42
