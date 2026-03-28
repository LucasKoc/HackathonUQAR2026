"""
Hackathon IA'Hack 2026 (hackathon.uqar.ca)
"""

from config import Config
from src.utils.dataset import AudioDataset
from src.utils.evaluate import Evaluate
from src.utils.features import Features
from src.utils.pipeline import Classification

if __name__ == "__main__":
    # 1. Chargement des données
    train_dataset = AudioDataset(Config.DATASET_PATH_AUDIO + Config.PATH_TRAIN)
    test_dataset = AudioDataset(Config.DATASET_PATH_AUDIO + Config.PATH_TEST)

    # 2. Extraction des features
    X_train, y_train = Features.extract_features(train_dataset)
    X_test, y_test = Features.extract_features(test_dataset)

    # 3. Entraînement du modèle
    # random_forest svm gradient_boosting
    model = "svm"
    pipeline = Classification(model)
    pipeline.train(X_train, y_train)
    Classification.save_model(pipeline.pipeline, name=model)
    del pipeline
    pipeline = Classification.load_model(name=model)

    # 4. Evaluation
    results = Evaluate.evaluate(pipeline, X_test, y_test)

    # 5. Visualisations
    Evaluate.plot_confusion_matrix(
        results["confusion_matrix"],
        save_path=str(Config.PATH_MODEL + f"confusion_matrix_{model}.png",
    ))
    Evaluate.plot_feature_importance(pipeline)

    exit()
