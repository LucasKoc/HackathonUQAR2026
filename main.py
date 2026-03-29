"""
Hackathon IA'Hack 2026 (hackathon.uqar.ca)
"""

import os
import shutil
from os import mkdir
from pathlib import Path

from config import Config
from src.utils.dataset import AudioDataset
from src.utils.evaluate import Evaluate
from src.utils.features import Features
from src.utils.files import Files
from src.utils.pipeline import Classification

if __name__ == "__main__":
    ###
    # Partie 1 du défi
    ###
    """
    # 0. Créer dossier model
    mkdir(Path(Config.PATH_MODEL)) if not Path(Config.PATH_MODEL).exists() else None

    # 1. Chargement des données
    train_dataset = AudioDataset(Config.DATASET_PATH_P1 + Config.PATH_TRAIN)
    test_dataset = AudioDataset(Config.DATASET_PATH_P1 + Config.PATH_TEST)

    # 2. Extraction des features
    X_train, y_train = Features.extract_features(train_dataset)
    X_test, y_test = Features.extract_features(test_dataset)

    # 3. Entraînement du modèle
    # random_forest svm gradient_boosting logistic_regression knn
    model = "random_forest"

    classification = Classification(model)

    # Tourner tous les modèles
    for model in classification.models:
        pipeline = Classification(model)
        pipeline.train(X_train, y_train)
        Classification.save_model(pipeline.pipeline, name=model)
        del pipeline
        pipeline = Classification.load_model(name=model)

        # 4. Evaluation
        results = Evaluate.evaluate(pipeline, X_test, y_test, model)
        Evaluate.export_report_json(
            results["report_dict"], Config.PATH_MODEL + f"/report_{model}.json"
        )

        # 5. Visualisations
        Evaluate.plot_confusion_matrix(
            results["confusion_matrix"],
            save_path=str(
                Config.PATH_MODEL + f"confusion_matrix_{model}.png",
            ),
            show_plot=False,
        )
        Evaluate.plot_feature_importance(pipeline, show_plot=False)
        ###
        # Prédiction d'un fichier audio
        ###

        # Prédiction sur un fichier audio
        pipeline = Classification.load_model(name=model)
        result = Classification.predict(
            pipeline, Config.DATASET_PATH_P1 + Config.PATH_TEST + "videoplayback.m4a"
        )

        print(f"Espèce prédite : {result['label']}")
        if result["confidence"]:
            for species, proba in sorted(
                result["confidence"].items(), key=lambda item: item[1], reverse=True
            ):
                print(f"  {species}: {proba:.2%}")
    """
    ###
    # Partie 2 du défi
    ###
    # 0. Copier la data de la Partie #1 vers la Partie #2
    Files.migration_p2()
    Config.CLASS_NAMES_P2 = Config.CLASS_NAMES + ["noise"]

