"""
Hackathon IA'Hack 2026 (hackathon.uqar.ca)
"""

import os
import shutil
from os import mkdir
from pathlib import Path

import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import Config
from src.utils.dataset import AudioDataset
from src.utils.detection import Detection
from src.utils.evaluate import Evaluate
from src.utils.features import Features
from src.utils.files import Files
from src.utils.pipeline import Classification

if __name__ == "__main__":
    """
    ###
    # Partie 1 du défi
    ###

    # 0. Créer dossier model
    (
        mkdir(Path(Config.DATASET_PATH_P1 + Config.PATH_MODEL))
        if not Path(Config.DATASET_PATH_P1 + Config.PATH_MODEL).exists()
        else None
    )

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
        Classification.save_model(
            pipeline.pipeline,
            name=model,
            path=Config.DATASET_PATH_P1 + Config.PATH_MODEL,
        )
        del pipeline
        pipeline = Classification.load_model(
            name=model, path=Config.DATASET_PATH_P1 + Config.PATH_MODEL
        )

        # 4. Evaluation
        results = Evaluate.evaluate(pipeline, X_test, y_test, model)
        Evaluate.export_report_json(
            results["report_dict"],
            Config.DATASET_PATH_P1 + Config.PATH_MODEL + f"/report_{model}.json",
        )

        # 5. Visualisations
        Evaluate.plot_confusion_matrix(
            results["confusion_matrix"],
            save_path=str(
                Config.DATASET_PATH_P1
                + Config.PATH_MODEL
                + f"confusion_matrix_{model}.png",
            ),
            show_plot=False,
        )
        Evaluate.plot_feature_importance(pipeline, show_plot=False)
        ###
        # Prédiction d'un fichier audio
        ###

        # Prédiction sur un fichier audio
        pipeline = Classification.load_model(
            name=model, path=Config.DATASET_PATH_P1 + Config.PATH_MODEL
        )
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
    # 0.1. Copier la data de la Partie #1 vers la Partie #2
    Files.migration_p2()

    # 0.2. Créer dossier model
    (
        mkdir(Path(Config.DATASET_PATH_P2 + Config.PATH_MODEL))
        if not Path(Config.DATASET_PATH_P2 + Config.PATH_MODEL).exists()
        else None
    )

    # 1. Chargement des données
    train_dataset = AudioDataset(Config.DATASET_PATH_P1 + Config.PATH_TRAIN)
    test_dataset = AudioDataset(Config.DATASET_PATH_P1 + Config.PATH_TEST)

    train_files = [record.path for record in train_dataset.records]
    test_files = [record.path for record in test_dataset.records]
    train_labels = [record.label for record in train_dataset.records]
    test_labels = [record.label for record in test_dataset.records]

    X_train, y_train_labels, group_train = Features.build_window_dataset(
        train_files, train_labels, window_size_sec=Config.WINDOW_SIZE_SEC, hop_size_sec=Config.HOP_SIZE_SEC)

    X_test, y_test_labels, _ = Features.build_window_dataset(
        test_files, test_labels, window_size_sec=Config.WINDOW_SIZE_SEC, hop_size_sec=Config.HOP_SIZE_SEC)

    # Encoder & Scaler
    le = LabelEncoder()
    le.fit(list(set(train_labels + test_labels)))
    y_train = le.transform(y_train_labels)
    y_test = le.transform(y_test_labels)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=Config.RANDOM_STATE,
        class_weight='balanced'
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Test sur un fichier
    audio_path = "sequence_07.wav"
    audio, sr = librosa.load(Config.DATASET_PATH_P2_LS + Config.PATH_AUDIO + audio_path, sr=16000)
    print(f"Audio chargé : {len(audio) / sr:.2f}s, {sr} Hz")

    detections = Detection.sliding_window_detection(
        audio, sr, clf, scaler, le,
        window_size_sec=Config.WINDOW_SIZE_SEC,
        hop_size_sec=Config.HOP_SIZE_SEC,
        confidence_threshold=0.4,
        energy_threshold=None
    )

    merged_events = Detection.merge_consecutive_same_label(detections)

    # Affichage
    print(f"Nombre de détections brutes : {len(detections)}")
    print(f"Nombre après fusion : {len(merged_events)}")
    for e in merged_events:
        duration_ms = (e['end'] - e['start']) * 1000
        print(f"[{e['start']:.3f}s - {e['end']:.3f}s] {e['label']:15s} "
              f"({duration_ms:.0f}ms, confiance: {e['confidence']:.0%}, {e['count']} fenêtres)")
        # f"{e['probas']}")