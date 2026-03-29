"""
Hackathon IA'Hack 2026 (hackathon.uqar.ca)
"""
import glob
import os
from os import mkdir
from pathlib import Path

from config import Config
from src.utils.dataset import AudioDataset
from src.utils.detection import Detection
from src.utils.evaluate import Evaluate
from src.utils.features import Features
from src.utils.files import Files
from src.utils.pipeline import Classification

if __name__ == "__main__":
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
    ###
    # Partie 2 du défi
    ###
    Config.AUDIO_DURATION = Config.WINDOW_SIZE_SEC

    # 0.1. Copier la data de la Partie #1 vers la Partie #2
    Files.migration_p2()

    # 0.2. Créer dossier model
    (
        mkdir(Path(Config.DATASET_PATH_P2 + Config.PATH_MODEL))
        if not Path(Config.DATASET_PATH_P2 + Config.PATH_MODEL).exists()
        else None
    )

    # 1. Chargement des données
    train_dataset = AudioDataset(Config.DATASET_PATH_P2 + Config.PATH_TRAIN)
    test_dataset = AudioDataset(Config.DATASET_PATH_P2 + Config.PATH_TEST)

    print(f"Train : {len(train_dataset)} fichiers")
    print(f"Test  : {len(test_dataset)} fichiers")

    # 2. Extraction des features
    X_train, y_train = Features.extract_features(train_dataset)
    X_test, y_test = Features.extract_features(test_dataset)

    print(f"Features train : {X_train.shape}")
    print(f"Features test  : {X_test.shape}")
    print(f"Classes : {sorted(set(y_train))}")

    # 3. Training
    # random_forest svm gradient_boosting logistic_regression knn
    model_name = "random_forest"
    classification = Classification(model_name)
    classification.train(X_train, y_train)

    # Sauvegarder
    Classification.save_model(
        classification.pipeline,
        name=f"{model_name}_p2",
        path=Config.DATASET_PATH_P2 + Config.PATH_MODEL,
    )

    # Charger le modèle enregistré
    classification = Classification.load_model(
        name=f"{model_name}_p2",
        path=Config.DATASET_PATH_P2 + Config.PATH_MODEL,
    )

    # 4. Évaluation

    results = Evaluate.evaluate(
        classification,
        X_test,
        y_test,
        model_name,
        class_names=Config.CLASS_NAMES_P2,
    )
    Evaluate.export_report_json(
        results["report_dict"],
        Config.DATASET_PATH_P2 + Config.PATH_MODEL + f"report_{model_name}_p2.json",
    )
    Evaluate.plot_confusion_matrix(
        results["confusion_matrix"],
        class_names=Config.CLASS_NAMES_P2,
        save_path=Config.DATASET_PATH_P2
                  + Config.PATH_MODEL
                  + f"confusion_matrix_{model_name}_p2.png",
        show_plot=False,
    )

    # 5. Détection sur les longues séquences

    annotations_dir = Config.DATASET_PATH_P2_LS + Config.PATH_ANNOTATIONS
    audio_dir = Config.DATASET_PATH_P2_LS + Config.PATH_AUDIO

    # Trouver tous les fichiers audio longs
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))

    print(f"\n{'═' * 60}")
    print(f"  Détection sur {len(audio_files)} séquence(s) longue(s)")
    print(f"  Fenêtre : {Config.WINDOW_SIZE_SEC}s | Hop : {Config.HOP_SIZE_SEC}s")
    print(f"{'═' * 60}")

    all_iou_global = []

    for audio_path in audio_files:
        seq_name = Path(audio_path).stem
        csv_path = os.path.join(annotations_dir, f"{seq_name}.csv")

        print(f"\n{'─' * 40}")
        print(f"  Séquence : {seq_name}")
        print(f"{'─' * 40}")

        # 5.1. Détection
        detections, raw_preds, signal = Detection.detect(
            audio_path,
            classification,
            window_size_sec=Config.WINDOW_SIZE_SEC,
            hop_size_sec=Config.HOP_SIZE_SEC,
        )

        total_duration = len(signal) / Config.AUDIO_SAMPLE_RATE
        print(f"  Durée audio     : {total_duration:.1f}s")
        print(f"  Fenêtres brutes : {len(raw_preds)}")
        print(f"  Détections      : {len(detections)}")

        for det in detections:
            print(
                f"    [{det['start_sec']:.1f}s - {det['end_sec']:.1f}s] "
                f"→ {det['label']} ({det['duration_sec']:.1f}s)"
            )

        # 5.2. Évaluation IoU (si annotation disponible)
        if os.path.exists(csv_path):
            annotations = Detection.load_annotations(csv_path)
            iou_results = Detection.compute_iou(
                detections, annotations, total_duration
            )

            print(f"\n  IoU globale : {iou_results['iou_global']:.4f}")
            for cls, iou_val in iou_results["iou_per_class"].items():
                print(f"    {cls:25s} : {iou_val:.4f}")

            all_iou_global.append(iou_results["iou_global"])

            # Exporter les résultats
            Detection.export_detections_csv(
                detections,
                Config.DATASET_PATH_P2
                + Config.PATH_MODEL
                + f"detections_{seq_name}.csv",
            )
            Detection.export_results_json(
                detections,
                iou_results,
                Config.DATASET_PATH_P2
                + Config.PATH_MODEL
                + f"results_{seq_name}.json",
            )
        else:
            print(f"!! Pas d'annotation trouvée pour {seq_name}")

    # 6. Résumé global

    if all_iou_global:
        mean_iou = sum(all_iou_global) / len(all_iou_global)
        print(f"\n{'═' * 60}")
        print(f"  IoU moyenne sur {len(all_iou_global)} séquence(s) : {mean_iou:.4f}")
        print(f"{'═' * 60}")
