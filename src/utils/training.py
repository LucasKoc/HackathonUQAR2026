from pathlib import Path

from config import Config
from src.utils.dataset import AudioDataset
from src.utils.evaluate import Evaluate
from src.utils.features import Features
from src.utils.pipeline import Classification


class Trainer:
    @staticmethod
    def train_part1(selected_models: list[str]) -> dict:
        Config.AUDIO_DURATION = Config.P1_AUDIO_DURATION

        model_dir = Path(Config.DATASET_PATH_P1) / Config.PATH_MODEL
        model_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = AudioDataset(Config.DATASET_PATH_P1 + Config.PATH_TRAIN)
        test_dataset = AudioDataset(Config.DATASET_PATH_P1 + Config.PATH_TEST)

        X_train, y_train = Features.extract_features(train_dataset)
        X_test, y_test = Features.extract_features(test_dataset)

        summary = {}

        for model_name in selected_models:
            clf = Classification(model_name)
            clf.train(X_train, y_train)

            Classification.save_model(
                clf.pipeline,
                path=Config.DATASET_PATH_P1 + Config.PATH_MODEL,
                name=model_name,
            )

            results = Evaluate.evaluate(clf.pipeline, X_test, y_test, model_name)

            Evaluate.export_report_json(
                results["report_dict"],
                Config.DATASET_PATH_P1
                + Config.PATH_MODEL
                + f"/report_{model_name}.json",
            )

            Evaluate.plot_confusion_matrix(
                results["confusion_matrix"],
                save_path=str(
                    Path(Config.DATASET_PATH_P1)
                    / Config.PATH_MODEL
                    / f"confusion_matrix_{model_name}.png"
                ),
                show_plot=False,
            )

            summary[model_name] = {
                "accuracy": results["accuracy"],
                "n_train": len(X_train),
                "n_test": len(X_test),
            }

        return summary

    @staticmethod
    def train_part2(model_name: str = "Random Forest") -> dict:
        Config.AUDIO_DURATION = Config.WINDOW_SIZE_SEC

        model_dir = Path(Config.DATASET_PATH_P2) / Config.PATH_MODEL
        model_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = AudioDataset(Config.DATASET_PATH_P2 + Config.PATH_TRAIN)
        test_dataset = AudioDataset(Config.DATASET_PATH_P2 + Config.PATH_TEST)

        X_train, y_train = Features.extract_features(train_dataset)
        X_test, y_test = Features.extract_features(test_dataset)

        clf = Classification(model_name)
        clf.train(X_train, y_train)

        save_name = f"{model_name}_p2"

        Classification.save_model(
            clf.pipeline,
            path=Config.DATASET_PATH_P2 + Config.PATH_MODEL,
            name=save_name,
        )

        results = Evaluate.evaluate(
            clf.pipeline,
            X_test,
            y_test,
            model_name,
            class_names=Config.CLASS_NAMES_P2,
        )

        Evaluate.export_report_json(
            results["report_dict"],
            Config.DATASET_PATH_P2 + Config.PATH_MODEL + f"report_{save_name}.json",
        )

        Evaluate.plot_confusion_matrix(
            results["confusion_matrix"],
            class_names=Config.CLASS_NAMES_P2,
            save_path=Config.DATASET_PATH_P2
            + Config.PATH_MODEL
            + f"confusion_matrix_{save_name}.png",
            show_plot=False,
        )

        return {
            "model": model_name,
            "accuracy": results["accuracy"],
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
