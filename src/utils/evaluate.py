import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from config import Config


class Evaluate:
    @staticmethod
    def evaluate(
        pipeline: Pipeline,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model: str,
        class_names: list = None,
    ) -> dict:
        if class_names is None:
            class_names = Config.CLASS_NAMES

        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=3,
        )
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{'─' * 20} Modèle : {model} {'─' * 20}")
        print(f"  Accuracy : {acc:.3f}  ({acc * 100:.1f} %)")
        print(f"{'─' * 60}")
        print(report)

        report_dict = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        )

        return {
            "accuracy": acc,
            "report": report,
            "report_dict": report_dict,
            "confusion_matrix": cm,
        }

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: list = None,
        save_path: str | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Affiche la matrice de confusion normalisée avec seaborn.
        """
        if class_names is None:
            class_names = Config.CLASS_NAMES

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        short_names = [n.replace("_", "\n") for n in class_names]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=short_names,
            yticklabels=short_names,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_xlabel("Classe prédite", fontsize=12)
        ax.set_ylabel("Classe réelle", fontsize=12)
        ax.set_title("Matrice de confusion (normalisée)", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        plt.show() if show_plot else None

    @staticmethod
    def plot_feature_importance(
        pipeline: Pipeline,
        top_n: int = 20,
        save_path: str | None = None,
        show_plot: bool = True,
    ) -> None:
        """
        Affiche les N features les plus importantes (Random Forest uniquement).
        Ignoré silencieusement pour SVM et GradientBoosting.
        """
        clf = pipeline.named_steps["clf"]
        if not hasattr(clf, "feature_importances_"):
            return

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(top_n), importances[indices], color="steelblue")
        ax.set_xticks(range(top_n))
        ax.set_xticklabels([f"f{i}" for i in indices], rotation=45, ha="right")
        ax.set_xlabel("Index feature")
        ax.set_ylabel("Importance")
        ax.set_title(f"Top {top_n} features importantes (Random Forest)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        plt.show() if show_plot else None

    @staticmethod
    def export_report_json(report_dict: dict, save_path: str) -> None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
