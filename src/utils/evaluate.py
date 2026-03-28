import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.pipeline import Pipeline

from config import Config


class Evaluate:
    @staticmethod
    def evaluate(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Calcule les métriques d'évaluation et affiche le rapport.
        """
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=Config.CLASS_NAMES,
            digits=3,
        )
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{'─' * 60}")
        print(f"  Accuracy : {acc:.3f}  ({acc * 100:.1f} %)")
        print(f"{'─' * 60}")
        print(report)

        return {"accuracy": acc, "report": report, "confusion_matrix": cm}

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, save_path: str | None = None) -> None:
        """
        Affiche la matrice de confusion normalisée avec seaborn.
        """
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        short_names = [n.replace("_", "\n") for n in Config.CLASS_NAMES]

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
            print(f"Matrice sauvegardée → {save_path}")

        plt.show()


    @staticmethod
    def plot_feature_importance(
        pipeline: Pipeline, top_n: int = 20, save_path: str | None = None
    ) -> None:
        """
        Affiche les N features les plus importantes (Random Forest uniquement).
        Ignoré silencieusement pour SVM et GradientBoosting.
        """
        clf = pipeline.named_steps["clf"]
        if not hasattr(clf, "feature_importances_"):
            print("[INFO] Importance des features non disponible pour ce modèle.")
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

        plt.show()
