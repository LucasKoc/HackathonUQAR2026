import csv
import json

import librosa
import numpy as np
from sklearn.metrics import jaccard_score

from config import Config
from src.utils.audio import AudioUtils

"""
Pipeline :
    1. Charger l'audio long sans troncature
    2. Découper en fenêtres (sliding window)
    3. Classifier chaque fenêtre (6 classes : 5 espèces + noise)
    4. Fusionner les fenêtres consécutives du même label
    5. Filtrer le bruit
    6. Évaluer avec IoU contre les annotations CSV
"""
class Detection:
    @staticmethod
    def load_full_audio(filepath, sr=Config.AUDIO_SAMPLE_RATE):
        """Charge un fichier audio SANS troncature ni padding."""
        signal, _ = librosa.load(filepath, sr=sr, mono=True)
        return signal

    @staticmethod
    def sliding_window_predict(
        signal,
        pipeline,
        window_size_sec=Config.WINDOW_SIZE_SEC,
        hop_size_sec=Config.HOP_SIZE_SEC,
        sr=Config.AUDIO_SAMPLE_RATE,
    ):
        """
        Découpe le signal en fenêtres et classifie chacune.
        """
        window_samples = int(window_size_sec * sr)
        hop_samples = int(hop_size_sec * sr)

        predictions = []

        for start in range(0, len(signal) - window_samples + 1, hop_samples):
            end = start + window_samples
            window = signal[start:end]

            # Extraction des mêmes features que l'entraînement
            features = AudioUtils.extract_features(window)
            X = features.reshape(1, -1)

            label = pipeline.predict(X)[0]

            # Probabilités si le classifieur les supporte
            confidence = None
            if hasattr(pipeline.named_steps["clf"], "predict_proba"):
                probas = pipeline.predict_proba(X)[0]
                classes = pipeline.classes_
                confidence = {
                    name: round(float(p), 4) for name, p in zip(classes, probas)
                }

            predictions.append(
                {
                    "start_sec": round(start / sr, 3),
                    "end_sec": round(end / sr, 3),
                    "label": label,
                    "confidence": confidence,
                }
            )

        return predictions

    @staticmethod
    def merge_detections(predictions, max_gap_sec=None):
        """
        Fusionne les fenêtres consécutives ayant le même label (non-noise).
        """
        if max_gap_sec is None:
            max_gap_sec = Config.HOP_SIZE_SEC

        # Filtrer le bruit
        filtered = [p for p in predictions if p["label"] != "noise"]

        if not filtered:
            return []

        merged = []
        current = {
            "label": filtered[0]["label"],
            "start_sec": filtered[0]["start_sec"],
            "end_sec": filtered[0]["end_sec"],
        }

        for pred in filtered[1:]:
            same_label = pred["label"] == current["label"]
            gap = pred["start_sec"] - current["end_sec"]

            if same_label and gap <= max_gap_sec:
                # Étendre l'intervalle courant
                current["end_sec"] = pred["end_sec"]
            else:
                # Sauvegarder l'intervalle courant et en démarrer un nouveau
                current["duration_sec"] = round(
                    current["end_sec"] - current["start_sec"], 3
                )
                merged.append(current)
                current = {
                    "label": pred["label"],
                    "start_sec": pred["start_sec"],
                    "end_sec": pred["end_sec"],
                }

        # Dernier intervalle
        current["duration_sec"] = round(current["end_sec"] - current["start_sec"], 3)
        merged.append(current)

        return merged

    @staticmethod
    def load_annotations(csv_path):
        annotations = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotations.append(
                    {
                        "label": row["animal"],
                        "start_sec": float(row["start_sec"]),
                        "end_sec": float(row["end_sec"]),
                        "duration_sec": float(row["duration_sec"]),
                    }
                )
        return annotations

    @staticmethod
    def compute_iou(predictions, annotations, total_duration, resolution=0.01):
        num_frames = int(total_duration / resolution) + 1

        gt_global = np.zeros(num_frames, dtype=int)
        pred_global = np.zeros(num_frames, dtype=int)

        for ann in annotations:
            s, e = int(ann["start_sec"] / resolution), int(ann["end_sec"] / resolution)
            gt_global[s:e] = 1

        for pred in predictions:
            s, e = int(pred["start_sec"] / resolution), int(pred["end_sec"] / resolution)
            pred_global[s:e] = 1

        # IoU globale avec jaccard_score
        iou_global = jaccard_score(gt_global, pred_global)

        # IoU par classe (même principe)
        all_labels = set(
            [a["label"] for a in annotations] + [p["label"] for p in predictions]
        )
        iou_per_class = {}

        for lbl in sorted(all_labels):
            gt_class = np.zeros(num_frames, dtype=int)
            pred_class = np.zeros(num_frames, dtype=int)

            for ann in annotations:
                if ann["label"] == lbl:
                    s, e = int(ann["start_sec"] / resolution), int(ann["end_sec"] / resolution)
                    gt_class[s:e] = 1

            for pred in predictions:
                if pred["label"] == lbl:
                    s, e = int(pred["start_sec"] / resolution), int(pred["end_sec"] / resolution)
                    pred_class[s:e] = 1

            iou_per_class[lbl] = jaccard_score(gt_class, pred_class)

        return {"iou_global": round(iou_global, 4), "iou_per_class": iou_per_class}

    @staticmethod
    def detect(
        audio_path,
        pipeline,
        window_size_sec=Config.WINDOW_SIZE_SEC,
        hop_size_sec=Config.HOP_SIZE_SEC,
        max_gap_sec=None,
    ):
        """
        Pipeline complet : charger -> sliding window -> classifier -> fusion.
        """
        signal = Detection.load_full_audio(audio_path)

        raw_preds = Detection.sliding_window_predict(
            signal, pipeline, window_size_sec, hop_size_sec
        )

        merged = Detection.merge_detections(raw_preds, max_gap_sec)

        return merged, raw_preds, signal

    @staticmethod
    def export_detections_csv(detections, save_path):
        """Exporte les détections au même format que les annotations."""
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["animal", "start_sec", "end_sec", "duration_sec"]
            )
            writer.writeheader()
            for det in detections:
                writer.writerow(
                    {
                        "animal": det["label"],
                        "start_sec": round(det["start_sec"], 3),
                        "end_sec": round(det["end_sec"], 3),
                        "duration_sec": round(det["duration_sec"], 3),
                    }
                )

    @staticmethod
    def export_results_json(detections, iou_results, save_path):
        """Exporte détections + métriques IoU en JSON."""
        output = {
            "num_events": len(detections),
            "iou_global": iou_results["iou_global"],
            "iou_per_class": iou_results["iou_per_class"],
            "events": [
                {
                    "label": d["label"],
                    "start_sec": d["start_sec"],
                    "end_sec": d["end_sec"],
                    "duration_sec": d["duration_sec"],
                }
                for d in detections
            ],
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
