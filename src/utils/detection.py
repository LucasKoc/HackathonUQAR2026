import csv
import json

import librosa
import numpy as np
from sklearn.metrics import jaccard_score

from config import Config
from src.utils.audio import AudioUtils


class Detection:
    """
    Pipeline :
        1. Charger l'audio long sans troncature
        2. Découper en fenêtres (sliding window)
        3. Classifier chaque fenêtre (6 classes : 5 espèces + noise)
        4. Fusionner les fenêtres consécutives du même label
        5. Filtrer le bruit
        6. Évaluer avec IoU contre les annotations CSV
    """

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
        confidence_threshold=0.7,
        energy_threshold=0.005,
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

            # ── Filtre énergie : skip les fenêtres silencieuses ──
            rms = np.sqrt(np.mean(window**2))
            if rms < energy_threshold:
                predictions.append(
                    {
                        "start_sec": round(start / sr, 3),
                        "end_sec": round(end / sr, 3),
                        "label": "noise",
                        "confidence": None,
                    }
                )
                continue

            # Extraction des mêmes features que l'entraînement
            features = AudioUtils.extract_features(window)
            X = features.reshape(1, -1)

            label = pipeline.predict(X)[0]

            # Filtre confidence
            confidence = None
            if hasattr(pipeline.named_steps["clf"], "predict_proba"):
                probas = pipeline.predict_proba(X)[0]
                classes = pipeline.classes_
                max_proba = float(np.max(probas))
                confidence = {
                    name: round(float(p), 4) for name, p in zip(classes, probas)
                }

                # Si la confiance est trop faible, marquer comme noise
                if max_proba < confidence_threshold:
                    label = "noise"

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
    def merge_detections(predictions, max_gap_sec=None, min_duration_sec=0.8):
        """
        Fusionne les fenêtres consécutives ayant le même label (non-noise),
        puis filtre les détections trop courtes.
        """
        if max_gap_sec is None:
            max_gap_sec = Config.HOP_SIZE_SEC + 0.01

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
                current["end_sec"] = pred["end_sec"]
            else:
                current["duration_sec"] = round(
                    current["end_sec"] - current["start_sec"], 3
                )
                merged.append(current)
                current = {
                    "label": pred["label"],
                    "start_sec": pred["start_sec"],
                    "end_sec": pred["end_sec"],
                }

        current["duration_sec"] = round(current["end_sec"] - current["start_sec"], 3)
        merged.append(current)

        # ── Filtrer les détections trop courtes ──
        merged = [d for d in merged if d["duration_sec"] >= min_duration_sec]

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
            s = int(ann["start_sec"] / resolution)
            e = int(ann["end_sec"] / resolution)
            gt_global[s:e] = 1

        for pred in predictions:
            s = int(pred["start_sec"] / resolution)
            e = int(pred["end_sec"] / resolution)
            pred_global[s:e] = 1

        # IoU globale avec jaccard_score
        iou_global = jaccard_score(gt_global, pred_global)

        # IoU par classe
        all_labels = set(
            [a["label"] for a in annotations] + [p["label"] for p in predictions]
        )
        iou_per_class = {}

        for lbl in sorted(all_labels):
            gt_class = np.zeros(num_frames, dtype=int)
            pred_class = np.zeros(num_frames, dtype=int)

            for ann in annotations:
                if ann["label"] == lbl:
                    s = int(ann["start_sec"] / resolution)
                    e = int(ann["end_sec"] / resolution)
                    gt_class[s:e] = 1

            for pred in predictions:
                if pred["label"] == lbl:
                    s = int(pred["start_sec"] / resolution)
                    e = int(pred["end_sec"] / resolution)
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
        confidence_threshold=0.5,
        energy_threshold=0.005,
        min_duration_sec=0.8,
    ):
        """
        Pipeline complet : charger -> sliding window -> classifier -> fusion.
        """
        signal = Detection.load_full_audio(audio_path)

        raw_preds = Detection.sliding_window_predict(
            signal,
            pipeline,
            window_size_sec,
            hop_size_sec,
            confidence_threshold=confidence_threshold,
            energy_threshold=energy_threshold,
        )

        merged = Detection.merge_detections(raw_preds, max_gap_sec, min_duration_sec)

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
