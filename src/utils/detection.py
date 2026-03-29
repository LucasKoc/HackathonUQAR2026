import numpy as np

from src.utils.audio import AudioUtils


class Detection:
    @staticmethod
    def sliding_window_detection(audio, sr, classifier, scaler, label_encoder,
                                 window_size_sec=0.2, hop_size_sec=0.1,
                                 confidence_threshold=0.5, energy_threshold=None):
        window_samples = int(window_size_sec * sr)
        hop_samples = int(hop_size_sec * sr)
        if not hasattr(classifier, 'predict_proba'):
            raise TypeError("Classifier doit avoir predict_proba()")

        results = []

        for start_sample in range(0, len(audio) - window_samples + 1, hop_samples):
            end_sample = start_sample + window_samples
            window = audio[start_sample:end_sample]

            # Optionnel : ignorer faible énergie
            if energy_threshold is not None:
                energy = np.sqrt(np.mean(window ** 2))
                if energy < energy_threshold:
                    continue

            features = AudioUtils.extract_features(window).reshape(1, -1)
            features_scaled = scaler.transform(features)

            # Prédiction probabiliste
            probas = classifier.predict_proba(features_scaled)[0]
            pred_idx = np.argmax(probas)
            confidence = probas[pred_idx]
            label_name = label_encoder.inverse_transform([pred_idx])[0]

            if confidence >= confidence_threshold:
                results.append({
                    'start': start_sample / sr,
                    'end': end_sample / sr,
                    'label': label_name,
                    'confidence': confidence,
                    'probas': {cls: float(p) for cls, p in zip(label_encoder.classes_, probas)}
                })
        return results

    @staticmethod
    def merge_consecutive_same_label(detections):
        if not detections:
            return []

        merged = []
        current = {
            'start': detections[0]['start'],
            'end': detections[0]['end'],
            'label': detections[0]['label'],
            'confidence': detections[0]['confidence'],
            'probas': detections[0]['probas'],
            'count': 1
        }

        for det in detections[1:]:
            if det['label'] == current['label']:
                current['end'] = det['end']
                current['confidence'] = max(current['confidence'], det['confidence'])
                current['count'] += 1
            else:
                merged.append(current)
                current = {
                    'start': det['start'],
                    'end': det['end'],
                    'label': det['label'],
                    'confidence': det['confidence'],
                    'probas': det['probas'],
                    'count': 1
                }

        merged.append(current)
        return merged