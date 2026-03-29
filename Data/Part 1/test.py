#%% Imports
import os
import glob
import numpy as np
import librosa
from config import Config
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

#%% Paramètres fenêtre glissante
WINDOW_SIZE_SEC = 1    # taille d'une fenêtre
HOP_SIZE_SEC = 0.5     # pas de glissement

#%% Chargement des fichiers audio
def load_wav_files(base_path):
    wav_files = []
    labels = []

    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for cls in classes:
        class_path = os.path.join(base_path, cls)
        files = glob.glob(os.path.join(class_path, "*.wav"))
        for f in files:
            wav_files.append(f)
            labels.append(cls)  # label = nom du dossier (inclut noise)
    return wav_files, labels

#%% Extraction de features
def extract_features(signal):
    n_fft = min(2048, len(signal))
    print("HIT") if len(signal) <= 2048 else None

    mfccs = np.mean(librosa.feature.mfcc(
        y=signal, sr=Config.AUDIO_SAMPLE_RATE, n_mfcc=Config.AUDIO_N_MFCC), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(
        y=signal, sr=Config.AUDIO_SAMPLE_RATE), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(
        y=signal, sr=Config.AUDIO_SAMPLE_RATE), axis=1)

    return np.concatenate((mfccs, chroma, contrast, zcr))

#%% Construction du dataset par fenêtres
def build_window_dataset(wav_files, labels, window_size_sec=0.05, hop_size_sec=0.025):
    all_features = []
    all_labels = []
    all_group_ids = []

    window_samples = int(window_size_sec * 16000)
    hop_samples = int(hop_size_sec * 16000)

    for i, (wf, label) in enumerate(zip(wav_files, labels)):
        if i % 50 == 0:
            print(f"Traitement {i+1}/{len(wav_files)}...")

        audio, sr = librosa.load(wf, sr=16000)
        group_id = wf  # pour éviter fuite train/test

        for start_sample in range(0, len(audio) - window_samples + 1, hop_samples):
            end_sample = start_sample + window_samples
            segment = audio[start_sample:end_sample]
            if len(segment) < window_samples:
                continue

            features = extract_features(segment)
            all_features.append(features)
            all_labels.append(label)
            all_group_ids.append(group_id)

    return np.array(all_features), np.array(all_labels), np.array(all_group_ids)

#%% Chemins
TRAIN_PATH = "audio/train"
TEST_PATH = "audio/test"

train_files, train_labels = load_wav_files(TRAIN_PATH)
test_files, test_labels = load_wav_files(TEST_PATH)


#%% Construire datasets
X_train, y_train_labels, group_train = build_window_dataset(
    train_files, train_labels, window_size_sec=WINDOW_SIZE_SEC, hop_size_sec=HOP_SIZE_SEC)

X_test, y_test_labels, _ = build_window_dataset(
    test_files, test_labels, window_size_sec=WINDOW_SIZE_SEC, hop_size_sec=HOP_SIZE_SEC)

#%% Encoder et scaler
le = LabelEncoder()
le.fit(list(set(train_labels + test_labels)))
y_train = le.transform(y_train_labels)
y_test = le.transform(y_test_labels)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Entraînement du classifieur
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

#%% Détection par fenêtre glissante
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

        features = extract_features(window).reshape(1, -1)
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

#%% Fusion des fenêtres consécutives du même label
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

#%% Test sur un fichier
audio_path = "sequence_07.wav"
audio, sr = librosa.load(audio_path, sr=16000)
print(f"Audio chargé : {len(audio)/sr:.2f}s, {sr} Hz")

detections = sliding_window_detection(
    audio, sr, clf, scaler, le,
    window_size_sec=WINDOW_SIZE_SEC,
    hop_size_sec=HOP_SIZE_SEC,
    confidence_threshold=0.4,
    energy_threshold=None
)

merged_events = merge_consecutive_same_label(detections)

# Affichage
print(f"Nombre de détections brutes : {len(detections)}")
print(f"Nombre après fusion : {len(merged_events)}")
for e in merged_events:
    duration_ms = (e['end'] - e['start'])*1000
    print(f"[{e['start']:.3f}s - {e['end']:.3f}s] {e['label']:15s} "
          f"({duration_ms:.0f}ms, confiance: {e['confidence']:.0%}, {e['count']} fenêtres)")
          #f"{e['probas']}")