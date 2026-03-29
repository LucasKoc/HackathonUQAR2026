from pathlib import Path

import matplotlib.pyplot as plt

import streamlit as st
from config import Config
from src.utils.detection import Detection
from src.utils.pipeline import Classification
from src.utils.spectrogram import Spectrogram


def _pretty_label(label: str) -> str:
    mapping = {
        "Beluga_WhiteWhale": "Beluga",
        "Fin_FinbackWhale": "Fin Whale",
        "HumpbackWhale": "Humpback",
        "SpermWhale": "Sperm Whale",
        "White_sidedDolphin": "Dolphin",
        "noise": "Noise",
    }
    return mapping.get(label, label.replace("_", " "))


def _plot_timeline(detections, total_duration):
    labels = sorted(set(d["label"] for d in detections)) if detections else []
    n_labels = max(len(labels), 1)

    fig_height = max(2.8, 1.2 + 0.7 * n_labels)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    if not detections:
        ax.set_xlim(0, max(total_duration, 1))
        ax.set_ylim(0, 1)
        ax.text(
            max(total_duration, 1) / 2,
            0.5,
            "Aucune zone détectée",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_yticks([])
        ax.set_xlabel("Temps (s)")
        ax.set_title("Timeline des zones reconnues", loc="left", fontsize=14, pad=12)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        plt.tight_layout()
        return fig

    color_map = {
        "Beluga_WhiteWhale": "#4C78A8",
        "Fin_FinbackWhale": "#F58518",
        "HumpbackWhale": "#54A24B",
        "SpermWhale": "#E45756",
        "White_sidedDolphin": "#72B7B2",
        "noise": "#B0B0B0",
    }

    y_gap = 12
    bar_height = 7
    y_positions = {label: i * y_gap for i, label in enumerate(labels)}

    for det in detections:
        start = det["start_sec"]
        end = det["end_sec"]
        duration = det["duration_sec"]
        label = det["label"]

        ax.broken_barh(
            [(start, duration)],
            (y_positions[label], bar_height),
            facecolors=color_map.get(label, "#888888"),
            edgecolors="none",
            alpha=0.9,
        )

        # n'afficher le texte que si la barre est assez large
        if duration >= max(total_duration * 0.06, 0.8):
            ax.text(
                start + duration / 2,
                y_positions[label] + bar_height / 2,
                f"{duration:.1f}s",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    ax.set_xlim(0, max(total_duration, 1))
    ax.set_ylim(-2, max(y_positions.values()) + y_gap - 2)
    ax.set_yticks([y + bar_height / 2 for y in y_positions.values()])
    ax.set_yticklabels([_pretty_label(label) for label in labels], fontsize=10)

    ax.set_xlabel("Temps (s)")
    ax.set_title("Timeline des zones reconnues", loc="left", fontsize=14, pad=12)

    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    return fig


@st.cache_resource
def _load_p2_model():
    model_dir = Config.DATASET_PATH_P2 + Config.PATH_MODEL
    return Classification.load_model(path=model_dir, name="Random Forest_p2")


def show_long_audio():
    st.markdown("### Détection sur séquence longue — Partie 2")
    st.write(
        "Déposez un fichier audio long pour détecter les zones reconnues sur la timeline."
    )

    uploaded_file = st.file_uploader(
        "Choisissez un fichier audio long",
        type=["wav", "mp3", "m4a"],
        key="p2_uploader",
    )

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        launch_analysis = st.button(
            "🚀 Lancer l'analyse longue séquence", width="stretch"
        )

    if launch_analysis:
        if uploaded_file is None:
            st.error("Veuillez d'abord importer un fichier audio.")
        else:
            with st.spinner("Détection des zones en cours..."):
                temp_dir = Path("streamlit/temp")
                temp_dir.mkdir(parents=True, exist_ok=True)

                temp_audio_path = temp_dir / uploaded_file.name
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                spectro_path = temp_dir / f"{temp_audio_path.stem}_spectrogram.png"
                Spectrogram.create_single_spectrogram(temp_audio_path, spectro_path)

                # Important : P2 travaille avec des fenêtres courtes
                Config.AUDIO_DURATION = Config.WINDOW_SIZE_SEC

                pipeline_p2 = _load_p2_model()

                detections, raw_preds, signal = Detection.detect(
                    audio_path=str(temp_audio_path),
                    pipeline=pipeline_p2,
                    window_size_sec=Config.WINDOW_SIZE_SEC,
                    hop_size_sec=Config.HOP_SIZE_SEC,
                    confidence_threshold=Config.CONFIDENCE_THRESHOLD,
                    energy_threshold=Config.ENERGY_THRESHOLD,
                    min_duration_sec=Config.MIN_DETECTION_SEC,
                )

                total_duration = len(signal) / Config.AUDIO_SAMPLE_RATE

                st.session_state["p2_done"] = True
                st.session_state["p2_uploaded_filename"] = uploaded_file.name
                st.session_state["p2_spectrogram_path"] = str(spectro_path)
                st.session_state["p2_detections"] = detections
                st.session_state["p2_raw_preds"] = raw_preds
                st.session_state["p2_total_duration"] = total_duration

    if st.session_state.get("p2_done", False):
        st.divider()
        st.markdown(
            "<h2 style='text-align: center;'>🧭 Résultat de la détection</h2>",
            unsafe_allow_html=True,
        )

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            spectrogram_path = st.session_state.get("p2_spectrogram_path")
            if spectrogram_path and Path(spectrogram_path).exists():
                st.image(
                    spectrogram_path,
                    caption="Spectrogramme de la séquence",
                    width="stretch",
                )
            else:
                st.info("🖼️ Spectrogramme introuvable")

        with res_col2:
            st.write("### Analyse terminée")
            st.write(
                f"**Fichier analysé :** {st.session_state.get('p2_uploaded_filename', 'N/A')}"
            )
            st.write(
                f"**Durée totale :** {st.session_state.get('p2_total_duration', 0):.2f} s"
            )
            st.write(
                f"**Nombre de détections fusionnées :** {len(st.session_state.get('p2_detections', []))}"
            )

            detections = st.session_state.get("p2_detections", [])
            total_duration = st.session_state.get("p2_total_duration", 0)

            fig = _plot_timeline(detections, total_duration)
            st.pyplot(fig)

            st.markdown("#### Zones reconnues")

            if detections:
                cols = st.columns(3)

                for i, det in enumerate(detections):
                    with cols[i % 3]:
                        st.markdown(
                            f"""
                            <div style="
                                margin-bottom:10px;
                            ">
                                <div style="font-weight:600; margin-bottom:6px;">
                                    {_pretty_label(det['label'])}
                                </div>
                                <div><b>Début :</b> {det['start_sec']:.2f}s</div>
                                <div><b>Fin :</b> {det['end_sec']:.2f}s</div>
                                <div><b>Durée :</b> {det['duration_sec']:.2f}s</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.info("Aucune zone reconnue après fusion et filtrage.")
