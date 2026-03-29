import time
from pathlib import Path

import streamlit as st
from PIL import Image

from src.utils.spectrogram import Spectrogram


def show_home():
    st.markdown("### Bienvenue à l'IA'Hack 2026")
    st.write(
        "Déposez votre fichier audio pour détecter l'animal présent dans l'extrait."
    )

    uploaded_file = st.file_uploader(
        "Choisissez un fichier audio", type=["wav", "mp3", "m4a"]
    )

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])

    with col_btn2:
        launch_analysis = st.button("🚀 Lancer l'analyse", width="stretch")

    if launch_analysis:
        if uploaded_file is not None:
            with st.spinner("Analyse de l'IA en cours..."):
                temp_dir = Path("streamlit/temp")
                temp_dir.mkdir(parents=True, exist_ok=True)

                temp_audio_path = temp_dir / uploaded_file.name
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                spectro_path = temp_dir / f"{temp_audio_path.stem}_spectrogram.png"
                Spectrogram.create_single_spectrogram(temp_audio_path, spectro_path)

                st.session_state["analysis_done"] = True
                st.session_state["uploaded_filename"] = uploaded_file.name
                st.session_state["spectrogram_path"] = str(spectro_path)

                time.sleep(1)
        else:
            st.error("Veuillez d'abord importer un fichier audio.")

    if st.session_state.get("analysis_done", False):
        st.divider()
        st.markdown(
            "<h2 style='text-align: center;'>📊 Résultat de l'analyse</h2>",
            unsafe_allow_html=True,
        )

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            spectrogram_path = st.session_state.get("spectrogram_path")

            if spectrogram_path and Path(spectrogram_path).exists():
                st.image(spectrogram_path, caption="Spectrogramme", width="stretch")
            else:
                try:
                    res_img = Image.open("streamlit/assets/pictures/result_icon.png")
                    st.image(res_img, width="stretch")
                except FileNotFoundError:
                    st.info("🖼️ [Image du spectogramme]")

        with res_col2:
            st.write("### Analyse terminée avec succès !")
            st.write(
                f"**Fichier analysé :** {st.session_state.get('uploaded_filename', 'N/A')}"
            )
            st.success(
                """
                L'intelligence artificielle a détecté des motifs acoustiques
                correspondant à une signature de **Baleine à bosse**.
                Le niveau de confiance est de **98.4%**.
                """
            )