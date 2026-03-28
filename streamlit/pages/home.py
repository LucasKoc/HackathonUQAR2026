import streamlit as st
import time
from PIL import Image

def show_home():
    st.markdown("### Bienvenue à l'IA'Hack 2026")
    st.write("Déposez votre fichier audio pour détecter l'animal présent dans l'extrait.")

    # Zone d'upload de fichier
    uploaded_file = st.file_uploader(
        "Choisissez un fichier (WAV)", 
        type=["wav"]
    )

    # Bouton d'analyse
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    
    with col_btn2:
        launch_analysis = st.button("🚀 Lancer l'analyse", width='stretch')

    # Lancement de l'analyse
    if launch_analysis:
        if uploaded_file is not None:
            with st.spinner('Analyse de l\'IA en cours...'):
                # On simule un temps de chargement (Mock)
                time.sleep(2)
            
            st.session_state['analysis_done'] = True
        else:
            st.error("Veuillez d'abord importer un fichier audio.")

    # Section résultat - Affichée uniquement après analyse
    if st.session_state.get('analysis_done', False):
        st.divider()
        st.markdown("<h2 style='text-align: center;'>📊 Résultat de l'analyse</h2>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            # Image du spectogramme
            try:
                res_img = Image.open("streamlit/assets/pictures/result_icon.png")
                st.image(res_img, width='stretch')
            except FileNotFoundError:
                st.info("🖼️ [Image du spectogram]")
        
        with res_col2:
            st.write("### Analyse terminée avec succès !")
            st.write(f"**Fichier analysé :** {uploaded_file.name}")
            st.success("""
                L'intelligence artificielle a détecté des motifs acoustiques 
                correspondant à une signature de **Baleine à bosse**. 
                Le niveau de confiance est de **98.4%**.
            """)