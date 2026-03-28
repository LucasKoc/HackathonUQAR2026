from PIL import Image

import streamlit as st


def show_song_exemples():
    st.markdown("### Exemples de Songs")
    st.write("Écoutez des extraits d'animaux reconnus par notre IA :")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baleine à bosse")
        try:
            image1 = Image.open("streamlit/assets/pictures/animals/humpback_whale.jpg")
            st.image(image1, caption="Humpback Whale", width="stretch")
        except FileNotFoundError:
            st.warning("Image cover1.jpg non trouvée. Assurez-vous que le chemin est correct.")
        
        # Remplacez par le chemin du fichier audio
        # st.audio("streamlit/assets/songs/animals/audio1.mp3", format='audio/mp3')
        # if st.button("Écouter (1)", key="song1_btn"):
        #     st.write("Lecture de Song Title 1...")

    with col2:
        st.subheader("Bélugua")
        try:
            image2 = Image.open("streamlit/assets/pictures/animals/beluga.jpg")
            st.image(image2, caption="Bélugua", width="stretch")
        except FileNotFoundError:
            st.warning("Image belugua non trouvée. Assurez-vous que le chemin est correct.")
        
        # st.audio("streamlit/assets/songs/animals/audio2.mp3", format='audio/mp3')
        # if st.button("Écouter (2)", key="song2_btn"):
        #     st.write("Lecture de Song Title 2...")

    # La deuxième ligne (2 cartes)
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Cachalot")
        try:
            image3 = Image.open("streamlit/assets/pictures/animals/sperm_whale.jpg")
            st.image(image3, caption="Sperm Whale", width="stretch")
        except FileNotFoundError:
            st.warning("Image cachalot non trouvée. Assurez-vous que le chemin est correct.")
        
        # st.audio("streamlit/assets/songs/animals/audio3.mp3", format='audio/mp3')
        # if st.button("Écouter (3)", key="song3_btn"):
        #     st.write("Lecture de Song Title 3...")

    with col4:
        st.subheader("Dauphin à flancs blancs")
        try:
            image4 = Image.open(
                "streamlit/assets/pictures/animals/white_sided_dolphin.jpg"
            )
            st.image(image4, caption="White sided Dolphin", width="stretch")
        except FileNotFoundError:
            st.warning("Image dauphin non trouvée. Assurez-vous que le chemin est correct.")
        
        # st.audio("streamlit/assets/songs/animals/audio4.mp3", format='audio/mp3')
        # if st.button("Écouter (4)", key="song4_btn"):
        #     st.write("Lecture de Song Title 4...")

    # --- Ligne 3 : Une seule carte centrée ---
    col_empty1, col_center, col_empty2 = st.columns([1, 2, 1]) # Ratios pour centrer

    with col_center:
        st.subheader("Rorqual Commun")
        try:
            image5 = Image.open(
                "streamlit/assets/pictures/animals/fin_finback_whale.jpg"
            )
            st.image(image5, caption="Fin finback whale", width="stretch")
        except FileNotFoundError:
            st.warning("Image rorqual commun non trouvée. Assurez-vous que le chemin est correct.")
        
        # st.audio("streamlit/assets/songs/animals/audio5.mp3", format='audio/mp3')
        # if st.button("Écouter (5)", key="song5_btn"):
        #     st.write("Lecture de Song Title 5...")