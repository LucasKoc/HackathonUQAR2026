from PIL import Image

import streamlit as st


def show_song_exemples():
    st.markdown("### Exemples de chants")
    st.write("Écoutez des extraits d'animaux reconnus par notre IA :")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("[Baleine à bosse](https://fr.wikipedia.org/wiki/Baleine_%C3%A0_bosse)")
        try:
            image1 = Image.open("streamlit/assets/pictures/animals/humpback_whale.jpg")
            st.image(image1, caption="Humpback Whale", width="stretch")
        except FileNotFoundError:
            st.warning("Image baleine à bosse non trouvée. Assurez-vous que le chemin est correct.")
        try:
            audio4 = open("streamlit/assets/songs/animals/humpback_whale.mp3", "rb")
            st.audio(audio4, format='audio/mp3')
        except FileNotFoundError:
            st.warning("Audio baleine à bosse non trouvé. Assurez-vous que le chemin est correct.")

    with col2:
        st.subheader("[Bélugua](https://fr.wikipedia.org/wiki/B%C3%A9luga_(c%C3%A9tac%C3%A9))")
        try:
            image2 = Image.open("streamlit/assets/pictures/animals/beluga.jpg")
            st.image(image2, caption="Bélugua", width="stretch")
        except FileNotFoundError:
            st.warning("Image belugua non trouvée. Assurez-vous que le chemin est correct.")
        try:
            audio2 = open("streamlit/assets/songs/animals/beluga.mp3", "rb")
            st.audio(audio2, format='audio/mp3')
        except FileNotFoundError:
            st.warning("Audio bélugua non trouvé. Assurez-vous que le chemin est correct.")
        
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("[Cachalot](https://fr.wikipedia.org/wiki/Grand_cachalot)")
        try:
            image3 = Image.open("streamlit/assets/pictures/animals/sperm_whale.jpg")
            st.image(image3, caption="Sperm Whale", width="stretch")
        except FileNotFoundError:
            st.warning("Image cachalot non trouvée. Assurez-vous que le chemin est correct.")
        try:
            audio3 = open("streamlit/assets/songs/animals/sperm_whale.mp3", "rb")
            st.audio(audio3, format='audio/mp3')
        except FileNotFoundError:
            st.warning("Audio cachalot non trouvé. Assurez-vous que le chemin est correct.")
 
    with col4:
        st.subheader("[Dauphin à flancs blancs](https://fr.wikipedia.org/wiki/Lagenorhynchus_acutus)")
        try:
            image4 = Image.open(
                "streamlit/assets/pictures/animals/white_sided_dolphin.jpg"
            )
            st.image(image4, caption="White sided Dolphin", width="stretch")
        except FileNotFoundError:
            st.warning("Image dauphin non trouvée. Assurez-vous que le chemin est correct.")
        try:
            audio4 = open("streamlit/assets/songs/animals/white_sided_dolphin.mp3", "rb")
            st.audio(audio4, format='audio/mp3')
        except FileNotFoundError:
            st.warning("Audio dauphin non trouvé. Assurez-vous que le chemin est correct.")

    col_empty1, col_center, col_empty2 = st.columns([1, 2, 1])

    with col_center:
        st.subheader("[Rorqual Commun](https://fr.wikipedia.org/wiki/Rorqual_commun)")
        try:
            image5 = Image.open(
                "streamlit/assets/pictures/animals/fin_finback_whale.jpg"
            )
            st.image(image5, caption="Fin finback whale", width="stretch")
        except FileNotFoundError:
            st.warning("Image rorqual commun non trouvée. Assurez-vous que le chemin est correct.")
        try:
            audio5 = open("streamlit/assets/songs/animals/fin_finback_whale.mp3", "rb")
            st.audio(audio5, format='audio/mp3')
        except FileNotFoundError:
            st.warning("Audio rorqual commun non trouvé. Assurez-vous que le chemin est correct.")