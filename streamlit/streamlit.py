import streamlit as st
from PIL import Image

from pages.home import show_home
from pages.performances import show_performances
from pages.songs_exemples import show_song_exemples

# Configuration de la page
logo_directory = "streamlit/assets/pictures/logo.png"
try:
    img = Image.open(logo_directory)
except Exception:
    img = None

st.set_page_config(
    page_title="IA Hack 2026",
    page_icon=img,
    layout="wide"
)

# Style CSS
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Centrer le titre H1 */
    h1 {
        text-align: center;
        margin-top: -10px;
    }

    /* Centrer la barre de navigation (onglets) */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 30px;
    }

    /* Forcer le centrage de l'image à l'intérieur de sa colonne */
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        margin-top: 15px;
    }

    /* Réduire l'espace blanc en haut de la page */
    .block-container {
        padding-top: 1.5rem;
    }         
    </style>
    """, unsafe_allow_html=True)

# En-tête
col_side1, col_logo, col_side2 = st.columns([2, 1, 2])

with col_logo:
    if img:
        st.image(img, width=200)
    else:
        st.write("*(Logo introuvable)*")

st.title("Hackathon IA'Hack 2026")

# Navigation
tab1, tab2, tab3 = st.tabs(["🏠 Accueil", "🔊 Exemples sonors", "📊 Performances"])

with tab1:
    show_home()

with tab2:
    show_song_exemples()

with tab3:
    show_performances()