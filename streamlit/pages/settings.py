import streamlit as st
from config import Config
from src.utils.training import Trainer


def _apply_config_from_ui():
    payload = {
        "general": {
            "AUDIO_SAMPLE_RATE": st.session_state["cfg_AUDIO_SAMPLE_RATE"],
            "AUDIO_N_MFCC": st.session_state["cfg_AUDIO_N_MFCC"],
            "P1_AUDIO_DURATION": st.session_state["cfg_P1_AUDIO_DURATION"],
            "WINDOW_SIZE_SEC": st.session_state["cfg_WINDOW_SIZE_SEC"],
            "HOP_SIZE_SEC": st.session_state["cfg_HOP_SIZE_SEC"],
            "RANDOM_STATE": st.session_state["cfg_RANDOM_STATE"],
            "CONFIDENCE_THRESHOLD": st.session_state["cfg_CONFIDENCE_THRESHOLD"],
            "ENERGY_THRESHOLD": st.session_state["cfg_ENERGY_THRESHOLD"],
            "MIN_DETECTION_SEC": st.session_state["cfg_MIN_DETECTION_SEC"],
        },
        "models": {
            "Random Forest": {
                "n_estimators": st.session_state["rf_n_estimators"],
                "max_depth": st.session_state["rf_max_depth"],
                "min_samples_leaf": st.session_state["rf_min_samples_leaf"],
                "class_weight": st.session_state["rf_class_weight"],
                "n_jobs": st.session_state["rf_n_jobs"],
            },
            "Support Vector Machine": {
                "C": st.session_state["svm_C"],
                "gamma": st.session_state["svm_gamma"],
                "kernel": st.session_state["svm_kernel"],
                "probability": st.session_state["svm_probability"],
                "class_weight": st.session_state["svm_class_weight"],
            },
            "Gradient Boosting": {
                "n_estimators": st.session_state["gb_n_estimators"],
                "learning_rate": st.session_state["gb_learning_rate"],
                "max_depth": st.session_state["gb_max_depth"],
            },
            "Logistic Regression": {
                "max_iter": st.session_state["lr_max_iter"],
                "class_weight": st.session_state["lr_class_weight"],
            },
            "KNN": {
                "n_neighbors": st.session_state["knn_n_neighbors"],
                "algorithm": st.session_state["knn_algorithm"],
                "n_jobs": st.session_state["knn_n_jobs"],
            },
        },
    }

    Config.apply_runtime_overrides(payload)


def show_settings():
    st.markdown("### Paramètres & Entraînement")
    st.write(
        "Modifiez les valeurs de Config et les hyperparamètres des modèles, puis lancez un entraînement."
    )

    with st.form("config_form"):
        st.markdown("#### Paramètres généraux")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.number_input(
                "AUDIO_SAMPLE_RATE",
                min_value=8000,
                max_value=96000,
                step=1000,
                value=Config.AUDIO_SAMPLE_RATE,
                key="cfg_AUDIO_SAMPLE_RATE",
            )
            st.number_input(
                "AUDIO_N_MFCC",
                min_value=5,
                max_value=40,
                step=1,
                value=Config.AUDIO_N_MFCC,
                key="cfg_AUDIO_N_MFCC",
            )
            st.number_input(
                "P1_AUDIO_DURATION",
                min_value=1.0,
                max_value=20.0,
                step=0.5,
                value=float(Config.P1_AUDIO_DURATION),
                key="cfg_P1_AUDIO_DURATION",
            )

        with col2:
            st.number_input(
                "WINDOW_SIZE_SEC",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                value=float(Config.WINDOW_SIZE_SEC),
                key="cfg_WINDOW_SIZE_SEC",
            )
            st.number_input(
                "HOP_SIZE_SEC",
                min_value=0.05,
                max_value=5.0,
                step=0.05,
                value=float(Config.HOP_SIZE_SEC),
                key="cfg_HOP_SIZE_SEC",
            )
            st.number_input(
                "RANDOM_STATE",
                min_value=0,
                max_value=9999,
                step=1,
                value=Config.RANDOM_STATE,
                key="cfg_RANDOM_STATE",
            )

        with col3:
            st.number_input(
                "CONFIDENCE_THRESHOLD",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=float(Config.CONFIDENCE_THRESHOLD),
                key="cfg_CONFIDENCE_THRESHOLD",
            )
            st.number_input(
                "ENERGY_THRESHOLD",
                min_value=0.0,
                max_value=1.0,
                step=0.001,
                value=float(Config.ENERGY_THRESHOLD),
                key="cfg_ENERGY_THRESHOLD",
            )
            st.number_input(
                "MIN_DETECTION_SEC",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                value=float(Config.MIN_DETECTION_SEC),
                key="cfg_MIN_DETECTION_SEC",
            )

        st.markdown("#### Random Forest")
        rf1, rf2, rf3 = st.columns(3)
        rf_params = Config.get_model_params("Random Forest")
        rf1.number_input(
            "n_estimators",
            min_value=10,
            max_value=2000,
            step=10,
            value=rf_params["n_estimators"],
            key="rf_n_estimators",
        )
        rf2.number_input(
            "max_depth (0 = None)",
            min_value=0,
            max_value=100,
            step=1,
            value=rf_params["max_depth"],
            key="rf_max_depth",
        )
        rf3.number_input(
            "min_samples_leaf",
            min_value=1,
            max_value=50,
            step=1,
            value=rf_params["min_samples_leaf"],
            key="rf_min_samples_leaf",
        )

        rf4, rf5 = st.columns(2)
        rf4.selectbox(
            "class_weight", options=["balanced", None], index=0, key="rf_class_weight"
        )
        rf5.number_input(
            "n_jobs",
            min_value=-1,
            max_value=32,
            step=1,
            value=rf_params["n_jobs"],
            key="rf_n_jobs",
        )

        st.markdown("#### Support Vector Machine")
        svm1, svm2, svm3, svm4 = st.columns(4)
        svm_params = Config.get_model_params("Support Vector Machine")
        svm1.number_input(
            "C",
            min_value=0.1,
            max_value=1000.0,
            step=0.1,
            value=float(svm_params["C"]),
            key="svm_C",
        )
        svm2.selectbox("gamma", options=["scale", "auto"], index=0, key="svm_gamma")
        svm3.selectbox(
            "kernel",
            options=["rbf", "linear", "poly", "sigmoid"],
            index=0,
            key="svm_kernel",
        )
        svm4.checkbox(
            "probability", value=bool(svm_params["probability"]), key="svm_probability"
        )
        st.selectbox(
            "svm class_weight",
            options=["balanced", None],
            index=0,
            key="svm_class_weight",
        )

        st.markdown("#### Gradient Boosting")
        gb1, gb2, gb3 = st.columns(3)
        gb_params = Config.get_model_params("Gradient Boosting")
        gb1.number_input(
            "gb n_estimators",
            min_value=10,
            max_value=2000,
            step=10,
            value=gb_params["n_estimators"],
            key="gb_n_estimators",
        )
        gb2.number_input(
            "gb learning_rate",
            min_value=0.01,
            max_value=1.0,
            step=0.01,
            value=float(gb_params["learning_rate"]),
            key="gb_learning_rate",
        )
        gb3.number_input(
            "gb max_depth",
            min_value=1,
            max_value=20,
            step=1,
            value=gb_params["max_depth"],
            key="gb_max_depth",
        )

        st.markdown("#### Logistic Regression & KNN")
        lr_col, knn_col = st.columns(2)

        lr_params = Config.get_model_params("Logistic Regression")
        with lr_col:
            st.number_input(
                "lr max_iter",
                min_value=100,
                max_value=10000,
                step=100,
                value=lr_params["max_iter"],
                key="lr_max_iter",
            )
            st.selectbox(
                "lr class_weight",
                options=["balanced", None],
                index=0,
                key="lr_class_weight",
            )

        knn_params = Config.get_model_params("KNN")
        with knn_col:
            st.number_input(
                "knn n_neighbors",
                min_value=1,
                max_value=50,
                step=1,
                value=knn_params["n_neighbors"],
                key="knn_n_neighbors",
            )
            st.selectbox(
                "knn algorithm",
                options=["auto", "ball_tree", "kd_tree", "brute"],
                index=0,
                key="knn_algorithm",
            )
            st.number_input(
                "knn n_jobs",
                min_value=-1,
                max_value=32,
                step=1,
                value=knn_params["n_jobs"],
                key="knn_n_jobs",
            )

        submitted = st.form_submit_button("💾 Appliquer les paramètres")

    if submitted:
        _apply_config_from_ui()
        st.cache_resource.clear()
        st.success("Paramètres appliqués à Config pour cette session.")

    st.divider()
    st.markdown("#### Entraînement")

    part = st.radio("Jeu à entraîner", ["Partie 1", "Partie 2"], horizontal=True)

    model_options = list(Config.MODEL_PARAMS.keys())

    if part == "Partie 1":
        selected_models = st.multiselect(
            "Modèles à entraîner",
            options=model_options,
            default=model_options,
        )

        if st.button("🚀 Lancer le training P1", width="stretch"):
            _apply_config_from_ui()
            with st.spinner("Entraînement P1 en cours..."):
                summary = Trainer.train_part1(selected_models)
                st.cache_resource.clear()

            st.success("Training P1 terminé.")
            st.json(summary)

    else:
        selected_model = st.selectbox(
            "Modèle P2 à entraîner",
            options=model_options,
            index=0,
        )

        if st.button("🚀 Lancer le training P2", width="stretch"):
            _apply_config_from_ui()
            with st.spinner("Entraînement P2 en cours..."):
                summary = Trainer.train_part2(selected_model)
                st.cache_resource.clear()

            st.success("Training P2 terminé.")
            st.json(summary)
