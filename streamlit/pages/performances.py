import json
from pathlib import Path

import pandas as pd

import streamlit as st
from config import Config


def _load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_class_table(report_dict: dict) -> pd.DataFrame:
    rows = []
    for label, metrics in report_dict.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        if isinstance(metrics, dict):
            rows.append(
                {
                    "Classe": label,
                    "Precision": round(metrics.get("precision", 0.0), 3),
                    "Recall": round(metrics.get("recall", 0.0), 3),
                    "F1": round(metrics.get("f1-score", 0.0), 3),
                    "Support": int(metrics.get("support", 0)),
                }
            )
    return pd.DataFrame(rows)


def _model_card(title: str, report_path: Path, cm_path: Path):
    report = _load_report(report_path)

    st.markdown(
        f"""
        <div style="
            margin-bottom:12px;
        ">
            <h4 style="margin-top:0; margin-bottom:8px;">{title}</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if report is None:
        st.warning(f"Rapport introuvable : {report_path.name}")
        return

    accuracy = report.get("accuracy", 0.0)
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    weighted_f1 = report.get("weighted avg", {}).get("f1-score", 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy:.3f}")
    c2.metric("Macro F1", f"{macro_f1:.3f}")
    c3.metric("Weighted F1", f"{weighted_f1:.3f}")

    if cm_path.exists():
        st.image(str(cm_path), width="stretch")
    else:
        st.info("Matrice de confusion introuvable")

    with st.expander("Voir le détail par classe"):
        df = _build_class_table(report)
        if not df.empty:
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("Aucune donnée détaillée disponible.")


def show_performances():
    st.markdown("### Performances du Modèle")
    st.write("Vue compacte des résultats sauvegardés.")

    p1_model_dir = Path(Config.DATASET_PATH_P1) / Config.PATH_MODEL
    p2_model_dir = Path(Config.DATASET_PATH_P2) / Config.PATH_MODEL

    st.markdown("## Partie 1")

    p1_models = [
        "Random Forest",
        "Support Vector Machine",
        "Gradient Boosting",
        "Logistic Regression",
        "KNN",
    ]

    cols = st.columns(2)
    for i, model_name in enumerate(p1_models):
        report_path = p1_model_dir / f"report_{model_name}.json"
        cm_path = p1_model_dir / f"confusion_matrix_{model_name}.png"

        with cols[i % 2]:
            _model_card(model_name, report_path, cm_path)

    st.divider()
    st.markdown("## Partie 2")

    cols_p2 = st.columns(2)

    with cols_p2[0]:
        _model_card(
            "Random Forest (P2)",
            p2_model_dir / "report_Random Forest_p2.json",
            p2_model_dir / "confusion_matrix_Random Forest_p2.png",
        )
