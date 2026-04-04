import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os

# ============================================================
# CONFIGURATION
# ============================================================
API_URL = os.getenv("API_URL", "https://bankchurn-1-982n.onrender.com").rstrip("/")

st.set_page_config(
    page_title="🏦 Bank Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
@st.cache_data(ttl=30)
def check_api_health() -> dict | None:
    """Vérifie l'état de l'API."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except requests.ConnectionError:
        return None


@st.cache_data(ttl=60)
def get_metadata() -> dict | None:
    """Récupère les métadonnées du modèle."""
    try:
        resp = requests.get(f"{API_URL}/metadata", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except requests.ConnectionError:
        return None


def predict_single(features: dict) -> dict | None:
    """Envoie une prédiction unitaire."""
    try:
        resp = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=10)
        return resp.json() if resp.status_code == 200 else {"error": resp.text}
    except requests.ConnectionError:
        return None


def predict_batch(instances: list[dict]) -> dict | None:
    """Envoie une prédiction batch."""
    try:
        resp = requests.post(f"{API_URL}/predict/batch", json={"instances": instances}, timeout=30)
        return resp.json() if resp.status_code == 200 else {"error": resp.text}
    except requests.ConnectionError:
        return None


def render_gauge(probability: float, title: str) -> go.Figure:
    """Crée une jauge de probabilité."""
    color = "green" if probability < 0.3 else ("orange" if probability < 0.6 else "red")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#d4edda"},
                {"range": [30, 60], "color": "#fff3cd"},
                {"range": [60, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30))
    return fig


def render_confusion_matrix(cm: list[list[int]], title: str) -> go.Figure:
    """Crée un heatmap de la matrice de confusion."""
    labels = ["Fidèle (0)", "Churné (1)"]
    fig = px.imshow(
        cm,
        x=labels, y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Prédit", y="Réel", color="Nombre"),
    )
    fig.update_layout(title=title, height=350, margin=dict(t=50, b=30))
    return fig


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=80)
    st.title("🏦 Bank Churn")
    st.markdown("---")

    # Statut de l'API
    health = check_api_health()
    if health and health.get("model_loaded"):
        st.success("✅ API en ligne — Modèle chargé")
    elif health:
        st.warning("⚠️ API en ligne — Modèle non chargé")
    else:
        st.error("❌ API hors ligne")
        st.info("Lancez l'API avec :\n```\nuvicorn src.mlops_tp.api:app\n```")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "🔮 Prédiction unitaire", "📊 Prédiction batch", "📈 Métriques du modèle"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("MLOps TP — ISIMA 2026")


# ============================================================
# PAGE : ACCUEIL
# ============================================================
if page == "🏠 Accueil":
    st.title("🏦 Prédiction de Désabonnement Client")
    st.markdown("""
    **Bienvenue** sur le tableau de bord de prédiction de churn bancaire.

    Cette application permet de :
    - 🔮 **Prédire** si un client va quitter la banque
    - 📊 **Analyser en batch** plusieurs clients à la fois
    - 📈 **Consulter** les performances du modèle

    ---
    """)

    # Métadonnées du modèle
    meta = get_metadata()
    if meta:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Version du modèle", meta.get("model_version", "—"))
        with col2:
            st.metric("Type de tâche", meta.get("task_type", "—").capitalize())
        with col3:
            trained = meta.get("trained_at", "—")
            if trained and trained != "—":
                trained = trained[:19].replace("T", " ")
            st.metric("Entraîné le", trained)

        st.markdown("#### 📋 Features utilisées")
        features = meta.get("features", {})
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Variables numériques**")
            for f in features.get("numeric_features", []):
                st.markdown(f"- `{f}`")
        with col_b:
            st.markdown("**Variables catégorielles**")
            for f in features.get("categorical_features", []):
                st.markdown(f"- `{f}`")
    else:
        st.warning("Impossible de charger les métadonnées. Vérifiez que l'API est en ligne.")


# ============================================================
# PAGE : PRÉDICTION UNITAIRE
# ============================================================
elif page == "🔮 Prédiction unitaire":
    st.title("🔮 Prédiction Unitaire")
    st.markdown("Remplissez les informations du client pour prédire le risque de désabonnement.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📌 Profil")
        geography = st.selectbox("Pays", ["France", "Germany", "Spain"])
        gender = st.selectbox("Genre", ["Female", "Male"])
        age = st.slider("Âge", min_value=18, max_value=92, value=42)

    with col2:
        st.subheader("💳 Compte")
        credit_score = st.number_input("Score de crédit", min_value=300, max_value=900, value=619)
        balance = st.number_input("Solde du compte (€)", min_value=0.0, max_value=300000.0, value=0.0, step=1000.0)
        estimated_salary = st.number_input("Salaire estimé (€)", min_value=0.0, max_value=250000.0, value=101348.88, step=1000.0)

    with col3:
        st.subheader("📦 Produits")
        tenure = st.slider("Ancienneté (années)", min_value=0, max_value=10, value=2)
        num_products = st.selectbox("Nombre de produits", [1, 2, 3, 4], index=0)
        has_credit_card = st.selectbox("Carte de crédit", ["Oui", "Non"])
        is_active = st.selectbox("Membre actif", ["Oui", "Non"])

    st.markdown("---")

    features = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_credit_card == "Oui" else 0,
        "IsActiveMember": 1 if is_active == "Oui" else 0,
        "EstimatedSalary": estimated_salary,
    }

    if st.button("🚀 Prédire", type="primary", use_container_width=True):
        with st.spinner("Prédiction en cours..."):
            result = predict_single(features)

        if result is None:
            st.error("❌ Impossible de contacter l'API.")
        elif "error" in result:
            st.error(f"Erreur : {result['error']}")
        else:
            prediction = result["prediction"]
            proba = result.get("proba", {})
            latency = result.get("latency_ms", 0)

            st.markdown("---")

            # Résultat principal
            if prediction == "1":
                st.error("## ⚠️ Client à risque — Churn probable")
            else:
                st.success("## ✅ Client fidèle — Pas de churn prévu")

            # Probabilités et jauge
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                churn_prob = proba.get("Yes", 0) if proba else 0
                fig = render_gauge(churn_prob, "Probabilité de Churn")
                st.plotly_chart(fig, use_container_width=True)
            with col_r2:
                st.markdown("#### 📊 Détails")
                if proba:
                    st.markdown(f"- **Probabilité churn** : `{proba.get('Yes', 0):.2%}`")
                    st.markdown(f"- **Probabilité fidèle** : `{proba.get('No', 0):.2%}`")
                st.markdown(f"- **Latence** : `{latency:.1f} ms`")
                st.markdown(f"- **Version modèle** : `{result.get('model_version', '—')}`")

                # Résumé du client
                st.markdown("#### 👤 Résumé client")
                summary_df = pd.DataFrame([features]).T
                summary_df.columns = ["Valeur"]
                st.dataframe(summary_df, use_container_width=True)


# ============================================================
# PAGE : PRÉDICTION BATCH
# ============================================================
elif page == "📊 Prédiction batch":
    st.title("📊 Prédiction Batch")
    st.markdown("Chargez un fichier CSV pour prédire le churn sur plusieurs clients.")
    st.markdown("---")

    # Template CSV à télécharger
    template_df = pd.DataFrame([{
        "CreditScore": 619, "Geography": "France", "Gender": "Female",
        "Age": 42, "Tenure": 2, "Balance": 0.0, "NumOfProducts": 1,
        "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88,
    }])

    st.download_button(
        "📥 Télécharger le template CSV",
        data=template_df.to_csv(index=False),
        file_name="template_churn.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("📁 Charger un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")
            st.stop()

        st.markdown(f"**{len(df)} clients** chargés.")
        st.dataframe(df.head(10), use_container_width=True)

        # Supprimer les colonnes non-features si présentes
        cols_to_drop = [c for c in ["RowNumber", "CustomerId", "Surname", "Exited"] if c in df.columns]
        if cols_to_drop:
            st.info(f"Colonnes ignorées automatiquement : {', '.join(cols_to_drop)}")
            df = df.drop(columns=cols_to_drop)

        if st.button("🚀 Lancer les prédictions", type="primary", use_container_width=True):
            instances = df.to_dict(orient="records")

            with st.spinner(f"Prédiction de {len(instances)} clients en cours..."):
                result = predict_batch(instances)

            if result is None:
                st.error("❌ Impossible de contacter l'API.")
            elif "error" in result:
                st.error(f"Erreur : {result['error']}")
            else:
                predictions = result["predictions"]
                latency = result.get("latency_ms", 0)

                # Ajouter les résultats au DataFrame
                df["Prédiction"] = [p["prediction"] for p in predictions]
                df["Prédiction_label"] = df["Prédiction"].map({"0": "✅ Fidèle", "1": "⚠️ Churn"})
                df["Proba_Churn"] = [p["proba"]["Yes"] if p.get("proba") else None for p in predictions]
                df["Proba_Fidèle"] = [p["proba"]["No"] if p.get("proba") else None for p in predictions]

                st.markdown("---")

                # KPI
                n_churn = (df["Prédiction"] == "1").sum()
                n_total = len(df)
                n_fidele = n_total - n_churn

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total clients", n_total)
                c2.metric("✅ Fidèles", n_fidele)
                c3.metric("⚠️ Churn", n_churn)
                c4.metric("Taux de churn", f"{n_churn / n_total:.1%}")

                st.markdown("---")

                # Graphiques
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    fig_pie = px.pie(
                        names=["Fidèle", "Churn"],
                        values=[n_fidele, n_churn],
                        color_discrete_sequence=["#28a745", "#dc3545"],
                        title="Répartition des prédictions",
                        hole=0.4,
                    )
                    fig_pie.update_layout(height=350)
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_g2:
                    if df["Proba_Churn"].notna().any():
                        fig_hist = px.histogram(
                            df, x="Proba_Churn", nbins=30,
                            title="Distribution des probabilités de churn",
                            color_discrete_sequence=["#007bff"],
                            labels={"Proba_Churn": "Probabilité de churn"},
                        )
                        fig_hist.update_layout(height=350)
                        st.plotly_chart(fig_hist, use_container_width=True)

                # Tableau de résultats
                st.markdown("#### 📋 Résultats détaillés")
                st.dataframe(
                    df.style.apply(
                        lambda row: ["background-color: #f8d7da" if row["Prédiction"] == "1"
                                     else "background-color: #d4edda"] * len(row),
                        axis=1
                    ),
                    use_container_width=True,
                    height=400,
                )

                # Télécharger les résultats
                st.download_button(
                    "📥 Télécharger les résultats (CSV)",
                    data=df.to_csv(index=False),
                    file_name="resultats_churn.csv",
                    mime="text/csv",
                )

                st.caption(f"⏱️ Latence : {latency:.1f} ms — 📦 {result.get('count', '—')} prédictions")


# ============================================================
# PAGE : MÉTRIQUES DU MODÈLE
# ============================================================
elif page == "📈 Métriques du modèle":
    st.title("📈 Métriques du Modèle")
    st.markdown("Performances du modèle sur les ensembles de validation et de test.")
    st.markdown("---")

    meta = get_metadata()
    if meta:
        c1, c2 = st.columns(2)
        c1.metric("Version", meta.get("model_version", "—"))
        trained = meta.get("trained_at", "—")
        if trained and trained != "—":
            trained = trained[:19].replace("T", " ")
        c2.metric("Entraîné le", trained)

    # Charger les métriques depuis le fichier local
    try:
        from src.mlops_tp.config import METRICS_PATH
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    except Exception:
        st.warning("Impossible de charger les métriques. Vérifiez que le modèle est entraîné.")
        st.stop()

    # Hyperparamètres
    st.markdown("### ⚙️ Hyperparamètres")
    hyper = metrics.get("Hyperparameters", {})
    cols_h = st.columns(len(hyper))
    for col, (k, v) in zip(cols_h, hyper.items()):
        col.metric(k, str(v))

    st.markdown("---")

    # Métriques Validation vs Test
    for split_name in ["Validation", "Test"]:
        split_data = metrics.get(split_name)
        if not split_data:
            continue

        st.markdown(f"### 📊 {split_name}")

        # KPI
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{split_data['accuracy']:.4f}")
        c2.metric("F1 Score (weighted)", f"{split_data['f1_score']:.4f}")
        roc = split_data.get("roc_auc")
        c3.metric("ROC AUC", f"{roc:.4f}" if roc else "N/A")

        col_m1, col_m2 = st.columns(2)

        # Matrice de confusion
        with col_m1:
            cm = split_data.get("confusion_matrix", [])
            if cm:
                fig_cm = render_confusion_matrix(cm, f"Matrice de confusion — {split_name}")
                st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report
        with col_m2:
            report = split_data.get("classification_report", {})
            if report:
                # Extraire les métriques par classe
                classes = ["0", "1"]
                report_data = []
                for cls in classes:
                    if cls in report:
                        row = report[cls]
                        report_data.append({
                            "Classe": f"{'Fidèle' if cls == '0' else 'Churné'} ({cls})",
                            "Précision": f"{row['precision']:.4f}",
                            "Recall": f"{row['recall']:.4f}",
                            "F1-Score": f"{row['f1-score']:.4f}",
                            "Support": int(row['support']),
                        })

                st.markdown(f"**Classification Report — {split_name}**")
                st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

                # Bar chart comparatif
                metrics_names = ["precision", "recall", "f1-score"]
                fig_bar = go.Figure()
                for cls in classes:
                    if cls in report:
                        vals = [report[cls][m] for m in metrics_names]
                        fig_bar.add_trace(go.Bar(
                            name=f"{'Fidèle' if cls == '0' else 'Churné'} ({cls})",
                            x=["Précision", "Recall", "F1-Score"],
                            y=vals,
                            text=[f"{v:.3f}" for v in vals],
                            textposition="auto",
                        ))
                fig_bar.update_layout(
                    barmode="group", height=300,
                    title=f"Métriques par classe — {split_name}",
                    yaxis=dict(range=[0, 1.05]),
                    margin=dict(t=50, b=30),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

