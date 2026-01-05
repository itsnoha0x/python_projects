import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------
# Chargement des modèles
# --------------------------------------------------------
@st.cache_resource
def load_all():
    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)

    scaler = pickle.load(open("models/scaler.pkl", "rb"))

    models = {
        "Logistic Regression": pickle.load(open("models/Logistic_Regression.pkl", "rb")),
        "KNN": pickle.load(open("models/KNN.pkl", "rb")),
        "Decision Tree": pickle.load(open("models/Decision_Tree.pkl", "rb")),
        "Random Forest": pickle.load(open("models/Random_Forest.pkl", "rb"))
    }

    # For evaluation
    test = pd.read_csv("KDDTest+.txt", header=None)
    # numerical features only
    selected = [
        0,4,5,22,23,24,26,28,31,32
    ]
    X_test = test[selected]
    y_test = (test[41].str.strip().str.lower() != "normal").astype(int)

    X_test = scaler.transform(X_test)

    return models, scaler, metrics, X_test, y_test

models, scaler, metrics, X_test, y_test = load_all()

# --------------------------------------------------------
# Interface
# --------------------------------------------------------
st.title("Détection d’intrusions – NSL-KDD")

mode = st.sidebar.selectbox(
    "Choisir mode",
    ["Évaluation des modèles", "Entrée manuelle", "Upload CSV", "Explications"]
)

# --------------------------------------------------------
# Page 1 : Évaluation des modèles
# --------------------------------------------------------
if mode == "Évaluation des modèles":
    st.header("Évaluation des modèles")

    options = list(models.keys()) + ["Comparaison globale"]
    chosen = st.selectbox("Choisir un modèle", options)

    # ============================
    # CAS 1 : Évaluation d’un modèle unique
    # ============================
    if chosen != "Comparaison globale":
        st.subheader(f"Métriques pour : {chosen}")

        m = metrics[chosen]

        st.write(f"Accuracy : {m['accuracy']:.4f}")
        st.write(f"Precision : {m['precision']:.4f}")
        st.write(f"Recall : {m['recall']:.4f}")
        st.write(f"F1-Score : {m['f1']:.4f}")

        # Matrice de confusion
        st.subheader("Matrice de Confusion")
        cm = np.array(m["confusion_matrix"])

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        st.pyplot(fig)

    # ============================
    # CAS 2 : Comparaison globale des 4 modèles
    # ============================
    else:
        st.subheader("Comparaison des 4 modèles")

        rows = []
        for name in models.keys():
            m = metrics[name]
            rows.append({
                "Modèle": name,
                "Accuracy": float(m["accuracy"]),
                "Precision": float(m["precision"]),
                "Recall": float(m["recall"]),
                "F1-score": float(m["f1"])
            })

        df_compare = pd.DataFrame(rows)

        # Affichage formaté (format appliqué uniquement aux colonnes numériques)
        st.dataframe(
            df_compare.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-score": "{:.4f}"
            })
        )

        # Trouver le meilleur modèle selon le F1-score
        best = df_compare.loc[df_compare["F1-score"].idxmax()]

        st.subheader("Conclusion")
        st.write(f"""
Le modèle le plus performant est : **{best['Modèle']}**

Il obtient un F1-score de **{best['F1-score']:.4f}**,  
ce qui indique qu’il fournit le meilleur équilibre entre :

- précision (qualité des détections)
- rappel (capacité à ne pas rater les attaques)

👉 C’est donc le modèle recommandé pour une détection robuste des intrusions.
""")


# --------------------------------------------------------
# Page 2 : Entrée manuelle
# --------------------------------------------------------
elif mode == "Entrée manuelle":
    st.header("Entrée manuelle des caractéristiques")

    features = ["duration","src_bytes","dst_bytes","count","srv_count",
                "serror_rate","rerror_rate","same_srv_rate",
                "dst_host_count","dst_host_srv_count"]

    values = []

    for f in features:
        val = st.number_input(f, min_value=0.0, value=1.0)
        values.append(val)

    if st.button("Prédire"):
        X = np.array(values).reshape(1,-1)
        X = scaler.transform(X)

        model = models["Random Forest"]
        pred = model.predict(X)[0]

        if pred == 1:
            st.error("Intrusion détectée")
        else:
            st.success("Trafic normal")

# --------------------------------------------------------
# Page 3 : Upload CSV
# --------------------------------------------------------
elif mode == "Upload CSV":
    st.header("Upload d’un fichier CSV")

    file = st.file_uploader("Choisir un fichier CSV")

    if file:
        df = pd.read_csv(file)

        X = df.values
        X = scaler.transform(X)

        model = models["Random Forest"]
        preds = model.predict(X)

        # Affichage ligne par ligne
        pred_df = pd.DataFrame({"Prediction": preds})
        st.subheader("Prédictions ligne par ligne")
        st.write(pred_df)

        # -----------------------------
        #  ANALYSE GLOBALE DU FICHIER
        # -----------------------------
        total = len(preds)
        attacks = np.sum(preds == 1)
        normals = np.sum(preds == 0)

        st.subheader("Analyse globale du fichier")

        st.write(f"- Nombre total de lignes : **{total}**")
        st.write(f"- Trafic normal : **{normals}**")
        st.write(f"- Attaques détectées : **{attacks}**")

        # Conclusion globale
        st.subheader("Conclusion")

        if attacks == 0:
            st.success("Le fichier est probablement **NORMAL** (aucune attaque détectée).")
        elif attacks < total * 0.3:
            st.warning("Le fichier contient quelques anomalies → activité **suspecte**.")
        else:
            st.error("Le fichier est probablement **MALVEILLANT** (forte présence d’attaques).")


# --------------------------------------------------------
# Page 4 : Explications
# --------------------------------------------------------
elif mode == "Explications":
    st.header("📘 Explications du projet")

    st.write("""
Ce projet utilise le dataset **NSL-KDD**, un ensemble de données de référence en cybersécurité,
pour entraîner quatre algorithmes de Machine Learning destinés à la détection d'intrusions. 🔐

---

## 1. Régression Logistique ⚙️
La régression logistique est un **modèle linéaire de classification binaire**.
Elle estime la probabilité qu’un trafic soit normal ou malveillant à partir des caractéristiques réseau.

- Produit une probabilité entre 0 et 1  
- Basée sur la fonction sigmoïde  
- Rapide, stable et efficace sur des données structurées  
- Interprétation simple

---

## 2. KNN — K-Nearest Neighbors 🤝
KNN classe un nouvel échantillon en regardant ses **K voisins les plus similaires** dans le dataset.

- Pas d’apprentissage direct (lazy learning)  
- Repose sur la distance → importance de la standardisation  
- Très intuitif  
- Performant lorsque les données sont bien distribuées

---

## 3. Arbre de Décision 🌳
L’arbre de décision construit un ensemble de **règles conditionnelles** sous forme de branches.

- Très facile à interpréter  
- Capture naturellement des relations non linéaires  
- Peut sur-apprendre si non régulé  
- Fonctionne bien sur des jeux de données tabulaires

---

## 4. Random Forest 🌲🌲
Random Forest est un ensemble de plusieurs arbres de décision entraînés de façon indépendante.

- Réduit fortement le sur-apprentissage  
- Plus stable qu’un seul arbre  
- Gère bien les données bruitées  
- Très utilisé en détection d’anomalies

---

## Standardisation des données (Normalization) 📏
Nous utilisons `StandardScaler` avant l’entraînement :

- moyenne = 0  
- écart-type = 1  
- met toutes les variables sur la même échelle  

Cela améliore la stabilité du modèle et est **essentiel** pour KNN et utile pour les autres algorithmes.

---

## Structure de l’application 🖥️
L’application propose :

- Une page d’**évaluation des modèles**  
- Une page d’**entrée manuelle**  
- Une page d’**upload CSV**  
- Une page d’**explications pédagogiques**

Elle permet ainsi de tester, comparer et utiliser un système d’apprentissage automatique
pour la détection d’intrusions réseau.
""")
