import pandas as pd
import numpy as np
import pickle
import json

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------------
# 1. Charger NSL-KDD
# ---------------------------------------------
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

df_train = pd.read_csv("KDDTrain+.txt", names=columns)
df_test  = pd.read_csv("KDDTest+.txt", names=columns)

# ---------------------------------------------
# 2. Encode Label (normal = 0, attack = 1)
# ---------------------------------------------
df_train["label"] = (df_train["label"].str.strip().str.lower() != "normal").astype(int)
df_test["label"]  = (df_test["label"].str.strip().str.lower() != "normal").astype(int)

# ---------------------------------------------
# 3. Sélection simple de features numériques
# ---------------------------------------------
selected_features = [
    "duration","src_bytes","dst_bytes","count","srv_count",
    "serror_rate","rerror_rate","same_srv_rate",
    "dst_host_count","dst_host_srv_count"
]

X_train = df_train[selected_features]
y_train = df_train["label"]

X_test = df_test[selected_features]
y_test = df_test["label"]

# ---------------------------------------------
# 4. Scaling
# ---------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pickle.dump(scaler, open("models/scaler.pkl", "wb"))

# ---------------------------------------------
# 5. Training des 4 modèles
# ---------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=80)
}

metrics = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    # Save model
    pickle.dump(model, open(f"models/{name.replace(' ', '_')}.pkl", "wb"))

    # Predictions
    pred = model.predict(X_test_scaled)

    # Metrics simples
    metrics[name] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }

# Save Metrics JSON
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training terminé. Modèles et métriques sauvegardés.")
