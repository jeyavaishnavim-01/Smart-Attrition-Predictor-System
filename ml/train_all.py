import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hr_data.csv"
MDL_DIR   = ROOT / "models"
MDL_DIR.mkdir(exist_ok=True)

CAT_COLS = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime",
]
TARGET = "Attrition"


def save(name, obj):
    with open(MDL_DIR / name, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved → models/{name}")


def header(title):
    print(f"\n{'─'*50}\n  {title}\n{'─'*50}")


# ----- STEP 1: Preprocessing -----#
def preprocess():
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

    header("STEP 1 — Preprocessing")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    target_le        = LabelEncoder()
    df[TARGET]       = target_le.fit_transform(df[TARGET])
    encoders["target"] = target_le

    X         = df.drop(columns=[TARGET])
    y         = df[TARGET].values
    feat_cols = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    save("label_encoders.pkl", encoders)
    save("scaler.pkl",         scaler)
    save("feature_columns.pkl", feat_cols)

    print(f"Done — Train: {X_tr_sc.shape}  Test: {X_te_sc.shape}")
    return X_tr_sc, X_te_sc, y_tr, y_te, feat_cols


#-----STEP 2: scikit-learn-----#
def train_sklearn(X_tr, X_te, y_tr, y_te, feat_cols):
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, classification_report
    )

    header("STEP 2 — scikit-learn Ensemble")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05,
        max_depth=4, random_state=42,
    )
    lr = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft", weights=[3, 2, 1],
    )
    ensemble.fit(X_tr, y_tr)

    y_pred = ensemble.predict(X_te)
    y_prob = ensemble.predict_proba(X_te)[:, 1]
    acc    = round(accuracy_score(y_te, y_pred), 4)
    auc    = round(roc_auc_score(y_te, y_prob), 4)
    print(f"  Accuracy : {acc}  |  ROC-AUC : {auc}")

    fi_raw = dict(zip(feat_cols, ensemble.estimators_[0].feature_importances_))
    fi_top = dict(sorted(fi_raw.items(), key=lambda x: x[1], reverse=True)[:15])

    metrics = {
        "accuracy":           acc,
        "roc_auc":            auc,
        "report":             classification_report(y_te, y_pred, target_names=["Stay", "Leave"]),
        "feature_importance": fi_top,
    }

    save("sklearn_model.pkl",   ensemble)
    save("sklearn_metrics.pkl", metrics)
    print("scikit-learn done")
    return metrics


#-----STEP 3: TensorFlow-----#
def train_tensorflow(X_tr, X_te, y_tr, y_te):
    import tensorflow as tf
    from tensorflow.keras import layers, callbacks, regularizers
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    header("STEP 3 — TensorFlow ANN")

    model = tf.keras.Sequential([
        layers.Input(shape=(X_tr.shape[1],)),
        layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Dropout(0.40),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Dropout(0.30),
        layers.Dense(64,  activation="relu"), layers.Dropout(0.20),
        layers.Dense(32,  activation="relu"),
        layers.Dense(1,   activation="sigmoid"),
    ], name="AttritionANN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    n_neg, n_pos = np.sum(y_tr == 0), np.sum(y_tr == 1)
    class_weight = {0: 1.0, 1: round(n_neg / n_pos, 2)}
    print(f"  Class weights: {class_weight}")

    cb = [
        callbacks.EarlyStopping(
            monitor="val_auc", patience=15,
            restore_best_weights=True, mode="max",
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6,
        ),
    ]

    print("  Training (up to 100 epochs)...")
    history = model.fit(
        X_tr, y_tr,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=cb,
        verbose=0,
    )
    print(f"  Stopped at epoch {len(history.history['loss'])}")

    y_prob = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    acc    = round(accuracy_score(y_te, y_pred), 4)
    auc    = round(roc_auc_score(y_te, y_prob), 4)
    print(f"  Accuracy : {acc}  |  ROC-AUC : {auc}")

    model.save(MDL_DIR / "tf_model")
    print(f"Saved → models/tf_model/")

    metrics = {
        "accuracy":       acc,
        "roc_auc":        auc,
        "report":         classification_report(y_te, y_pred, target_names=["Stay", "Leave"]),
        "epochs_trained": len(history.history["loss"]),
    }
    save("tf_metrics.pkl", metrics)
    print("TensorFlow done")
    return metrics


#-----STEP 4: PyTorch-----#
def train_pytorch(X_tr, X_te, y_tr, y_te):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    header("STEP 4 — PyTorch AttritionNet")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {DEVICE}")

    class AttritionNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.40),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.30),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.20),
                nn.Linear(64, 32),   nn.GELU(),
                nn.Linear(32, 1),    nn.Sigmoid(),
            )
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.net(x)

    counts  = np.bincount(y_tr.astype(int))
    weights = 1.0 / counts[y_tr.astype(int)]
    sampler = WeightedRandomSampler(
        torch.DoubleTensor(weights), len(y_tr), replacement=True
    )

    def make_loader(X, y, sampler=None, shuffle=False):
        ds = TensorDataset(
            torch.FloatTensor(X).to(DEVICE),
            torch.FloatTensor(y).unsqueeze(1).to(DEVICE),
        )
        return DataLoader(ds, batch_size=32, sampler=sampler, shuffle=shuffle)

    train_loader = make_loader(X_tr, y_tr, sampler=sampler)
    test_loader  = make_loader(X_te, y_te, shuffle=False)

    model     = AttritionNet(X_tr.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    criterion = nn.BCELoss()

    best_auc, best_state = 0.0, None
    patience, no_improve = 15, 0

    print("  Training (up to 80 epochs)...")

    for epoch in range(1, 81):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                probs.extend(model(xb).cpu().numpy().flatten())
                labels.extend(yb.cpu().numpy().flatten())

        val_auc = roc_auc_score(labels, probs)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            probs.extend(model(xb).cpu().numpy().flatten())
            labels.extend(yb.cpu().numpy().flatten())

    y_pred = (np.array(probs) >= 0.5).astype(int)
    acc    = round(accuracy_score(labels, y_pred), 4)
    auc    = round(roc_auc_score(labels, probs), 4)
    print(f"  Accuracy : {acc}  |  ROC-AUC : {auc}")

    torch.save(
        {"model_state_dict": model.state_dict(), "input_dim": X_tr.shape[1]},
        MDL_DIR / "pytorch_model.pth",
    )
    print(f"Saved → models/pytorch_model.pth")

    metrics = {
        "accuracy": acc,
        "roc_auc":  auc,
        "report":   classification_report(labels, y_pred, target_names=["Stay", "Leave"]),
    }
    save("pytorch_metrics.pkl", metrics)
    print("PyTorch done")
    return metrics


#-----MAIN-----#
if __name__ == "__main__":
    print("\n" + "═"*50)
    print("  HR ATTRITION — Training Pipeline")
    print("═"*50)

    X_tr, X_te, y_tr, y_te, feat_cols = preprocess()
    sk_m = train_sklearn(X_tr, X_te, y_tr, y_te, feat_cols)
    tf_m = train_tensorflow(X_tr, X_te, y_tr, y_te)
    pt_m = train_pytorch(X_tr, X_te, y_tr, y_te)

    print("\n" + "═"*50)
    print("  RESULTS")
    print("═"*50)
    print(f"  scikit-learn  →  Acc: {sk_m['accuracy']}  AUC: {sk_m['roc_auc']}")
    print(f"  TensorFlow    →  Acc: {tf_m['accuracy']}  AUC: {tf_m['roc_auc']}")
    print(f"  PyTorch       →  Acc: {pt_m['accuracy']}  AUC: {pt_m['roc_auc']}")






""" 
HR ATTRITION — Training Pipeline
══════════════════════════════════════════
              RESULTS
══════════════════════════════════════════
  scikit-learn  →  Acc: 0.9267  AUC: 0.9820
  TensorFlow    →  Acc: 0.8813  AUC: 0.9612
  PyTorch       →  Acc: 0.8880  AUC: 0.9578
"""
