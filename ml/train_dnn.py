from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path("../outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_EFF = 0.70
BATCH_SIZE = 1024
EPOCHS = 50
PATIENCE = 8

META_COLS = {"sample_label", "is_b", "root_file", "event_in_file", "jet_rank"}


class MLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def get_features(df: pd.DataFrame):
    return [c for c in df.columns if c not in META_COLS]


def mistag_report(df: pd.DataFrame, scores: np.ndarray, target_eff: float = 0.70):
    y = df["is_b"].values
    score_b = scores[y == 1]
    threshold = np.quantile(score_b, 1.0 - target_eff)

    pred = scores >= threshold
    sample = df["sample_label"].values

    b_eff = float(np.mean(pred[y == 1]))
    c_mistag = float(np.mean(pred[sample == "c"])) if np.any(sample == "c") else np.nan
    light_mask = np.isin(sample, ["g", "uds"])
    light_mistag = float(np.mean(pred[light_mask])) if np.any(light_mask) else np.nan

    return {
        "threshold": float(threshold),
        "b_eff": b_eff,
        "c_mistag": c_mistag,
        "light_mistag": light_mistag,
        "c_rejection": (1.0 / c_mistag) if c_mistag > 0 else np.inf,
        "light_rejection": (1.0 / light_mistag) if light_mistag > 0 else np.inf,
    }


def main():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    features = get_features(train)

    X_train = train[features].values.astype(np.float32)
    y_train = train["is_b"].values.astype(np.float32)
    X_val = val[features].values.astype(np.float32)
    y_val = val["is_b"].values.astype(np.float32)
    X_test = test[features].values.astype(np.float32)
    y_test = test["is_b"].values.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MLP(n_in=X_train.shape[1]).to(DEVICE)

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_auc = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_logits = []
        val_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                val_logits.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(yb.numpy())

        val_scores = np.concatenate(val_logits)
        val_true = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_true, val_scores)

        print(f"Epoch {epoch+1:02d} | train loss={np.mean(train_losses):.5f} | val AUC={val_auc:.5f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_scores = torch.sigmoid(model(torch.tensor(X_test).to(DEVICE))).cpu().numpy()

    test_auc = roc_auc_score(y_test, test_scores)
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Test AUC           : {test_auc:.4f}")

    report = mistag_report(test, test_scores, target_eff=TARGET_EFF)
    print(json.dumps(report, indent=2))

    torch.save(model.state_dict(), DATA_DIR / "dnn_model.pt")
    joblib.dump(scaler, DATA_DIR / "dnn_scaler.joblib")
    joblib.dump(features, DATA_DIR / "dnn_features.joblib")

    out_scores = test.copy()
    out_scores["score_dnn"] = test_scores
    out_scores.to_csv(DATA_DIR / "test_scores_dnn.csv", index=False)

    print("Saved DNN model and test scores.")


if __name__ == "__main__":
    main()