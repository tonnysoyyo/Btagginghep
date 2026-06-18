from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier


DATA_DIR = Path("../outputs")
MODEL_DIR = DATA_DIR
TARGET_EFF = 0.70

META_COLS = {"sample_label", "is_b", "root_file", "event_in_file", "jet_rank"}


def load_data():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    return train, val, test


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
    train, val, test = load_data()
    features = get_features(train)

    X_train = train[features].values
    y_train = train["is_b"].values
    X_val = val[features].values
    y_val = val["is_b"].values
    X_test = test[features].values
    y_test = test["is_b"].values

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    val_auc = roc_auc_score(y_val, val_scores)
    test_auc = roc_auc_score(y_test, test_scores)

    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Test AUC      : {test_auc:.4f}")

    report = mistag_report(test, test_scores, target_eff=TARGET_EFF)
    print(json.dumps(report, indent=2))

    model.save_model(MODEL_DIR / "bdt_model.json")
    joblib.dump(features, MODEL_DIR / "bdt_features.joblib")

    feat_imp = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(MODEL_DIR / "bdt_feature_importance.csv", index=False)

    out_scores = test.copy()
    out_scores["score_bdt"] = test_scores
    out_scores.to_csv(MODEL_DIR / "test_scores_bdt.csv", index=False)

    print("Saved BDT model and test scores.")


if __name__ == "__main__":
    main()