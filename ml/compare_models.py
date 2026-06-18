from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


DATA_DIR = Path("../outputs")
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def mistag_at_eff(df: pd.DataFrame, score_col: str, target_eff: float = 0.70):
    y = df["is_b"].values
    scores = df[score_col].values

    thr = np.quantile(scores[y == 1], 1.0 - target_eff)
    pred = scores >= thr

    sample = df["sample_label"].values
    b_eff = np.mean(pred[y == 1])

    c_mask = sample == "c"
    light_mask = np.isin(sample, ["g", "uds"])

    c_mistag = np.mean(pred[c_mask]) if np.any(c_mask) else np.nan
    light_mistag = np.mean(pred[light_mask]) if np.any(light_mask) else np.nan

    return b_eff, c_mistag, light_mistag


def main():
    bdt = pd.read_csv(DATA_DIR / "test_scores_bdt.csv")
    dnn = pd.read_csv(DATA_DIR / "test_scores_dnn.csv")

    # same ordering assumed because both come from test.parquet
    df = bdt.copy()
    df["score_dnn"] = dnn["score_dnn"]

    y = df["is_b"].values

    auc_bdt = roc_auc_score(y, df["score_bdt"])
    auc_dnn = roc_auc_score(y, df["score_dnn"])

    fpr_bdt, tpr_bdt, _ = roc_curve(y, df["score_bdt"])
    fpr_dnn, tpr_dnn, _ = roc_curve(y, df["score_dnn"])

    plt.figure(figsize=(7, 6))
    plt.plot(tpr_bdt, 1.0 / np.clip(fpr_bdt, 1e-6, None), label=f"BDT (AUC={auc_bdt:.3f})")
    plt.plot(tpr_dnn, 1.0 / np.clip(fpr_dnn, 1e-6, None), label=f"DNN (AUC={auc_dnn:.3f})")
    plt.yscale("log")
    plt.xlabel("b-jet efficiency")
    plt.ylabel("Background rejection = 1 / mistag")
    plt.title("b-tagging comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_rejection.png", dpi=200)

    for score_col in ["score_bdt", "score_dnn"]:
        b_eff, c_mistag, light_mistag = mistag_at_eff(df, score_col, target_eff=0.70)
        print(f"\nModel: {score_col}")
        print(f"  b-eff @ threshold:     {b_eff:.4f}")
        print(f"  c mistag:              {c_mistag:.4f}")
        print(f"  light mistag:          {light_mistag:.4f}")
        print(f"  c rejection:           {(1.0/c_mistag) if c_mistag > 0 else np.inf:.2f}")
        print(f"  light rejection:       {(1.0/light_mistag) if light_mistag > 0 else np.inf:.2f}")


if __name__ == "__main__":
    main()