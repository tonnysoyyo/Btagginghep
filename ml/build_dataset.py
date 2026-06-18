from __future__ import annotations

import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

# Change this if needed
# Example 1: Path("../data")
# Example 2: Path("../rootfiles")
DATA_DIR = Path("../data")

OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If you want CSV instead of parquet, change to "csv"
SAVE_FORMAT = "parquet"   # "parquet" or "csv"

SAMPLE_MAP = {
    "QCD_bbbar": "b",
    "QCD_ccbar": "c",
    "QCD_gg": "g",
    "QCD_uds": "uds",
}

BRANCHES = [
    "Jet.PT",
    "Jet.Eta",
    "Jet.Phi",
    "Jet.Mass",
    "Jet.NCharged",
    "Jet.NNeutrals",
    "Track.PT",
    "Track.Eta",
    "Track.Phi",
    "Track.Mass",
    "Track.Charge",
    "Track.D0",
    "Track.DZ",
]

JET_CONE = 0.4
MAX_JETS_PER_EVENT = 4
SEED = 42


# ============================================================
# Small helpers
# ============================================================

def delta_phi(phi1: np.ndarray, phi2: float) -> np.ndarray:
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def delta_r(jet_eta: float, jet_phi: float, trk_eta: np.ndarray, trk_phi: np.ndarray) -> np.ndarray:
    deta = trk_eta - jet_eta
    dphi = delta_phi(trk_phi, jet_phi)
    return np.sqrt(deta**2 + dphi**2)


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) > 0 else 0.0


def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if len(x) > 0 else 0.0


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    if len(x) == 0 or np.sum(w) <= 0:
        return 0.0
    return float(np.average(x, weights=w))


def first_radius(sorted_dr: np.ndarray, cum_frac: np.ndarray, target: float) -> float:
    mask = cum_frac >= target
    if np.any(mask):
        return float(sorted_dr[np.argmax(mask)])
    return 0.0


def save_dataframe(df: pd.DataFrame, path_stem: Path) -> Path:
    if SAVE_FORMAT == "parquet":
        out_path = path_stem.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
    elif SAVE_FORMAT == "csv":
        out_path = path_stem.with_suffix(".csv")
        df.to_csv(out_path, index=False)
    else:
        raise ValueError("SAVE_FORMAT must be 'parquet' or 'csv'")
    return out_path


# ============================================================
# Feature extraction
# ============================================================

def compute_jet_features(
    jet_pt: float,
    jet_eta: float,
    jet_phi: float,
    jet_mass: float,
    jet_ncharged: float,
    jet_nneutral: float,
    trk_pt: np.ndarray,
    trk_eta: np.ndarray,
    trk_phi: np.ndarray,
    trk_d0: np.ndarray,
    trk_dz: np.ndarray,
) -> dict:
    dr = delta_r(jet_eta, jet_phi, trk_eta, trk_phi)
    mask = dr < JET_CONE

    trk_pt = trk_pt[mask]
    trk_eta = trk_eta[mask]
    trk_phi = trk_phi[mask]
    trk_d0 = trk_d0[mask]
    trk_dz = trk_dz[mask]
    dr = dr[mask]

    n_tracks = len(trk_pt)

    if n_tracks == 0:
        return {
            "jet_pt": jet_pt,
            "jet_eta": jet_eta,
            "jet_abs_eta": abs(jet_eta),
            "jet_mass": jet_mass,
            "jet_ncharged": jet_ncharged,
            "jet_nneutral": jet_nneutral,
            "n_tracks_cone": 0,
            "track_pt_sum": 0.0,
            "track_pt_sum_over_jet_pt": 0.0,
            "avg_track_pt": 0.0,
            "std_track_pt": 0.0,
            "n_tracks_below_avg_pt": 0,
            "n_tracks_above_avg_pt": 0,
            "max_pt_ratio": 0.0,
            "min_pt_ratio": 0.0,
            "max_dr_ratio": 0.0,
            "min_dr_ratio": 0.0,
            "dr_max_pt": 0.0,
            "dr_min_pt": 0.0,
            "dr_max_dr": 0.0,
            "dr_min_dr": 0.0,
            "pt_difference": 0.0,
            "r50": 0.0,
            "r95": 0.0,
            "mean_abs_d0": 0.0,
            "max_abs_d0": 0.0,
            "std_abs_d0": 0.0,
            "mean_abs_dz": 0.0,
            "max_abs_dz": 0.0,
            "std_abs_dz": 0.0,
            "pt_weighted_abs_d0": 0.0,
            "pt_weighted_abs_dz": 0.0,
        }

    track_pt_sum = float(np.sum(trk_pt))
    avg_track_pt = float(np.mean(trk_pt))
    std_track_pt = float(np.std(trk_pt))

    n_tracks_below_avg_pt = int(np.sum(trk_pt < avg_track_pt))
    n_tracks_above_avg_pt = int(np.sum(trk_pt >= avg_track_pt))

    idx_max_pt = int(np.argmax(trk_pt))
    idx_min_pt = int(np.argmin(trk_pt))
    idx_max_dr = int(np.argmax(dr))
    idx_min_dr = int(np.argmin(dr))

    max_pt_ratio = float(trk_pt[idx_max_pt] / jet_pt) if jet_pt > 0 else 0.0
    min_pt_ratio = float(trk_pt[idx_min_pt] / jet_pt) if jet_pt > 0 else 0.0
    max_dr_ratio = float(trk_pt[idx_max_dr] / jet_pt) if jet_pt > 0 else 0.0
    min_dr_ratio = float(trk_pt[idx_min_dr] / jet_pt) if jet_pt > 0 else 0.0

    dr_max_pt = float(dr[idx_max_pt])
    dr_min_pt = float(dr[idx_min_pt])
    dr_max_dr = float(dr[idx_max_dr])
    dr_min_dr = float(dr[idx_min_dr])

    pt_difference = float(trk_pt[idx_max_pt] - trk_pt[idx_min_pt])

    order = np.argsort(dr)
    sorted_dr = dr[order]
    sorted_pt = trk_pt[order]

    cum_pt = np.cumsum(sorted_pt)
    cum_frac = cum_pt / cum_pt[-1]

    r50 = first_radius(sorted_dr, cum_frac, 0.50)
    r95 = first_radius(sorted_dr, cum_frac, 0.95)

    abs_d0 = np.abs(trk_d0)
    abs_dz = np.abs(trk_dz)

    return {
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "jet_abs_eta": abs(jet_eta),
        "jet_mass": jet_mass,
        "jet_ncharged": jet_ncharged,
        "jet_nneutral": jet_nneutral,
        "n_tracks_cone": n_tracks,
        "track_pt_sum": track_pt_sum,
        "track_pt_sum_over_jet_pt": (track_pt_sum / jet_pt) if jet_pt > 0 else 0.0,
        "avg_track_pt": avg_track_pt,
        "std_track_pt": std_track_pt,
        "n_tracks_below_avg_pt": n_tracks_below_avg_pt,
        "n_tracks_above_avg_pt": n_tracks_above_avg_pt,
        "max_pt_ratio": max_pt_ratio,
        "min_pt_ratio": min_pt_ratio,
        "max_dr_ratio": max_dr_ratio,
        "min_dr_ratio": min_dr_ratio,
        "dr_max_pt": dr_max_pt,
        "dr_min_pt": dr_min_pt,
        "dr_max_dr": dr_max_dr,
        "dr_min_dr": dr_min_dr,
        "pt_difference": pt_difference,
        "r50": r50,
        "r95": r95,
        "mean_abs_d0": safe_mean(abs_d0),
        "max_abs_d0": float(np.max(abs_d0)),
        "std_abs_d0": safe_std(abs_d0),
        "mean_abs_dz": safe_mean(abs_dz),
        "max_abs_dz": float(np.max(abs_dz)),
        "std_abs_dz": safe_std(abs_dz),
        "pt_weighted_abs_d0": weighted_mean(abs_d0, trk_pt),
        "pt_weighted_abs_dz": weighted_mean(abs_dz, trk_pt),
    }


# ============================================================
# File processing
# ============================================================

def collect_all_files(data_dir: Path) -> list[tuple[Path, str]]:
    items = []

    for folder, label in SAMPLE_MAP.items():
        root_files = sorted((data_dir / folder).glob("*.root"))
        if not root_files:
            print(f"Warning: no ROOT files found in {data_dir / folder}")
            continue

        for rf in root_files:
            items.append((rf, label))

    return items


def process_one_file(root_file: Path, sample_label: str) -> pd.DataFrame:
    rows = []
    event_counter = 0

    for batch in uproot.iterate(f"{root_file}:Delphes", BRANCHES, step_size="100 MB", library="ak"):
        n_events = len(batch["Jet.PT"])

        for ievt in range(n_events):
            jet_pt = np.asarray(ak.to_numpy(batch["Jet.PT"][ievt]), dtype=np.float32)
            jet_eta = np.asarray(ak.to_numpy(batch["Jet.Eta"][ievt]), dtype=np.float32)
            jet_phi = np.asarray(ak.to_numpy(batch["Jet.Phi"][ievt]), dtype=np.float32)
            jet_mass = np.asarray(ak.to_numpy(batch["Jet.Mass"][ievt]), dtype=np.float32)
            jet_ncharged = np.asarray(ak.to_numpy(batch["Jet.NCharged"][ievt]), dtype=np.float32)
            jet_nneutral = np.asarray(ak.to_numpy(batch["Jet.NNeutrals"][ievt]), dtype=np.float32)

            trk_pt = np.asarray(ak.to_numpy(batch["Track.PT"][ievt]), dtype=np.float32)
            trk_eta = np.asarray(ak.to_numpy(batch["Track.Eta"][ievt]), dtype=np.float32)
            trk_phi = np.asarray(ak.to_numpy(batch["Track.Phi"][ievt]), dtype=np.float32)
            trk_d0 = np.asarray(ak.to_numpy(batch["Track.D0"][ievt]), dtype=np.float32)
            trk_dz = np.asarray(ak.to_numpy(batch["Track.DZ"][ievt]), dtype=np.float32)

            n_jets = min(len(jet_pt), MAX_JETS_PER_EVENT)

            for j in range(n_jets):
                feats = compute_jet_features(
                    jet_pt=float(jet_pt[j]),
                    jet_eta=float(jet_eta[j]),
                    jet_phi=float(jet_phi[j]),
                    jet_mass=float(jet_mass[j]),
                    jet_ncharged=float(jet_ncharged[j]),
                    jet_nneutral=float(jet_nneutral[j]),
                    trk_pt=trk_pt,
                    trk_eta=trk_eta,
                    trk_phi=trk_phi,
                    trk_d0=trk_d0,
                    trk_dz=trk_dz,
                )

                feats["sample_label"] = sample_label
                feats["is_b"] = 1 if sample_label == "b" else 0
                feats["root_file"] = str(root_file)
                feats["event_in_file"] = event_counter
                feats["jet_rank"] = j
                feats["global_event_id"] = f"{root_file}::evt::{event_counter}"
                rows.append(feats)

            event_counter += 1

    return pd.DataFrame(rows)


# ============================================================
# Splitting
# ============================================================

def split_by_event(full_df: pd.DataFrame):
    event_df = full_df[["global_event_id", "sample_label"]].drop_duplicates().reset_index(drop=True)

    # First split: train vs temp
    train_events, temp_events = train_test_split(
        event_df,
        test_size=0.30,
        random_state=SEED,
        stratify=event_df["sample_label"],
    )

    # Second split: val vs test
    val_events, test_events = train_test_split(
        temp_events,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_events["sample_label"],
    )

    train_df = full_df[full_df["global_event_id"].isin(train_events["global_event_id"])].copy()
    val_df = full_df[full_df["global_event_id"].isin(val_events["global_event_id"])].copy()
    test_df = full_df[full_df["global_event_id"].isin(test_events["global_event_id"])].copy()

    return train_df, val_df, test_df, train_events, val_events, test_events


# ============================================================
# Main
# ============================================================

def main():
    items = collect_all_files(DATA_DIR)

    if len(items) == 0:
        raise RuntimeError(f"No ROOT files found under {DATA_DIR}")

    all_dfs = []

    for root_file, label in items:
        print(f"Processing {root_file.name} as {label}")
        df = process_one_file(root_file, label)
        print(f"  -> {len(df)} jets")
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    print("\nTotal jets in full dataset:", len(full_df))
    print("\nJets per class:")
    print(full_df["sample_label"].value_counts())

    train_df, val_df, test_df, train_events, val_events, test_events = split_by_event(full_df)

    print("\nSplit summary:")
    print(f"Train jets: {len(train_df)}")
    print(f"Val jets:   {len(val_df)}")
    print(f"Test jets:  {len(test_df)}")

    print("\nTrain class counts:")
    print(train_df["sample_label"].value_counts())

    print("\nVal class counts:")
    print(val_df["sample_label"].value_counts())

    print("\nTest class counts:")
    print(test_df["sample_label"].value_counts())

    train_path = save_dataframe(train_df, OUTPUT_DIR / "train")
    val_path = save_dataframe(val_df, OUTPUT_DIR / "val")
    test_path = save_dataframe(test_df, OUTPUT_DIR / "test")

    manifest = {
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "save_format": SAVE_FORMAT,
        "seed": SEED,
        "jet_cone": JET_CONE,
        "max_jets_per_event": MAX_JETS_PER_EVENT,
        "files": [{"file": str(f), "label": label} for f, label in items],
        "n_full_jets": int(len(full_df)),
        "n_train_jets": int(len(train_df)),
        "n_val_jets": int(len(val_df)),
        "n_test_jets": int(len(test_df)),
        "n_train_events": int(len(train_events)),
        "n_val_events": int(len(val_events)),
        "n_test_events": int(len(test_events)),
        "train_output": str(train_path),
        "val_output": str(val_path),
        "test_output": str(test_path),
    }

    with open(OUTPUT_DIR / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nSaved:")
    print(" -", train_path)
    print(" -", val_path)
    print(" -", test_path)
    print(" -", OUTPUT_DIR / "dataset_manifest.json")
    print("\nDone.")


if __name__ == "__main__":
    main()