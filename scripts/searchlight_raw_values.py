"""
Searchlight analysis for fMRI data using Nilearn. 
This script is an analysis pipeline that performs 
searchlight analysis on fMRI data within modality
StandardScaler removed completely explicit meta 
sorting before concat_imgs, y, and groups to ensure 
deterministic sample order.
Created 2026/04/07 for use in the master thesis.
"""

# run : python searchlight_raw_values.py --subject sub-001

from pathlib import Path
import argparse
import json
import re
import time

import numpy as np
import pandas as pd

from nilearn.image import load_img, concat_imgs, mean_img, new_img_like
from nilearn.plotting import plot_stat_map
from nilearn.decoding import SearchLight

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import LinearSVC


# PATHS (EDIT IF NEEDED)

BETAS_ROOT = Path(r"D:\singleN_betas")

PROJECT_ROOT = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\Results\New searchlight raw results\happiness vs anger (8 mm radius)"
)
OUTDIR = PROJECT_ROOT   # e.g., OUTDIR / "mod-video", OUTDIR / "mod-audio", etc.

BAS_FOLDER = "BAS2"
LABELS_FILE = "regressor_labels.csv"


# ANALYSIS CHOICES
TASK_FILTER = "recog"   # change to "recog" if needed
PAIR = ("happiness", "anger")
MODALITIES = ["audio"]    # e.g. ["video"], ["audio"], ["audiovisual"]

RADIUS_MM = 8.0
N_JOBS = 4
VERBOSE = 1
C_SVM = 1.0


# HELPERS

def _read_regressor_labels(labels_path: Path) -> list[str]:
    df = pd.read_csv(labels_path, header=None)
    col0 = df.iloc[:, 0].astype(str).str.strip()
    col0 = col0[col0.notna() & (col0 != "")]
    return col0.tolist()


def _beta_filename(i: int) -> str:
    return f"beta_{i:04d}.nii"


def parse_label(label: str) -> dict | None:
    """
    Expected format:
    run-<n>_<task>_<emotion>_<modality>

    Example:
    run-1_passive_happiness_video
    """
    if f"_{TASK_FILTER}_" not in label:
        return None

    m = re.match(rf"run-(\d+)_{TASK_FILTER}_([a-zA-Z]+)_([a-zA-Z]+)", label)
    if not m:
        return None

    run = int(m.group(1))
    emotion = m.group(2).lower()
    modality = m.group(3).lower()

    if modality not in {"audio", "video", "audiovisual"}:
        return None

    return {
        "run": run,
        "task": TASK_FILTER,
        "emotion": emotion,
        "modality": modality,
    }


def choose_matching_mask(bas_dir: Path, beta_img_path: Path) -> Path:
    candidates = [bas_dir / "_mask.nii", bas_dir / "mask.nii"]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        raise FileNotFoundError(
            f"No mask found in {bas_dir} (expected _mask.nii or mask.nii)"
        )

    beta_img = load_img(str(beta_img_path))
    beta_shape = beta_img.shape[:3]
    beta_affine = beta_img.affine

    matches = []
    for mpath in candidates:
        mimg = load_img(str(mpath))
        if mimg.shape[:3] == beta_shape and np.allclose(mimg.affine, beta_affine):
            matches.append(mpath)

    if matches:
        for pref in [bas_dir / "_mask.nii", bas_dir / "mask.nii"]:
            if pref in matches:
                return pref
        return matches[0]

    raise ValueError(
        f"Neither _mask.nii nor mask.nii matches beta geometry in {bas_dir}.\n"
        f"beta shape: {beta_shape}\n"
        f"candidates: {[p.name for p in candidates]}"
    )


def make_estimator():
    """
    Final estimator with NO StandardScaler.
    """
    return LinearSVC(C=C_SVM, dual="auto", max_iter=10_000)


def run_searchlight(X_img, y, groups, mask_img):
    cv = LeaveOneGroupOut()

    sl = SearchLight(
        mask_img=mask_img,
        process_mask_img=mask_img,
        radius=RADIUS_MM,
        estimator=make_estimator(),
        cv=cv,
        n_jobs=N_JOBS,
        verbose=VERBOSE,
        scoring="accuracy",
    )

    sl.fit(X_img, y, groups=groups)
    return sl.scores_  # 3D numpy array


# DATA COLLECTION

def collect_samples(subject: str) -> pd.DataFrame:
    subj_dir = BETAS_ROOT / subject
    if not subj_dir.exists():
        raise FileNotFoundError(f"Subject folder not found: {subj_dir}")

    rows = []

    for ses_dir in sorted(subj_dir.glob("ses-*")):
        bas_dir = ses_dir / BAS_FOLDER
        if not bas_dir.exists():
            continue

        labels_path = bas_dir / LABELS_FILE
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing {LABELS_FILE} in {bas_dir}")

        labels = _read_regressor_labels(labels_path)

        for idx, label in enumerate(labels, start=1):
            info = parse_label(label)
            if info is None:
                continue

            beta_name = _beta_filename(idx)
            beta_path = bas_dir / beta_name
            if not beta_path.exists():
                continue

            rows.append(
                {
                    "subject": subject,
                    "session": ses_dir.name,
                    "bas_dir": str(bas_dir),
                    "label_raw": label,
                    "run": info["run"],
                    "emotion": info["emotion"],
                    "modality": info["modality"],
                    "beta_name": beta_name,
                    "beta_path": str(beta_path),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No {TASK_FILTER} samples found for {subject}. "
            f"Check that labels include 'run-*_{TASK_FILTER}_*_*'."
        )

    return df


def build_pair_for_modality(df: pd.DataFrame, modality: str):
    emo_a, emo_b = PAIR

    meta = df[
        (df["modality"] == modality) &
        (df["emotion"].isin([emo_a, emo_b]))
    ].copy()

    if meta.empty:
        raise ValueError(f"No samples for modality={modality} and pair={PAIR}")

    # Explicit sorting for deterministic sample order
    meta = meta.sort_values(
        ["session", "run", "emotion", "beta_name"]
    ).reset_index(drop=True)

    X_img = concat_imgs(meta["beta_path"].tolist())
    y = meta["emotion"].to_numpy()

    # leave-one-session-out
    groups = meta["session"].to_numpy()

    return X_img, y, groups, meta


# OUTPUTS

def save_outputs(subject: str, modality: str, scores_arr: np.ndarray, bg_img, meta: pd.DataFrame, mask_img):
    emo_a, emo_b = PAIR
    outdir = OUTDIR / subject / f"mod-{modality}"
    outdir.mkdir(parents=True, exist_ok=True)

    stem = f"{subject}_mod-{modality}_pair-{emo_a}-vs-{emo_b}"
    nii_path = outdir / f"{stem}_searchlight-acc.nii.gz"
    png_path = outdir / f"{stem}_searchlight-acc.png"
    log_path = outdir / f"{stem}_runlog.json"
    meta_path = outdir / f"{stem}_design_labels.csv"

    scores_arr = np.nan_to_num(scores_arr, nan=0.0, posinf=0.0, neginf=0.0)

    scores_nii = new_img_like(mask_img, scores_arr)
    scores_nii.to_filename(nii_path)

    display = plot_stat_map(
        scores_nii,
        bg_img=bg_img,
        title=f"{subject} | {modality} | {emo_a} vs {emo_b} ({TASK_FILTER})",
        display_mode="z",
        cut_coords=5,
        threshold=None,
    )
    display.savefig(png_path)
    display.close()

    meta.to_csv(meta_path, index=False)

    log = {
        "subject": subject,
        "task": TASK_FILTER,
        "modality": modality,
        "pair": [emo_a, emo_b],
        "n_samples": int(len(meta)),
        "class_counts": meta["emotion"].value_counts().to_dict(),
        "n_sessions": int(pd.Series(meta["session"]).nunique()),
        "radius_mm": RADIUS_MM,
        "classifier": "LinearSVC",
        "C": C_SVM,
        "standardization": "none",
        "cv": "LeaveOneGroupOut(session)",
        "n_jobs": N_JOBS,
        "outputs": {
            "nii": str(nii_path),
            "png": str(png_path),
            "design_labels_csv": str(meta_path),
        },
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"[saved] {nii_path}")
    print(f"[saved] {png_path}")
    print(f"[saved] {meta_path}")
    print(f"[saved] {log_path}")


# MAIN

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="e.g., sub-001")
    return p.parse_args()


def main():
    t0 = time.time()

    args = parse_args()
    subject = args.subject

    df = collect_samples(subject)

    first_row = df.iloc[0]
    bas_dir = Path(first_row["bas_dir"])
    first_beta = Path(first_row["beta_path"])

    mask_path = choose_matching_mask(bas_dir, first_beta)
    mask_img = load_img(str(mask_path))
    print(f"[mask] Using: {mask_path}")

    for modality in MODALITIES:
        print(
            f"\n=== {subject} | modality={modality} | "
            f"task={TASK_FILTER} | pair={PAIR[0]} vs {PAIR[1]} ==="
        )

        X_img, y, groups, meta = build_pair_for_modality(df, modality)

        n_sessions = pd.Series(groups).nunique()
        print(
            f"[info] n_samples={len(meta)} | "
            f"n_sessions={n_sessions} | n_jobs={N_JOBS} | radius_mm={RADIUS_MM}"
        )
        print(f"[info] class counts = {meta['emotion'].value_counts().to_dict()}")

        if n_sessions < 2:
            raise ValueError(
                f"Not enough unique session groups for CV in modality={modality}. "
                f"Found {n_sessions} session(s)."
            )

        t_mod = time.time()

        scores_arr = run_searchlight(X_img, y, groups, mask_img)
        # emergency backup save immediately after computation
        scores_arr = np.nan_to_num(scores_arr, nan=0.0, posinf=0.0, neginf=0.0)

        backup_dir = OUTDIR / subject / f"mod-{modality}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_npy = backup_dir / "backup_scores.npy"
        np.save(backup_npy, scores_arr)
        print(f"[saved backup] {backup_npy}")
        
        bg_img = mean_img(X_img, copy_header=True)
        save_outputs(subject, modality, scores_arr, bg_img, meta, mask_img)

        print(f"[time] modality={modality} took {(time.time() - t_mod)/3600:.2f} hours")

    print(f"\nDone - Total runtime: {(time.time() - t0)/3600:.2f} hours")


if __name__ == "__main__":
    main()