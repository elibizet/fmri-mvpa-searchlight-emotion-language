"""scripts/02_within_searchlight_happy_vs_anger.py"""

"""Searchlight analysis for fMRI data using Nilearn. 
This script is an analysis pipeline that performs searchlight analysis on fMRI data within modality"""

"""Searchlight runs a local (neighborhood) multivariate analysis across the brain: for each center voxel it extracts a 
neighborhood (usually spherical), fits a classifier/regressor with cross-validation on data inside that neighborhood, 
and assigns the resulting score to the center voxel. The output is a 3D map of scores.

Key parameters:

    mask_img: brain mask (restricts voxels to evaluate).
    radius: neighborhood radius (in mm); controls spatial extent of each searchlight.
    estimator: sklearn estimator used inside each searchlight (e.g., SVC or LogisticRegression).
    scoring: scoring metric (string or scorer object).
    cv: cross-validation splitter (int, CV object).
    n_jobs: parallel jobs.

Typical flow: instantiate SearchLight → fit(X_imgs, y) → inspect/save score map."""

#from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd

from nilearn.image import load_img, concat_imgs, mean_img
from nilearn.plotting import plot_stat_map
from nilearn.decoding import SearchLight

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler #estimator = make_pipeline(StandardScaler(), LinearSVC(...)) to avoid information leak. And passed that pipeline into SearchLight(..., cv=LeaveOneGroupOut()).
from sklearn.svm import LinearSVC


# In Nilearn’s SearchLight, for each CV split and each sphere:
#the pipeline is fit on training samples only
#scaler parameters (mean/std) are learned on training only
#then applied to test only



# Where betas are stored (read-only)
BETAS_ROOT = Path(r"D:\singleN_betas")

# Where ALL outputs are saved
PROJECT_ROOT = Path(r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis")
OUTDIR = PROJECT_ROOT / "outputs" / "within_recog_betas"

# ============================================================
# Analysis choices
# ============================================================

# Only recognition task, and start with one pair
TASK_FILTER = "recog"  # keep only labels that contain "recog_"
PAIR = ("happiness", "anger")
MODALITIES = ["audio", "video", "audiovisual"]  # run separate searchlight per modality

# Searchlight params
RADIUS_MM = 5.0
N_JOBS = 4
VERBOSE = 1
C_SVM = 1.0

# File naming
BAS_FOLDER = "BAS2"
LABELS_FILE = "regressor_labels.csv"


# ============================================================
# Helpers
# ============================================================

def _read_regressor_labels(labels_path: Path) -> list[str]:
    """
    Your regressor_labels.csv appears to be a 1-column file with regressor names.
    We read the first column, drop NaNs/empty rows.
    """
    df = pd.read_csv(labels_path, header=None)
    col0 = df.iloc[:, 0].astype(str).str.strip()
    col0 = col0[col0.notna() & (col0 != "")]
    return col0.tolist()


def _beta_filename(i: int) -> str:                                  # this assumes regressor_labels.csv is in the same order as SPM’s beta images (beta_0001.nii corresponds to regressor row 1, etc.)
    """SPM-style: beta_0001.nii, beta_0002.nii, ... (1-indexed).""" #align labels to betas by index (row number → beta file number)
    return f"beta_{i:04d}.nii"


def parse_label(label: str) -> dict | None:
    """
    Parses strings like:
      run-3_recog_anger_audio
      run-3_recog_happiness_audiovisual
      run-2_emo_R1   (should be excluded by TASK_FILTER)
    Returns dict with run, task, emotion, modality if it matches recog pattern.
    """
    # Only keep recognition labels
    if f"_{TASK_FILTER}_" not in label:
        return None

    # Expected: run-<n>_recog_<emotion>_<modality>
    m = re.match(r"run-(\d+)_recog_([a-zA-Z]+)_([a-zA-Z]+)", label)
    if not m:
        return None

    run = int(m.group(1))
    emotion = m.group(2).lower()
    modality = m.group(3).lower()

    # Normalize modality naming if needed
    if modality not in {"audio", "video", "audiovisual"}:
        return None

    return {"run": run, "task": "recog", "emotion": emotion, "modality": modality}


def choose_matching_mask(bas_dir: Path, beta_img_path: Path) -> Path:
    """
    There is both _mask.nii and mask.nii. We'll choose the one that matches beta geometry.
    Preference order if both match: _mask.nii then mask.nii.
    """
    candidates = [bas_dir / "_mask.nii", bas_dir / "mask.nii"]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        raise FileNotFoundError(f"No mask found in {bas_dir} (expected _mask.nii or mask.nii)")

    beta_img = load_img(str(beta_img_path))
    beta_shape = beta_img.shape[:3]
    beta_affine = beta_img.affine

    matches = []
    for mpath in candidates:
        mimg = load_img(str(mpath))
        if mimg.shape[:3] == beta_shape and np.allclose(mimg.affine, beta_affine):
            matches.append(mpath)

    if matches:
        # Prefer _mask.nii if it matches
        for pref in [bas_dir / "_mask.nii", bas_dir / "mask.nii"]:
            if pref in matches:
                return pref
        return matches[0]

    # If nothing matches exactly, still return _mask.nii if present (but warn)
    # Better to fail loudly than run wrong geometry.
    raise ValueError(
        f"Neither _mask.nii nor mask.nii matches beta geometry in {bas_dir}.\n"
        f"beta: {beta_shape}\n"
        f"candidates: {[p.name for p in candidates]}"
    )


def make_estimator():
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LinearSVC(C=C_SVM, dual="auto", max_iter=10_000),
    )


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
    )
    sl.fit(X_img, y, groups=groups)
    return sl.scores_


# ============================================================
# Loading betas + labels across all sessions
# ============================================================

def collect_samples(subject: str) -> pd.DataFrame:
    """
    Walks:
      BETAS_ROOT/subject/ses-*/BAS2/
    Reads regressor_labels.csv in each BAS2 folder.
    Maps label row i -> beta_i file (beta_0001.nii, beta_0002.nii, ...).
    Filters to recognition task and parses emotion/modality/run.
    """
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
                # Some pipelines also create _beta_####.nii; we are *not* using those here
                # because labels typically correspond to beta_####.nii in SPM.
                continue

            rows.append(
                {
                    "subject": subject,
                    "session": ses_dir.name,            # ses-01, ses-02...
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
            f"No recognition samples found for {subject}. "
            f"Check that regressor labels include 'run-*_recog_*_*'."
        )
    return df


def build_pair_for_modality(df: pd.DataFrame, modality: str):
    """
    Within-modality: filter to one modality and one emotion pair (happiness vs anger).
    groups: leave-one-group-out across runs *within sessions* using session+run.
    """
    emo_a, emo_b = PAIR
    meta = df[
        (df["modality"] == modality)
        & (df["emotion"].isin([emo_a, emo_b]))
    ].copy()

    if meta.empty:
        raise ValueError(f"No samples for modality={modality} and pair={PAIR}")

    # X: 4D image (one beta per sample)
    X_img = concat_imgs(meta["beta_path"].tolist())

    # y: labels
    y = meta["emotion"].to_numpy()

    # groups: use session change the line below to this: --> groups = meta["session"].to_numpy()
    groups = meta["session"].to_numpy()
    print("CV groups:", sorted(set(groups)))

    # Alternative: combine session+run to ensure runs are unique across sessions
    # groups = (meta["session"].astype(str) + "_run-" + meta["run"].astype(str)).to_numpy()

    return X_img, y, groups, meta


def save_outputs(subject: str, modality: str, scores_img, bg_img, meta: pd.DataFrame, mask_path_used: str):
    emo_a, emo_b = PAIR
    outdir = OUTDIR / subject / f"mod-{modality}"
    outdir.mkdir(parents=True, exist_ok=True)

    stem = f"{subject}_mod-{modality}_pair-{emo_a}-vs-{emo_b}"

    nii_path = outdir / f"{stem}_searchlight-acc.nii.gz"
    png_path = outdir / f"{stem}_searchlight-acc.png"
    log_path = outdir / f"{stem}_runlog.json"

    scores_img.to_filename(nii_path)

    display = plot_stat_map(
        scores_img,
        bg_img=bg_img,
        title=f"{subject} | {modality} | {emo_a} vs {emo_b} (recog)",
        display_mode="z",
        cut_coords=5,
        threshold=None,
    )
    display.savefig(png_path)
    display.close()

    log = {
        "subject": subject,
        "task": "recognition",
        "modality": modality,
        "pair": [emo_a, emo_b],
        "n_samples": int(len(meta)),
        "class_counts": meta["emotion"].value_counts().to_dict(),
        "n_groups_unique": int(pd.Series((meta["session"] + "_run-" + meta["run"].astype(str))).nunique()),
        "radius_mm": RADIUS_MM,
        "classifier": "LinearSVC",
        "C": C_SVM,
        "cv": "LeaveOneGroupOut(session+run)",
        "mask_used": mask_path_used,
        "outputs": {"nii": str(nii_path), "png": str(png_path)},
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"[saved] {nii_path}")
    print(f"[saved] {png_path}")
    print(f"[saved] {log_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="e.g., sub-001")
    return p.parse_args()


def main():
    args = parse_args()
    subject = args.subject

    # 1) Load dataset (all sessions)
    df = collect_samples(subject)

    # Choose a mask based on the first BAS2 folder we encounter
    # (Assumes same geometry across sessions; typical for single-trial betas.)
    first_row = df.iloc[0]
    bas_dir = Path(first_row["bas_dir"])
    first_beta = Path(first_row["beta_path"])
    mask_path = choose_matching_mask(bas_dir, first_beta)
    mask_img = load_img(str(mask_path))
    print(f"[mask] Using: {mask_path}")

    # Run within-modality searchlight for each modality
    for modality in MODALITIES:
        print(f"\n=== {subject} | modality={modality} | task=recognition | pair={PAIR[0]} vs {PAIR[1]} ===")

        X_img, y, groups, meta = build_pair_for_modality(df, modality)

        # Need >=2 groups for LeaveOneGroupOut
        if pd.Series(groups).nunique() < 2:
            raise ValueError(
                f"Not enough unique (session+run) groups for CV in modality={modality}. "
                f"Found {pd.Series(groups).nunique()} groups. "
                f"You need at least 2 runs (or sessions) containing the pair."
            )

        # 3) Searchlight computation
        scores_img = run_searchlight(X_img, y, groups, mask_img)

        # 4) Visualization + saving
        bg_img = mean_img(X_img)
        save_outputs(subject, modality, scores_img, bg_img, meta, str(mask_path))

    print("\nDone.")


if __name__ == "__main__":
    main()
