"""
single_axial_slice_searchlight.py

Within-modality searchlight MVPA (recog only) for single-trial beta maps.
- Labels come from regressor_labels.csv (1-column list of regressor names)
- Samples = beta maps (beta_0001.nii, beta_0002.nii, ...)
- Features = voxel values within each searchlight sphere
- CV = LeaveOneGroupOut, grouping by SESSION (leave-one-session-out)

TEST MODE (recommended while debugging):
- Restrict search to a single axial slice to make runs finish quickly
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd

from nilearn.image import load_img, concat_imgs, mean_img, new_img_like
from nilearn.plotting import plot_stat_map
from nilearn.decoding import SearchLight

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# PATHS (EDIT IF NEEDED)
# ============================================================

# Where betas are stored (read-only)
BETAS_ROOT = Path(r"D:\singleN_betas")

# Where ALL outputs are saved
PROJECT_ROOT = Path(r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis")
OUTDIR = PROJECT_ROOT / "outputs" / "within_recog_betas"

# ============================================================
# ANALYSIS CHOICES
# ============================================================

# Only recognition task, and start with one pair
TASK_FILTER = "recog"  # keep only labels that contain "_recog_"
PAIR = ("happiness", "anger")
MODALITIES = ["audio", "video", "audiovisual"]  # run separate searchlight per modality

# Searchlight params
RADIUS_MM = 5.0
N_JOBS = 4          # use 4 cores for stability while testing
VERBOSE = 1
C_SVM = 1.0

# File naming
BAS_FOLDER = "BAS2"
LABELS_FILE = "regressor_labels.csv"

# ============================================================
# TEST MODE (slice restriction)
# ============================================================
TEST_MODE = True              # True = one slice only; False = full mask
TEST_SLICE_Z = None           # None = middle slice; or set an int like 45


# ============================================================
# HELPERS
# ============================================================

def _read_regressor_labels(labels_path: Path) -> list[str]:
    """regressor_labels.csv is a 1-column list of regressor names."""
    df = pd.read_csv(labels_path, header=None)
    col0 = df.iloc[:, 0].astype(str).str.strip()
    col0 = col0[col0.notna() & (col0 != "")]
    return col0.tolist()


def _beta_filename(i: int) -> str:
    """SPM-style: beta_0001.nii, beta_0002.nii, ... (1-indexed)."""
    return f"beta_{i:04d}.nii"


def parse_label(label: str) -> dict | None:
    """
    Expected pattern: run-<n>_recog_<emotion>_<modality>
    Example: run-3_recog_anger_audio
    """
    if f"_{TASK_FILTER}_" not in label:
        return None

    m = re.match(r"run-(\d+)_recog_([a-zA-Z]+)_([a-zA-Z]+)", label)
    if not m:
        return None

    run = int(m.group(1))
    emotion = m.group(2).lower()
    modality = m.group(3).lower()

    if modality not in {"audio", "video", "audiovisual"}:
        return None

    return {"run": run, "task": "recog", "emotion": emotion, "modality": modality}


def choose_matching_mask(bas_dir: Path, beta_img_path: Path) -> Path:
    """
    Choose between _mask.nii and mask.nii based on geometry match to beta image.
    Preference: _mask.nii if both match.
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
    """Pipeline prevents leakage: scaler is fit only on training data within each CV fold."""
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
    return sl.scores_  # <-- numpy ndarray (3D)


# ============================================================
# LOADING BETAS + LABELS
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
                continue

            rows.append(
                {
                    "subject": subject,
                    "session": ses_dir.name,   # ses-01, ses-02...
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
            f"Check that labels include 'run-*_recog_*_*'."
        )
    return df


def build_pair_for_modality(df: pd.DataFrame, modality: str):
    """
    Within-modality: filter to one modality and one emotion pair.
    CV grouping: SESSION ONLY (leave-one-session-out).
    """
    emo_a, emo_b = PAIR
    meta = df[(df["modality"] == modality) & (df["emotion"].isin([emo_a, emo_b]))].copy()

    if meta.empty:
        raise ValueError(f"No samples for modality={modality} and pair={PAIR}")

    # X: 4D image (one beta per sample)
    X_img = concat_imgs(meta["beta_path"].tolist())

    # y: labels
    y = meta["emotion"].to_numpy()

    # groups: leave-one-session-out
    groups = meta["session"].to_numpy()

    return X_img, y, groups, meta


# ============================================================
# OUTPUTS
# ============================================================

def save_outputs(subject: str, modality: str, scores_arr: np.ndarray, bg_img, meta: pd.DataFrame, mask_img):
    """
    scores_arr: 3D numpy array from SearchLight (sl.scores_)
    mask_img: NIfTI image used as mask (template for affine/shape)
    """
    emo_a, emo_b = PAIR
    outdir = OUTDIR / subject / f"mod-{modality}"
    outdir.mkdir(parents=True, exist_ok=True)

    stem = f"{subject}_mod-{modality}_pair-{emo_a}-vs-{emo_b}"
    nii_path = outdir / f"{stem}_searchlight-acc.nii.gz"
    png_path = outdir / f"{stem}_searchlight-acc.png"
    log_path = outdir / f"{stem}_runlog.json"

    # Replace NaN/inf to avoid plotting/saving issues
    scores_arr = np.nan_to_num(scores_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Wrap scores array into NIfTI
    scores_nii = new_img_like(mask_img, scores_arr)
    scores_nii.to_filename(nii_path)

    # Plot
    display = plot_stat_map(
        scores_nii,
        bg_img=bg_img,
        title=f"{subject} | {modality} | {emo_a} vs {emo_b} (recog)",
        display_mode="z",
        cut_coords=5,
        threshold=None,
    )
    display.savefig(png_path)
    display.close()

    # Log
    log = {
        "subject": subject,
        "task": "recognition",
        "modality": modality,
        "pair": [emo_a, emo_b],
        "n_samples": int(len(meta)),
        "class_counts": meta["emotion"].value_counts().to_dict(),
        "n_sessions": int(pd.Series(meta["session"]).nunique()),
        "radius_mm": RADIUS_MM,
        "classifier": "LinearSVC",
        "C": C_SVM,
        "cv": "LeaveOneGroupOut(session)",
        "n_jobs": N_JOBS,
        "test_mode_slice": bool(TEST_MODE),
        "test_slice_z": TEST_SLICE_Z,
        "outputs": {"nii": str(nii_path), "png": str(png_path)},
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"[saved] {nii_path}")
    print(f"[saved] {png_path}")
    print(f"[saved] {log_path}")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="e.g., sub-001")
    return p.parse_args()


def restrict_mask_to_single_slice(mask_img):
    """Return a mask image restricted to one axial slice."""
    mask_data = mask_img.get_fdata()

    z = TEST_SLICE_Z if (TEST_SLICE_Z is not None) else (mask_data.shape[2] // 2)

    # Keep only one slice (z)
    new_mask = np.zeros_like(mask_data)
    new_mask[:, :, z] = (mask_data[:, :, z] > 0).astype(new_mask.dtype)

    # If the chosen slice has no mask voxels, fall back to middle slice with any voxels
    if new_mask.sum() == 0:
        z_mid = mask_data.shape[2] // 2
        new_mask[:, :, z_mid] = (mask_data[:, :, z_mid] > 0).astype(new_mask.dtype)
        z = z_mid

    return new_img_like(mask_img, new_mask), z


def main():
    args = parse_args()
    subject = args.subject

    # 1) Load dataset (all sessions)
    df = collect_samples(subject)

    # Choose a mask based on the first BAS2 folder we encounter
    first_row = df.iloc[0]
    bas_dir = Path(first_row["bas_dir"])
    first_beta = Path(first_row["beta_path"])

    mask_path = choose_matching_mask(bas_dir, first_beta)
    mask_img = load_img(str(mask_path))
    print(f"[mask] Using: {mask_path}")

    # TEST MODE: restrict to a single slice
    if TEST_MODE:
        mask_img, z = restrict_mask_to_single_slice(mask_img)
        print(f"[mask] Restricted to single slice z={z}")

    # Run within-modality searchlight for each modality
    for modality in MODALITIES:
        print(f"\n=== {subject} | modality={modality} | task=recognition | pair={PAIR[0]} vs {PAIR[1]} ===")

        X_img, y, groups, meta = build_pair_for_modality(df, modality)

        # Need >=2 sessions for LeaveOneGroupOut(session)
        if pd.Series(groups).nunique() < 2:
            raise ValueError(
                f"Not enough unique session groups for CV in modality={modality}. "
                f"Found {pd.Series(groups).nunique()} session(s)."
            )

        # 3) Searchlight computation
        scores_arr = run_searchlight(X_img, y, groups, mask_img)

        # 4) Visualization + saving
        bg_img = mean_img(X_img, copy_header=True)
        save_outputs(subject, modality, scores_arr, bg_img, meta, mask_img)

    print("\nDone.")


if __name__ == "__main__":
    main()
