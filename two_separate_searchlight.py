"""By splitting sessions into (1–10) and late (11–19) 
and doing leave-one-session-out within each block:

1. Is the emotion representation stable across time within an epoch?
If a classifier trained on 9 sessions can decode the held-out 10th 
session within the same block, then the multivoxel pattern that carries 
“anxiety vs sadness” is consistent across sessions 
(robust to day-to-day/session-to-session 
variability) in that time window.

2. Does that stability change across the experiment (early vs late)?
Comparing maps from block A vs block B tests whether the set of 
informative voxels (and/or decoding strength) is:
stable across the whole study, or
shifts over time (e.g., habituation, learning, fatigue, 
scanner drift, strategy change, 
changes in attention/arousal, etc.)"""


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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# PATHS (EDIT IF NEEDED)
BETAS_ROOT = Path(r"D:\singleN_betas")

PROJECT_ROOT = Path(r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis")
OUTDIR = PROJECT_ROOT / "Splitting sessions(anxiety vs sadness) in audio" / "within_recog_betas"

BAS_FOLDER = "BAS2"
LABELS_FILE = "regressor_labels.csv"


# ANALYSIS CHOICES
TASK_FILTER = "recog"
PAIR = ("anxiety", "sadness")  # or ("anger", "happiness")
MODALITIES = ["audio"]  # or ["audio","video","audiovisual"]

RADIUS_MM = 5.0
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
    return sl.scores_  # 3D numpy array


def session_to_int(session_name: str) -> int:
    """
    Expects session folder names like 'ses-01', 'ses-1', 'ses-019', etc.
    Returns the integer session number.
    """
    m = re.match(r"ses-(\d+)$", str(session_name))
    if not m:
        raise ValueError(f"Unexpected session format: {session_name} (expected 'ses-XX')")
    return int(m.group(1))


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
                    "session": ses_dir.name,              # e.g. "ses-01"
                    "session_num": session_to_int(ses_dir.name),  # e.g. 1
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


def build_pair_for_modality(df: pd.DataFrame, modality: str, session_range: tuple[int, int]):
    """
    session_range is inclusive: (start, end)
    """
    emo_a, emo_b = PAIR
    start_ses, end_ses = session_range

    meta = df[
        (df["modality"] == modality)
        & (df["emotion"].isin([emo_a, emo_b]))
        & (df["session_num"].between(start_ses, end_ses))
    ].copy()

    if meta.empty:
        raise ValueError(f"No samples for modality={modality}, pair={PAIR}, sessions={session_range}")

    # to keep groups stable, sort by session then run then emotion (optional)
    meta = meta.sort_values(["session_num", "run", "emotion"]).reset_index(drop=True)

    X_img = concat_imgs(meta["beta_path"].tolist())
    y = meta["emotion"].to_numpy()

    # leave-one-session-out within the block
    groups = meta["session"].to_numpy()  # e.g. "ses-01", "ses-02", ...

    return X_img, y, groups, meta


# OUTPUTS

def save_outputs(subject: str, modality: str, block_name: str,
                 session_range: tuple[int, int],
                 scores_arr: np.ndarray, bg_img, meta: pd.DataFrame, mask_img):

    emo_a, emo_b = PAIR
    start_ses, end_ses = session_range

    outdir = OUTDIR / subject / f"mod-{modality}" / f"block-{block_name}_ses-{start_ses:02d}-{end_ses:02d}"
    outdir.mkdir(parents=True, exist_ok=True)

    stem = f"{subject}_mod-{modality}_pair-{emo_a}-vs-{emo_b}_block-{block_name}_ses-{start_ses:02d}-{end_ses:02d}"
    nii_path = outdir / f"{stem}_searchlight-acc.nii.gz"
    png_path = outdir / f"{stem}_searchlight-acc.png"
    log_path = outdir / f"{stem}_runlog.json"

    scores_arr = np.nan_to_num(scores_arr, nan=0.0, posinf=0.0, neginf=0.0)

    scores_nii = new_img_like(mask_img, scores_arr)
    scores_nii.to_filename(nii_path)

    display = plot_stat_map(
        scores_nii,
        bg_img=bg_img,
        title=f"{subject} | {modality} | {emo_a} vs {emo_b} | {block_name} (ses {start_ses}-{end_ses})",
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
        "block": block_name,
        "session_range_inclusive": [start_ses, end_ses],
        "n_samples": int(len(meta)),
        "class_counts": meta["emotion"].value_counts().to_dict(),
        "n_sessions": int(pd.Series(meta["session"]).nunique()),
        "radius_mm": RADIUS_MM,
        "classifier": "LinearSVC",
        "C": C_SVM,
        "cv": "LeaveOneGroupOut(session) in this dataset session corresponds to a run",
        "n_jobs": N_JOBS,
        "outputs": {"nii": str(nii_path), "png": str(png_path)},
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"[saved] {nii_path}")
    print(f"[saved] {png_path}")
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
    
    print("\n[Sanity check] Number of unique runs per session:")
    #print(df.groupby("session")["run"].nunique())
    print(df.groupby("session")["run"].nunique().value_counts())


    # Pick mask based on first available beta
    first_row = df.iloc[0]
    bas_dir = Path(first_row["bas_dir"])
    first_beta = Path(first_row["beta_path"])

    mask_path = choose_matching_mask(bas_dir, first_beta)
    mask_img = load_img(str(mask_path))
    print(f"[mask] Using: {mask_path}")

    # Two blocks you requested
    blocks = [
        ("A", (1, 10)),   # train/test within sessions 1-10
        ("B", (11, 19)),  # train/test within sessions 11-19
    ]

    for modality in MODALITIES:
        for block_name, session_range in blocks:
            start_ses, end_ses = session_range
            print(
                f"\n=== {subject} | modality={modality} | pair={PAIR[0]} vs {PAIR[1]} "
                f"| BLOCK {block_name} sessions {start_ses}-{end_ses} ==="
            )

            X_img, y, groups, meta = build_pair_for_modality(df, modality, session_range)

            n_sessions = pd.Series(groups).nunique()
            print(f"[info] n_samples={len(meta)} | n_sessions={n_sessions} | n_jobs={N_JOBS} | radius_mm={RADIUS_MM}")

            if n_sessions < 2:
                raise ValueError(
                    f"Not enough unique session groups for CV in modality={modality}, block={block_name}. "
                    f"Found {n_sessions} session(s) within {session_range}."
                )

            t_run = time.time()
            scores_arr = run_searchlight(X_img, y, groups, mask_img)

            bg_img = mean_img(X_img, copy_header=True)
            save_outputs(subject, modality, block_name, session_range, scores_arr, bg_img, meta, mask_img)

            print(f"[time] modality={modality} block={block_name} took {(time.time() - t_run)/3600:.2f} hours")

    print(f"\nDone. Total runtime: {(time.time() - t0)/3600:.2f} hours")


if __name__ == "__main__":
    main()
