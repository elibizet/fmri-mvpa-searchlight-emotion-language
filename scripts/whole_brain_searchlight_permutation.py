"""debug-friendly permutation searchlight script built directly 
from my current one, 
with my supervisor's requirements kept in mind:
- same searchlight settings
- 8 mm radius
- Emotion pair e.g. happiness vs anger
- full unthresholded maps
- label shuffling
- start with 5 permutations
It does not: (see separate script for that)
- compute permutation maxima
- compute corrected thresholds
- compute corrected p-maps

Important choice: shuffle labels within session

Because your CV is LeaveOneGroupOut(session), the safest 
permutation is to shuffle labels within each session, 
not across the whole dataset."""

# run: python whole_brain_searchlight_permutation.py --subject sub-001 --n-perms 5 --seed 42

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

PROJECT_ROOT = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\Results\Searchlight results\BAS2 passive task\Permutation_test\Pair: anxiety vs sadness"
)
OUTDIR = (
    PROJECT_ROOT
    / "passive_task_video_whole_brain_searchlight_permutations"
    / "within_passive_betas"
)

BAS_FOLDER = "BAS2"
LABELS_FILE = "regressor_labels.csv"


# ANALYSIS CHOICES
TASK_FILTER = "passive"
PAIR = ("anxiety", "sadness")
MODALITIES = ["audiovisual"]   # e.g. ["audio", "video", "audiovisual"]

RADIUS_MM = 8.0
N_JOBS = 4
VERBOSE = 1
C_SVM = 1.0

DEFAULT_N_PERMS = 5
DEFAULT_RANDOM_SEED = 42


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

    m = re.match(rf"run-(\d+)_{TASK_FILTER}_([a-zA-Z]+)_([a-zA-Z]+)", label)
    if not m:
        return None

    run = int(m.group(1))
    emotion = m.group(2).lower()
    modality = m.group(3).lower()

    if modality not in {"audio", "video", "audiovisual"}:
        return None

    return {"run": run, "task": TASK_FILTER, "emotion": emotion, "modality": modality}


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
        scoring="accuracy",
    )
    sl.fit(X_img, y, groups=groups)
    return sl.scores_


def sanitize_scores(scores_arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(scores_arr, nan=0.0, posinf=0.0, neginf=0.0)


def permute_labels_within_groups(y, groups, rng):
    """
    Shuffle labels within each session/group.

    This preserves:
    - number of samples per session
    - class balance per session
    - the grouping structure used in LOSO CV

    For a 2-class design with one happiness and one anger per session,
    this becomes a swap-or-not-swap within each session.
    """
    y_perm = np.array(y, copy=True)

    groups = np.asarray(groups)
    unique_groups = pd.unique(groups)

    for g in unique_groups:
        idx = np.where(groups == g)[0]
        y_perm[idx] = rng.permutation(y_perm[idx])

    return y_perm


def save_map_as_nii(arr, ref_img, out_path: Path):
    nii = new_img_like(ref_img, sanitize_scores(arr))
    nii.to_filename(out_path)
    return nii


def save_plot(img, bg_img, title, out_path: Path, threshold=None):
    display = plot_stat_map(
        img,
        bg_img=bg_img,
        title=title,
        display_mode="z",
        cut_coords=5,
        threshold=threshold,
    )
    display.savefig(out_path)
    display.close()


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
            f"No passive samples found for {subject}. "
            f"Check that labels include 'run-*_passive_*_*'."
        )
    return df


def build_pair_for_modality(df: pd.DataFrame, modality: str):
    emo_a, emo_b = PAIR
    meta = df[(df["modality"] == modality) & (df["emotion"].isin([emo_a, emo_b]))].copy()

    if meta.empty:
        raise ValueError(f"No samples for modality={modality} and pair={PAIR}")

    # stable ordering helps reproducibility
    meta = meta.sort_values(["session", "run", "emotion", "beta_name"]).reset_index(drop=True)

    X_img = concat_imgs(meta["beta_path"].tolist())
    y = meta["emotion"].to_numpy()
    groups = meta["session"].to_numpy()   # Leave-one-session-out

    return X_img, y, groups, meta


# OUTPUTS
def save_real_outputs(subject: str, modality: str, scores_arr: np.ndarray, bg_img, meta: pd.DataFrame, mask_img, n_perms: int, seed: int):
    emo_a, emo_b = PAIR
    outdir = OUTDIR / subject / f"mod-{modality}"
    outdir.mkdir(parents=True, exist_ok=True)

    stem = f"{subject}_mod-{modality}_pair-{emo_a}-vs-{emo_b}"

    nii_path = outdir / f"{stem}_real_searchlight-acc.nii.gz"
    png_path = outdir / f"{stem}_real_searchlight-acc.png"
    log_path = outdir / f"{stem}_runlog.json"
    meta_path = outdir / f"{stem}_design_labels.csv"

    scores_img = save_map_as_nii(scores_arr, mask_img, nii_path)

    save_plot(
        scores_img,
        bg_img=bg_img,
        title=f"{subject} | {modality} | REAL | {emo_a} vs {emo_b} ({TASK_FILTER})",
        out_path=png_path,
        threshold=None,
    )

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
        "cv": "LeaveOneGroupOut(session)",
        "permutation_scheme": "shuffle labels within session",
        "n_jobs": N_JOBS,
        "n_permutations_requested": int(n_perms),
        "random_seed": int(seed),
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


def save_permutation_outputs(subject: str, modality: str, perm_idx: int, perm_scores: np.ndarray, y_true, y_perm, bg_img, meta: pd.DataFrame, mask_img):
    outdir = OUTDIR / subject / f"mod-{modality}" / "permutations"
    outdir.mkdir(parents=True, exist_ok=True)

    perm_stem = f"perm_{perm_idx:03d}"

    nii_path = outdir / f"{perm_stem}_searchlight-acc.nii.gz"
    png_path = outdir / f"{perm_stem}_searchlight-acc.png"
    labels_path = outdir / f"{perm_stem}_labels.csv"

    perm_img = save_map_as_nii(perm_scores, mask_img, nii_path)

    save_plot(
        perm_img,
        bg_img=bg_img,
        title=f"{subject} | {modality} | permutation {perm_idx}",
        out_path=png_path,
        threshold=None,
    )

    perm_lab_df = meta[["session", "run", "emotion", "beta_name", "beta_path"]].copy()
    perm_lab_df["y_true"] = y_true
    perm_lab_df["y_perm"] = y_perm
    perm_lab_df.to_csv(labels_path, index=False)

    print(f"[saved] {nii_path}")
    print(f"[saved] {png_path}")
    print(f"[saved] {labels_path}")


# MAIN
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="e.g., sub-001")
    p.add_argument("--n-perms", type=int, default=DEFAULT_N_PERMS, help="number of permutations")
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="random seed")
    return p.parse_args()


def main():
    t0 = time.time()

    args = parse_args()
    subject = args.subject
    n_perms = args.n_perms
    seed = args.seed

    rng = np.random.default_rng(seed)

    df = collect_samples(subject)

    first_row = df.iloc[0]
    bas_dir = Path(first_row["bas_dir"])
    first_beta = Path(first_row["beta_path"])

    mask_path = choose_matching_mask(bas_dir, first_beta)
    mask_img = load_img(str(mask_path))
    print(f"[mask] Using: {mask_path}")

    for modality in MODALITIES:
        print(f"\n=== {subject} | modality={modality} | task={TASK_FILTER} | pair={PAIR[0]} vs {PAIR[1]} ===")

        X_img, y, groups, meta = build_pair_for_modality(df, modality)
        bg_img = mean_img(X_img, copy_header=True)

        n_sessions = pd.Series(groups).nunique()
        print(f"[info] n_samples={len(meta)} | n_sessions={n_sessions} | n_jobs={N_JOBS} | radius_mm={RADIUS_MM}")

        if n_sessions < 2:
            raise ValueError(
                f"Not enough unique session groups for CV in modality={modality}. "
                f"Found {n_sessions} session(s)."
            )

        # sanity check for within-session permutation
        session_counts = (
            meta.groupby(["session", "emotion"])
            .size()
            .unstack(fill_value=0)
        )
        print("\n[debug] counts per session:")
        print(session_counts)

        missing_class_sessions = session_counts[
            (session_counts.get(PAIR[0], 0) == 0) |
            (session_counts.get(PAIR[1], 0) == 0)
        ]
        if len(missing_class_sessions) > 0:
            raise ValueError(
                "At least one session does not contain both emotions for this modality. "
                "Within-session permutation would not be valid there.\n"
                f"{missing_class_sessions}"
            )

        t_mod = time.time()

        # REAL SEARCHLIGHT
        print("\n[real] running searchlight with true labels...")
        real_scores = run_searchlight(X_img, y, groups, mask_img)
        real_scores = sanitize_scores(real_scores)

        save_real_outputs(
            subject=subject,
            modality=modality,
            scores_arr=real_scores,
            bg_img=bg_img,
            meta=meta,
            mask_img=mask_img,
            n_perms=n_perms,
            seed=seed,
        )

        # PERMUTATION SEARCHLIGHTS
        for perm_idx in range(1, n_perms + 1):
            print(f"\n[perm {perm_idx:03d}/{n_perms:03d}] shuffling labels within session...")
            y_perm = permute_labels_within_groups(y, groups, rng)

            print(f"[perm {perm_idx:03d}] running searchlight...")
            perm_scores = run_searchlight(X_img, y_perm, groups, mask_img)
            perm_scores = sanitize_scores(perm_scores)

            save_permutation_outputs(
                subject=subject,
                modality=modality,
                perm_idx=perm_idx,
                perm_scores=perm_scores,
                y_true=y,
                y_perm=y_perm,
                bg_img=bg_img,
                meta=meta,
                mask_img=mask_img,
            )

        print(f"[time] modality={modality} took {(time.time() - t_mod)/3600:.2f} hours")

    print(f"\nDone - Total runtime: {(time.time() - t0)/3600:.2f} hours")


if __name__ == "__main__":
    main()