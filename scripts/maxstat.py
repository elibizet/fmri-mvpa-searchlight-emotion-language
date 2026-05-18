"""
What this script does:
- load the real searchlight map
- load all saved permutation maps
- extract the maximum value inside the mask from each permutation map
- build the null max-stat distribution
- compute the 95th percentile threshold
- apply that threshold to the real map
- compute a corrected p-value map
- save outputs
- save a voxelwise accuracy histogram
"""

# run: python maxstat_from_saved_permutation_maps.py

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn.image import load_img, new_img_like


# PATHS

ALPHA = 0.05

SUBJECT_DIR = Path(
    "/home/rita/Elisabeth/Results/Searchlight results/BAS2 recog task/happiness vs anger (8 mm radius)/recog_task_whole_brain_searchlight_permutations_global/sub-001/mod-audiovisual"
)

REAL_DIR = SUBJECT_DIR

MASK_DIR = Path(
    "/home/rita/Elisabeth/singleN_betas/sub-001/ses-01/BAS2"
)

OUTDIR = SUBJECT_DIR / "posthoc_maxstat_correction"


# HELPERS

def sanitize_array(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def compute_corrected_p_map(
    real_data: np.ndarray,
    max_stats: np.ndarray,
    mask_bool: np.ndarray,
) -> np.ndarray:
    """
    Max-stat corrected voxelwise p map:
    p_corr(v) = proportion of permutation maxima >= observed voxel value.

    Only compute p-values inside the searchlight mask.
    Voxels outside the mask are set to 1.
    """
    p_map = np.ones(real_data.shape, dtype=float)

    valid_mask = np.isfinite(real_data) & mask_bool

    for idx in zip(*np.where(valid_mask)):
        obs = real_data[idx]
        p_map[idx] = (np.sum(max_stats >= obs) + 1) / (len(max_stats) + 1)

    return p_map


def find_real_map(real_dir: Path) -> Path:
    matches = sorted(real_dir.glob("*_real_searchlight-acc.nii.gz"))

    if len(matches) == 0:
        raise FileNotFoundError(f"No real searchlight map found in:\n{real_dir}")

    if len(matches) > 1:
        print("[warning] More than one real map found. Using first one:")
        for m in matches:
            print(" ", m.name)

    return matches[0]


def find_mask(mask_dir: Path) -> Path:
    candidates = [mask_dir / "_mask.nii", mask_dir / "mask.nii"]
    existing = [p for p in candidates if p.exists()]

    if not existing:
        raise FileNotFoundError(
            f"No mask found in {mask_dir} "
            f"(expected _mask.nii or mask.nii)"
        )

    for p in candidates:
        if p.exists():
            return p

    return existing[0]


def collect_permutation_maps(subject_dir: Path):
    perm_dirs = [
        subject_dir / "permutations",
        subject_dir / "permutations 2",
        subject_dir / "permutations 3",
        subject_dir / "permutations 4",
    ]

    perm_paths = []

    for d in perm_dirs:
        if d.exists():
            found = sorted(d.glob("perm_*_searchlight-acc.nii.gz"))
            print(f"[info] {d.name}: found {len(found)} permutation maps")
            perm_paths.extend(found)
        else:
            print(f"[info] {d.name}: folder not found, skipping")

    if len(perm_paths) == 0:
        raise ValueError(f"No permutation maps found under:\n{subject_dir}")

    return perm_paths, perm_dirs


def compute_max_in_mask(img, mask_bool: np.ndarray) -> float:
    data = sanitize_array(img.get_fdata())
    vals = data[mask_bool]

    if vals.size == 0:
        raise ValueError("Mask contains no voxels.")

    return float(vals.max())


# MAIN

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    real_map_path = find_real_map(REAL_DIR)
    mask_path = find_mask(MASK_DIR)
    perm_paths, perm_dirs = collect_permutation_maps(SUBJECT_DIR)

    print(f"[real map] {real_map_path}")
    print(f"[mask] {mask_path}")
    print(f"[info] total permutation maps found: {len(perm_paths)}")

    real_img = load_img(str(real_map_path))
    real_data = sanitize_array(real_img.get_fdata())

    mask_img = load_img(str(mask_path))
    mask_bool = mask_img.get_fdata() > 0

    # Build max-null distribution
    max_stats = []
    perm_rows = []

    for i, perm_path in enumerate(perm_paths, start=1):
        perm_img = load_img(str(perm_path))
        perm_max = compute_max_in_mask(perm_img, mask_bool)

        max_stats.append(perm_max)

        perm_rows.append({
            "perm_number_in_collection": i,
            "folder": perm_path.parent.name,
            "filename": perm_path.name,
            "max_stat": perm_max,
        })

        print(
            f"[perm {i:03d}] max = {perm_max:.4f} | "
            f"{perm_path.parent.name}/{perm_path.name}"
        )

    max_stats = np.array(max_stats, dtype=float)

    # Compute corrected threshold
    threshold_corr = float(np.percentile(max_stats, 100 * (1 - ALPHA)))

    # Apply threshold to real map
    thresholded_data = np.where(
        (real_data >= threshold_corr) & mask_bool,
        real_data,
        0.0
    )

    significant_mask_data = (
        (real_data >= threshold_corr) & mask_bool
    ).astype(np.int32)

    thresholded_img = new_img_like(real_img, thresholded_data)
    significant_mask_img = new_img_like(real_img, significant_mask_data)

    # Compute corrected p-value map
    p_corr_data = compute_corrected_p_map(real_data, max_stats, mask_bool)
    p_corr_img = new_img_like(real_img, p_corr_data)

    n_sig_voxels = int(significant_mask_data.sum())
    real_max = float(real_data[mask_bool].max())

    # Save NIfTI outputs
    thresholded_nii = OUTDIR / "real_map_maxstat_thresholded.nii.gz"
    significant_mask_nii = OUTDIR / "real_map_maxstat_significant_mask.nii.gz"
    p_corr_nii = OUTDIR / "real_map_pcorr_maxstat.nii.gz"

    thresholded_img.to_filename(str(thresholded_nii))
    significant_mask_img.to_filename(str(significant_mask_nii))
    p_corr_img.to_filename(str(p_corr_nii))

    # Save max-stat values as CSV
    maxstats_csv = OUTDIR / "max_stat_values.csv"
    pd.DataFrame(perm_rows).to_csv(maxstats_csv, index=False)

    # Save voxelwise accuracy histogram
    hist_png = OUTDIR / "voxel_accuracy_histogram.png"

    voxel_vals = real_data[mask_bool]
    voxel_vals = voxel_vals[np.isfinite(voxel_vals)]

    plt.figure(figsize=(7, 4))

    plt.hist(
        voxel_vals,
        bins=60,
        edgecolor="black",
        alpha=0.8
    )

    # Chance level for binary classification
    plt.axvline(
        0.5,
        linestyle="--",
        linewidth=1.5,
        label="Chance level = 0.50"
    )

    # Max-stat corrected threshold
    plt.axvline(
        threshold_corr,
        linestyle="--",
        linewidth=1.5,
        label=f"Max-stat threshold = {threshold_corr:.4f}"
    )

    plt.title("Searchlight Accuracy Histogram")
    plt.xlabel("Accuracy")
    plt.ylabel("Voxel count")
    plt.xlim(0.4, 1.0)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(hist_png, dpi=300)
    plt.close()

    # Save summary JSON
    summary = {
        "real_map": str(real_map_path),
        "mask": str(mask_path),
        "n_permutations": int(len(max_stats)),
        "alpha": float(ALPHA),
        "real_map_max": float(real_max),
        "max_stat_threshold": float(threshold_corr),
        "n_significant_voxels": int(n_sig_voxels),
        "min_max_stat": float(max_stats.min()),
        "mean_max_stat": float(max_stats.mean()),
        "max_max_stat": float(max_stats.max()),
        "permutation_folders_checked": [str(d) for d in perm_dirs],
        "outputs": {
            "thresholded_map": str(thresholded_nii),
            "significant_mask": str(significant_mask_nii),
            "p_corr_map": str(p_corr_nii),
            "max_stat_values_csv": str(maxstats_csv),
            "voxel_accuracy_histogram_png": str(hist_png),
        },
    }

    summary_json = OUTDIR / "maxstat_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print final summary
    print("\n=== DONE ===")
    print(f"[n permutations] {len(max_stats)}")
    print(f"[real max] {real_max:.4f}")
    print(f"[max-stat threshold @ alpha={ALPHA}] {threshold_corr:.4f}")
    print(f"[significant voxels] {n_sig_voxels}")
    print(f"[saved] {thresholded_nii}")
    print(f"[saved] {significant_mask_nii}")
    print(f"[saved] {p_corr_nii}")
    print(f"[saved] {maxstats_csv}")
    print(f"[saved] {hist_png}")
    print(f"[saved] {summary_json}")


if __name__ == "__main__":
    main()