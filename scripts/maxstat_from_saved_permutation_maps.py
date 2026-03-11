"""
What this script does:
- load the real searchlight map
- load all saved permutation maps
- extract the maximum value inside the mask from each permutation map
- build the null max-stat distribution
- compute the 95th percentile threshold
- apply that threshold to the real map
- optionally compute a corrected p-value map
- save outputs
"""
# run: python maxstat_from_saved_permutation_maps.py

from pathlib import Path
import json
import numpy as np

from nilearn.image import load_img, new_img_like
from nilearn.plotting import plot_stat_map


# EDIT PATHS

# Real searchlight map (true labels)
REAL_MAP = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\Results\Searchlight results\BAS2 passive task\anxiety vs sadness (8 mm radius)\passive_task_video_whole_brain_searchlight_permutations\within_passive_betas\sub-001\mod-audiovisual\sub-001_mod-audiovisual_pair-anxiety-vs-sadness_real_searchlight-acc.nii.gz"
)

# Folder containing saved permutation maps
PERM_DIR = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\Results\Searchlight results\BAS2 passive task\anxiety vs sadness (8 mm radius)\passive_task_video_whole_brain_searchlight_permutations\within_passive_betas\sub-001\mod-audiovisual\permutations"
)

# Matching brain mask
MASK_MAP = Path(
    r"D:\singleN_betas\sub-001\ses-01\BAS2\_mask.nii"
)

# Optional background image for plotting
BG_IMG = None
# Example if I want one:
# BG_IMG = Path(r"...\mean_img.nii.gz")

# Output folder
OUTDIR = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\Results\Searchlight results\BAS2 passive task\anxiety vs sadness (8 mm radius)\posthoc_maxstat_correction"
)

ALPHA = 0.05


# HELPERS
def sanitize_array(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def get_mask_bool(mask_img) -> np.ndarray:
    return mask_img.get_fdata() > 0


def compute_max_in_mask(img, mask_bool: np.ndarray) -> float:
    data = sanitize_array(img.get_fdata())
    vals = data[mask_bool]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(vals.max())


def compute_corrected_p_map(real_data: np.ndarray, max_stats: np.ndarray) -> np.ndarray:
    """
    Max-stat corrected voxelwise p map:
    p_corr(v) = proportion of permutation maxima >= observed voxel score
    """
    real_data = np.asarray(real_data)
    p_map = np.ones(real_data.shape, dtype=float)

    finite_mask = np.isfinite(real_data)
    for idx in zip(*np.where(finite_mask)):
        obs = real_data[idx]
        p_map[idx] = (np.sum(max_stats >= obs) + 1) / (len(max_stats) + 1)

    return p_map


def save_plot(img, out_png: Path, title: str, bg_img=None, threshold=None):
    display = plot_stat_map(
        img,
        bg_img=bg_img,
        title=title,
        display_mode="z",
        cut_coords=5,
        threshold=threshold,
    )
    display.savefig(out_png)
    display.close()


# MAIN
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not REAL_MAP.exists():
        raise FileNotFoundError(f"Real map not found:\n{REAL_MAP}")

    if not PERM_DIR.exists():
        raise FileNotFoundError(f"Permutation directory not found:\n{PERM_DIR}")

    if not MASK_MAP.exists():
        raise FileNotFoundError(f"Mask not found:\n{MASK_MAP}")

    real_img = load_img(str(REAL_MAP))
    mask_img = load_img(str(MASK_MAP))
    bg_img = load_img(str(BG_IMG)) if BG_IMG is not None else None

    mask_bool = get_mask_bool(mask_img)

    # find permutation maps
    perm_paths = sorted(PERM_DIR.glob("perm_*_searchlight-acc.nii.gz"))
    if len(perm_paths) == 0:
        raise ValueError(f"No permutation maps found in:\n{PERM_DIR}")

    print(f"[info] Found {len(perm_paths)} permutation maps")

    # compute max-stat null distribution
    max_stats = []
    for i, perm_path in enumerate(perm_paths, start=1):
        perm_img = load_img(str(perm_path))
        perm_max = compute_max_in_mask(perm_img, mask_bool)
        max_stats.append(perm_max)
        print(f"[perm {i:03d}] max = {perm_max:.4f} | {perm_path.name}")

    max_stats = np.array(max_stats, dtype=float)

    # real map
    real_data = sanitize_array(real_img.get_fdata())
    real_max = float(real_data[mask_bool].max())

    # corrected threshold
    threshold_corr = float(np.percentile(max_stats, 100 * (1 - ALPHA)))

    # thresholded real map
    real_thresh_data = np.where(real_data >= threshold_corr, real_data, 0.0)
    real_thresh_img = new_img_like(real_img, real_thresh_data)

    # corrected p map
    p_corr_data = compute_corrected_p_map(real_data, max_stats)
    p_corr_img = new_img_like(real_img, p_corr_data)

    # save NIfTI outputs
    real_thresh_nii = OUTDIR / "real_map_maxstat_thresholded.nii.gz"
    p_corr_nii = OUTDIR / "real_map_pcorr_maxstat.nii.gz"

    real_thresh_img.to_filename(real_thresh_nii)
    p_corr_img.to_filename(p_corr_nii)

    # save plots
    real_png = OUTDIR / "real_map_unthresholded.png"
    thresh_png = OUTDIR / "real_map_maxstat_thresholded.png"
    p_png = OUTDIR / "real_map_pcorr_maxstat.png"

    save_plot(
        real_img,
        real_png,
        title="Real searchlight map (unthresholded)",
        bg_img=bg_img,
        threshold=None,
    )

    save_plot(
        real_thresh_img,
        thresh_png,
        title=f"Real map thresholded by max-stat (alpha={ALPHA})",
        bg_img=bg_img,
        threshold=threshold_corr,
    )

    save_plot(
        p_corr_img,
        p_png,
        title="Corrected p-value map (max-stat)",
        bg_img=bg_img,
        threshold=None,
    )

    # summary log
    summary = {
        "real_map": str(REAL_MAP),
        "mask_map": str(MASK_MAP),
        "permutation_dir": str(PERM_DIR),
        "n_permutations": int(len(max_stats)),
        "alpha": float(ALPHA),
        "real_map_max": float(real_max),
        "max_stat_distribution": [float(x) for x in max_stats],
        "max_stat_threshold": float(threshold_corr),
        "outputs": {
            "real_thresh_nii": str(real_thresh_nii),
            "p_corr_nii": str(p_corr_nii),
            "real_png": str(real_png),
            "thresh_png": str(thresh_png),
            "p_png": str(p_png),
        },
    }

    summary_path = OUTDIR / "maxstat_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"[real max] {real_max:.4f}")
    print(f"[threshold @ alpha={ALPHA}] {threshold_corr:.4f}")
    print(f"[saved] {real_thresh_nii}")
    print(f"[saved] {p_corr_nii}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()