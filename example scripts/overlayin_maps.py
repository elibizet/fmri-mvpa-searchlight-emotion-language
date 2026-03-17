import os
import glob
import numpy as np
import nibabel as nib
from nilearn import image, plotting
from nilearn.image import new_img_like
from nilearn.datasets import load_mni152_template

# -------------------------
# SETTINGS
# -------------------------
root = r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\MVPA_happinessvsanger_results_maps"
subjects = ["sub-001", "sub-003"]
modalities = ["audio", "video", "audiovisual"]
analysis_folder = "mvpa_within_modality_happiness_anger"

out_dir = os.path.join(root, "OVERLAYS")
os.makedirs(out_dir, exist_ok=True)

TOP_PERCENT = 1.0  # show top 1% |weights|


# -------------------------
# HELPERS
# -------------------------
def find_weightmap_nii(subject, modality):
    """
    Find a weightmap NIfTI for subject+modality, requiring exact token '_{modality}_'
    to avoid audio matching audiovisual.
    """
    base = os.path.join(root, subject, analysis_folder)
    if not os.path.isdir(base):
        return None

    # search only for weightmaps
    candidates = glob.glob(os.path.join(base, "*weightmap*.nii*"))

    # enforce exact token match
    token = f"_{modality}_"
    matches = [p for p in candidates if token in os.path.basename(p)]

    if not matches:
        return None

    # most recent
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def abs_threshold_top_percent(img, percent=1.0):
    """Return a new image where only the top X% absolute voxels are kept (others set to 0)."""
    data = img.get_fdata()
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    abs_data = np.abs(data)
    nonzero = abs_data[abs_data > 0]

    if nonzero.size == 0:
        return new_img_like(img, abs_data)

    thr = np.percentile(nonzero, 100 - percent)
    thr_data = abs_data.copy()
    thr_data[thr_data < thr] = 0.0

    return new_img_like(img, thr_data)


# -------------------------
# MAIN
# -------------------------
bg = load_mni152_template()

for mod in modalities:
    p1 = find_weightmap_nii(subjects[0], mod)
    p2 = find_weightmap_nii(subjects[1], mod)

    if p1 is None or p2 is None:
        print(f"Skipping {mod}: missing files")
        print(f"  {subjects[0]} -> {p1}")
        print(f"  {subjects[1]} -> {p2}")
        continue

    print(f"\nModality: {mod}")
    print(f"  {subjects[0]}: {os.path.basename(p1)}")
    print(f"  {subjects[1]}: {os.path.basename(p2)}")

    img1 = image.load_img(p1)
    img2 = image.load_img(p2)

    # make absolute + threshold
    abs1_thr = abs_threshold_top_percent(img1, percent=TOP_PERCENT)
    abs2_thr = abs_threshold_top_percent(img2, percent=TOP_PERCENT)

    display = plotting.plot_stat_map(
        abs1_thr,
        bg_img=bg,
        title=f"Overlay |SVM weights| ({mod})\n{subjects[0]}=Red, {subjects[1]}=Blue",
        display_mode="ortho",
        cmap="Reds",
        colorbar=True
    )
    display.add_overlay(abs2_thr, cmap="Blues")

    out_png = os.path.join(out_dir, f"overlay_{subjects[0]}_{subjects[1]}_{mod}.png")
    display.savefig(out_png, dpi=300, bbox_inches="tight")
    display.close()

    print(f"Saved: {out_png}")

print("\nDone. Overlays saved to:", out_dir)
