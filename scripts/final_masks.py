"""
Create final ROI masks for MVPA by intersecting independently defined ROI masks
(visual, auditory, language) with the subject's data mask.

What this script does:
1. Loads one representative whole-brain data mask from the functional data.
2. Loads the ROI masks:
   - visual (NeuroQuery, thresholded and binarized)
   - auditory (NeuroQuery, thresholded and binarized)
   - language (supervisor-created SPM mask (No auditory/visual voxels), thresholded and binarized)
3. Resamples each ROI mask to the data-mask space using nearest-neighbor interpolation.
4. Intersects each ROI mask with the data mask.
5. Saves the final binary masks to disk.
6. Prints sanity checks (unique values, voxel counts, shapes).

Output masks are ready to be used as mask_img in MVPA analysis.
"""

from pathlib import Path
import numpy as np
from nilearn.image import load_img, math_img, resample_to_img


# PATHS
data_mask_path = Path(r"D:\singleN_betas\sub-001\ses-01\BAS2\mask.nii")

visual_mask_path = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\visual\visual_mask_z3.nii.gz"
)
auditory_mask_path = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\auditory\auditory_mask_z3.nii.gz"
)
language_mask_path = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\language\languagep03_noVisual_noAuditory.nii"
)

out_visual = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\final_visual_mask.nii.gz"
)
out_auditory = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\final_auditory_mask.nii.gz"
)
out_language = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\final_language_mask.nii.gz"
)

# CHECK INPUT FILES EXIST
for p in [data_mask_path, visual_mask_path, auditory_mask_path, language_mask_path]:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

# LOAD AND BINARIZE DATA MASK
data_mask = load_img(str(data_mask_path))
data_mask_bin = math_img("(img > 0).astype('int32')", img=data_mask)

# LOAD ROI MASKS
visual_mask = load_img(str(visual_mask_path))
auditory_mask = load_img(str(auditory_mask_path))
language_mask = load_img(str(language_mask_path))

# RESAMPLE ROI MASKS TO DATA MASK SPACE
# nearest interpolation is required for masks
visual_mask_res = resample_to_img(
    visual_mask, data_mask_bin, interpolation="nearest", force_resample=True
)
auditory_mask_res = resample_to_img(
    auditory_mask, data_mask_bin, interpolation="nearest", force_resample=True
)
language_mask_res = resample_to_img(
    language_mask, data_mask_bin, interpolation="nearest", force_resample=True
)

# BINARIZE RESAMPLED ROI MASKS
visual_mask_res_bin = math_img("(img > 0).astype('int32')", img=visual_mask_res)
auditory_mask_res_bin = math_img("(img > 0).astype('int32')", img=auditory_mask_res)
language_mask_res_bin = math_img("(img > 0).astype('int32')", img=language_mask_res)

# INTERSECT ROI MASKS WITH DATA MASK
final_visual = math_img(
    "((roi > 0) & (data > 0)).astype('int32')",
    roi=visual_mask_res_bin,
    data=data_mask_bin,
)

final_auditory = math_img(
    "((roi > 0) & (data > 0)).astype('int32')",
    roi=auditory_mask_res_bin,
    data=data_mask_bin,
)

final_language = math_img(
    "((roi > 0) & (data > 0)).astype('int32')",
    roi=language_mask_res_bin,
    data=data_mask_bin,
)

# SAVE FINAL MASKS
final_visual.to_filename(str(out_visual))
final_auditory.to_filename(str(out_auditory))
final_language.to_filename(str(out_language))

# SANITY CHECKS
print("\n=== DATA MASK ===")
data_arr = data_mask_bin.get_fdata()
print(f"shape={data_mask_bin.shape}, unique={np.unique(data_arr)}, n_voxels={int(data_arr.sum())}")

print("\n=== FINAL ROI MASKS ===")
for name, img, out_path in [
    ("visual", final_visual, out_visual),
    ("auditory", final_auditory, out_auditory),
    ("language", final_language, out_language),
]:
    arr = img.get_fdata()
    print(
        f"{name}: shape={img.shape}, unique={np.unique(arr)}, "
        f"n_voxels={int(arr.sum())}, saved_to={out_path}"
    )

print("\nDone.")