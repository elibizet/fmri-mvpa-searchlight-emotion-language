"""
The script can be addapted to align any mask to the beta image space, 
but here it is specifically used to prepare a new rest-of-brain mask for MVPA analyses.

The script loads a new rest-of-brain mask, 
resamples it to the beta image space, binarizes it, 
and then applies the data mask to ensure it only includes voxels 
present in the beta images. Finally, it saves the aligned mask for use in MVPA analyses.

Make sure to change the paths at the top of the script to your specific file locations before running."""

from pathlib import Path
import numpy as np
from nilearn.image import load_img, resample_to_img, new_img_like

# paths
mask_path = Path(r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\rest_brain.nii")
ref_beta_path = Path(r"D:\singleN_betas\sub-001\ses-01\BAS2\beta_0001.nii")
data_mask_path = Path(r"D:\singleN_betas\sub-001\ses-01\BAS2\mask.nii")

# load images
mask_img = load_img(str(mask_path))
ref_img = load_img(str(ref_beta_path))
data_mask_img = load_img(str(data_mask_path))

# clean original mask data first
mask_data = mask_img.get_fdata()
mask_data = np.nan_to_num(mask_data, nan=0.0, posinf=0.0, neginf=0.0)
mask_img_clean = new_img_like(mask_img, mask_data)

# resample mask to beta image space
mask_resampled = resample_to_img(
    mask_img_clean,
    ref_img,
    interpolation="nearest",
    force_resample=True,
    copy_header=True,
)

# binarize resampled mask
mask_resampled_data = mask_resampled.get_fdata()
mask_resampled_data = np.nan_to_num(mask_resampled_data, nan=0.0, posinf=0.0, neginf=0.0)
mask_resampled_bin = (mask_resampled_data > 0).astype(np.uint8)

# also clean and binarize data mask
data_mask_data = data_mask_img.get_fdata()
data_mask_data = np.nan_to_num(data_mask_data, nan=0.0, posinf=0.0, neginf=0.0)
data_mask_bin = (data_mask_data > 0).astype(np.uint8)

# final aligned mask = resampled ROI/rest-of-brain mask inside data mask
final_mask_data = (mask_resampled_bin & data_mask_bin).astype(np.uint8)

final_mask_img = new_img_like(ref_img, final_mask_data)

# save
out_path = Path(r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\aligned_rest_of_brain_mask.nii.gz")
final_mask_img.to_filename(str(out_path))

print("Saved aligned mask to:", out_path)
print("Shape:", final_mask_img.shape)
print("N voxels:", int(final_mask_data.sum()))
print("Unique:", np.unique(final_mask_data))

print("beta shape:", ref_img.shape)
print("mask shape:", final_mask_img.shape)
print("beta affine:\n", ref_img.affine)
print("mask affine:\n", final_mask_img.affine)
