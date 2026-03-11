"""Visualize 
an ROI-only mask (needs threshold) and 
restrict to SPM mask"""


import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn import plotting

beta_path = r"D:\singleN_betas\sub-001\ses-01\BAS2\beta_0101.nii"
spm_mask_path = r"D:\singleN_betas\sub-001\ses-01\BAS2\mask.nii"
lana_path = r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\SPM\LanA_n806.nii"

beta = nib.load(beta_path)
spm_mask = nib.load(spm_mask_path)
lana = nib.load(lana_path)

print("Beta shape/zooms:", beta.shape, beta.header.get_zooms()[:3])
print("LanA shape/zooms:", lana.shape, lana.header.get_zooms()[:3])

lana_to_beta = resample_to_img(lana, beta, interpolation="continuous")

# threshold probabilistic map -> binary ROI
p_thr = 0.2
lana_bin = (lana_to_beta.get_fdata() > p_thr)

# SPM analysis mask -> binary
mask_bin = (spm_mask.get_fdata() > 0)

# intersect
final_bin = (lana_bin & mask_bin).astype(np.uint8)
final_roi = nib.Nifti1Image(final_bin, beta.affine, beta.header)

disp = plotting.plot_stat_map(beta, title=f"Beta + LanA ROI (p>{p_thr})",
                             display_mode="ortho", cut_coords=(0, -20, 20))
disp.add_overlay(final_roi, alpha=0.6)
plotting.show()

print("ROI voxels:", int(final_bin.sum()))
