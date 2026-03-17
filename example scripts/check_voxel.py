"""
This code snippet is performing the following tasks:
alignment check between beta and LanA probabilistic map, 
to ensure they are in the same space and have the same voxel size. 
It loads both images, checks their shapes and voxel sizes, 
resamples the LanA map to match the beta image, 
and visualizes the overlay to confirm proper alignment. 
This is important for subsequent analyses that rely on accurate spatial correspondence 
between the two images
"""

# The seconsd part of the code (in threshold.py) 
# goes further by applying a threshold to the resampled LanA map to create 
# a binary ROI: create a binary mask from the thresholded map for the analysis.

"""import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

beta_path = r"D:\singleN_betas\sub-001\ses-01\BAS2\beta_0101.nii"
spm_mask_path = r"D:\singleN_betas\sub-001\ses-01\BAS2\mask.nii"
lana_path = r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\SPM\LanA_n806.nii"

out_mask_path = r"D:\singleN_betas\sub-001\ses-01\BAS2\LanA_ROI_pGT02_inSPMmask.nii.gz"

beta = nib.load(beta_path)
spm_mask = nib.load(spm_mask_path)
lana = nib.load(lana_path)

# resample LanA -> beta grid
lana_to_beta = resample_to_img(lana, beta, interpolation="continuous")

# threshold -> binary ROI
p_thr = 0.2
lana_bin = (lana_to_beta.get_fdata() > p_thr)

# restrict to voxels where SPM estimated betas
spm_bin = (spm_mask.get_fdata() > 0)

final_bin = (lana_bin & spm_bin).astype(np.uint8)

final_roi = nib.Nifti1Image(final_bin, beta.affine, beta.header)
nib.save(final_roi, out_mask_path)

print("Saved:", out_mask_path)
print("ROI voxels:", int(final_bin.sum()))"""


import nibabel as nib
from nilearn.image import resample_to_img
from nilearn import plotting

beta_path = r"D:\singleN_betas\sub-001\ses-01\BAS2\beta_0101.nii"
lana_path = r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\SPM\LanA_n806.nii"

beta = nib.load(beta_path)
lana = nib.load(lana_path)

print("Beta shape/zooms:", beta.shape, beta.header.get_zooms()[:3])
print("LanA shape/zooms:", lana.shape, lana.header.get_zooms()[:3])

lana_to_beta = resample_to_img(lana, beta, interpolation="continuous")

disp = plotting.plot_stat_map(beta, title="Beta + LanA probabilistic map (overlay)",
                             display_mode="ortho", cut_coords=(0, -20, 20))
disp.add_overlay(lana_to_beta, alpha=0.6)
plotting.show()
