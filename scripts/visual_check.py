""""This script is for visual checks of the final masks. 
It loads the final masks and plots them on top of a functional image 
to check that they look correct."""

from pathlib import Path
from nilearn.image import load_img, mean_img
from nilearn.plotting import plot_roi, show
import numpy as np

# PATHS
func_example_path = Path(r"D:\singleN_betas\sub-001\ses-01\BAS2\beta_0001.nii")  # any beta or functional image

visual_mask_path = Path(r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\final_visual_mask.nii.gz")
auditory_mask_path = Path(r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\final_auditory_mask.nii.gz")
language_mask_path = Path(r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\final_language_mask.nii.gz")
rest_brain_mask_path = Path(r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\aligned_rest_of_brain_mask.nii.gz")
# LOAD
bg_img = load_img(str(func_example_path))

visual_mask = load_img(str(visual_mask_path))
auditory_mask = load_img(str(auditory_mask_path))
language_mask = load_img(str(language_mask_path))
rest_brain_mask = load_img(str(rest_brain_mask_path))

# get data
data = rest_brain_mask.get_fdata() # change mask when needed

# count voxels (assuming binary mask: 0/1)
n_voxels = np.sum(data > 0)

print(f"Number of voxels in rest brain mask: {int(n_voxels)}")

# PLOT
plot_roi(visual_mask, bg_img=bg_img, title="Visual mask")
plot_roi(auditory_mask, bg_img=bg_img, title="Auditory mask")
plot_roi(language_mask, bg_img=bg_img, title="Language mask")
plot_roi(rest_brain_mask, bg_img=bg_img, title="Rest brain mask")

show()
