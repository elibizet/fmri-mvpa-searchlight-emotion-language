"""This script creates a binary mask from a NeuroQuery z-map by thresholding at z > 3."""

from pathlib import Path
from nilearn.image import load_img, math_img
import numpy as np

# Paths 
zmap_path = Path(r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\language_network_neuroquery.nii.gz") # change to visual/auditory/language as needed
output_path = Path(r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\language_mask_z3.nii.gz")

# Load the z-map
z_map = load_img(str(zmap_path))

# Threshold at z > 3 and binarize
binary_mask = math_img("(img > 3).astype('int32')", img=z_map)

# Save the binary mask
binary_mask.to_filename(str(output_path))

# Sanity check
data = binary_mask.get_fdata()
print("Unique values in mask:", np.unique(data))
print("ROI voxels:", int((data > 0).sum()))

print(f"Binary mask saved to: {output_path}")