"""
Code for creating the language binary mask from LanA
Here img > 0.3
means:
voxels above 0.3 → 1
voxels 0.3 or below → 0
"""

from pathlib import Path
from nilearn.image import load_img, math_img
import numpy as np

# Paths
atlas_path = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\SPM\LanA_n806.nii"
)

output_path = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\SPM\language_mask_p03.nii.gz"
)

# Load probabilistic atlas
atlas = load_img(str(atlas_path))

# Check atlas values
data = atlas.get_fdata()
print("Min / Max in atlas:", data.min(), data.max())

# Threshold probabilistic atlas at > 0.3 and binarize
language_mask = math_img("(img > 0.3).astype('int32')", img=atlas)

# Save binary mask
language_mask.to_filename(str(output_path))

# Sanity check
mask_data = language_mask.get_fdata()
print("Unique values in mask:", np.unique(mask_data))
print("Number of voxels:", int(mask_data.sum()))
print(f"Saved to: {output_path}")
