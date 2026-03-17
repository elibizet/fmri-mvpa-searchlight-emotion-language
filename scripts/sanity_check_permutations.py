"""Before trusting permutation results, 
neuroscientists always check:

Are permutation maps near chance?
For binary classification:
chance accuracy ≈ 0.5
If permutation maps are too high ~ 0.65, 
something is wrong (usually CV leakage)."""

import nibabel as nib
import numpy as np

img = nib.load("perm_001_map.nii.gz")
data = img.get_fdata()

print("Mean accuracy:", np.mean(data))
print("Max accuracy:", np.max(data))