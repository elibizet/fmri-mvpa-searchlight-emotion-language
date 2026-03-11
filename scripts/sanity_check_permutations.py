import nibabel as nib
import numpy as np

img = nib.load("perm_001_map.nii.gz")
data = img.get_fdata()

print("Mean accuracy:", np.mean(data))
print("Max accuracy:", np.max(data))