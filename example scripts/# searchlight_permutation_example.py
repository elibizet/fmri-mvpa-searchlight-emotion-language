# searchlight_permutation_example.py
import numpy as np
from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from nilearn.decoding import SearchLight
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import time

# Parameters (reduce for quick test)
n_permutations = 100        # increase for production (500-5000)
radius = 4.0
n_jobs = 1                  # set >1 to speed up (requires more memory)
cv = StratifiedKFold(n_splits=5)

# Load data (same setup as A)
haxby = datasets.fetch_haxby(n_subjects=1)
func_img = haxby.func[0]
mask_vt = haxby.mask_vt[0]
import pandas as pd
beh = pd.read_csv(haxby.session_target[0], sep=' ')
labels = beh['labels'].values
cond_mask = np.logical_or(labels == 'face', labels == 'house')
idx = np.where(cond_mask)[0].tolist()
X_imgs = image.index_img(func_img, idx)
y = np.where(labels[cond_mask] == 'face', 1, 0)

# Fit the observed searchlight
est = make_pipeline(StandardScaler(), SVC(kernel='linear'))
searchlight = SearchLight(mask_img=mask_vt, radius=radius, estimator=est,
                          scoring='accuracy', cv=cv, n_jobs=n_jobs, verbose=0)

print("Fitting observed (non-permuted) searchlight")
t0 = time.time()
searchlight.fit(X_imgs, y)
print("Observed fit time: %.1f s" % (time.time() - t0))
observed_scores = searchlight.scores_.copy()   # 1D array over mask voxels
n_vox = observed_scores.size

# Prepare storage for permutation results
perm_scores = np.zeros((n_permutations, n_vox), dtype=float)
perm_max = np.zeros(n_permutations, dtype=float)

# Re-run searchlight for each permutation
rng = np.random.RandomState(42)
for p in range(n_permutations):
    permuted_y = rng.permutation(y)
    # Reuse the same searchlight object: fit with permuted labels
    searchlight.fit(X_imgs, permuted_y)
    cur_scores = searchlight.scores_.copy()
    perm_scores[p, :] = cur_scores
    perm_max[p] = cur_scores.max()
    if (p + 1) % 10 == 0 or p == 0:
        print("Completed permutation %d / %d" % (p + 1, n_permutations))

# 1) Voxelwise (uncorrected) p-values
# p_voxel_uncorrected = proportion of permuted scores >= observed score
p_voxel_uncorr = (1 + (perm_scores >= observed_scores).sum(axis=0)) / (1 + n_permutations)

# 2) Max-stat FWER-corrected p-values
# For each voxel, compute p = proportion of permuted max-stat >= observed_voxel
p_voxel_fwer = (1 + (perm_max[:, None] >= observed_scores[None, :]).sum(axis=0)) / (1 + n_permutations)

# Convert p-value vectors back to 3D images for saving/visualization
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_vt).fit()

# Save observed accuracy map and corrected p-value map
from nilearn.image import new_img_like
observed_img = masker.inverse_transform(observed_scores)
uncorr_p_img = masker.inverse_transform(p_voxel_uncorr)
fwer_p_img = masker.inverse_transform(p_voxel_fwer)

observed_img.to_filename('searchlight_observed_acc.nii.gz')
uncorr_p_img.to_filename('searchlight_uncorrected_p.nii.gz')
fwer_p_img.to_filename('searchlight_fwer_corrected_p.nii.gz')

print("Saved observed accuracy map and p-value maps.")