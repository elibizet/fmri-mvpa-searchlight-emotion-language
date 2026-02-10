# searchlight_crossmodal_example.py
import numpy as np
import pandas as pd
from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from nilearn.decoding import SearchLight
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit

# Load Haxby
haxby = datasets.fetch_haxby(n_subjects=1)
func_img = haxby.func[0]
mask_vt = haxby.mask_vt[0]
beh = pd.read_csv(haxby.session_target[0], sep=' ')

# Example: use 'labels' for classes and 'chunks' or 'session' for runs if available.
# Inspect columns
print("Behavior columns:", beh.columns)

labels = beh['labels'].values
# If dataset has a 'chunks' or 'session' column, use it; here we try to use 'chunks' or 'chunks' fallback
if 'chunks' in beh.columns:
    runs = beh['chunks'].values
elif 'session' in beh.columns:
    runs = beh['session'].values
else:
    # fallback: consider first half of volumes as run A and second half as run B (toy demo)
    n = len(beh)
    runs = np.array([0 if i < n//2 else 1 for i in range(n)])

# Select two categories to decode (face vs house)
cond_mask = np.logical_or(labels == 'face', labels == 'house')
selected_idx = np.where(cond_mask)[0]
X_imgs = image.index_img(func_img, selected_idx)
y = np.where(labels[cond_mask] == 'face', 1, 0)

# Now create a PredefinedSplit that trains on run 0 (or run A) and tests on run 1 (run B)
runs_selected = runs[cond_mask]
# Define test_fold: -1 entries are used for training-only; non-negative integers define folds.
# We'll build a single test fold: where runs_selected == 1 -> test fold 0; where runs_selected == 0 -> train (-1)
test_fold = np.full(len(runs_selected), -1, dtype=int)
test_fold[runs_selected == 1] = 0  # fold 0 is the test fold containing modality B

ps = PredefinedSplit(test_fold)

# Set up searchlight with PredefinedSplit so that inside each neighborhood the estimator is trained on modality A and tested on modality B
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

est = make_pipeline(StandardScaler(), SVC(kernel='linear'))
searchlight = SearchLight(mask_img=mask_vt, radius=5.0, estimator=est, scoring='accuracy', cv=ps, n_jobs=1, verbose=1)

print("Running cross-modal searchlight (train on runs where test_fold == -1, test on runs where test_fold == 0)")
searchlight.fit(X_imgs, y)

# Save result as 3D nifti
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_vt).fit()
scores_img = masker.inverse_transform(searchlight.scores_)
scores_img.to_filename('searchlight_crossmodal.nii.gz')
print("Saved cross-modal searchlight map: searchlight_crossmodal.nii.gz")