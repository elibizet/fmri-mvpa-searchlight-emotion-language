#searchlight analysis example
"""Minimal example skeleton (adapt to my data and nilearn version)"""

from nilearn.decoding import SearchLight
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

# X_imgs: 4D nifti image or list of 3D images (n_samples x X x Y x Z)
# y: labels array with length n_samples
# mask_img: nifti mask limiting computation

cv = StratifiedKFold(n_splits=5)
estimator = SVC(kernel='linear')

searchlight = SearchLight(
    mask_img=mask_img,
    radius=5.0,            # mm
    estimator=estimator,
    scoring='accuracy',
    cv=cv,
    n_jobs=4,
    verbose=1
)

searchlight.fit(X_imgs, y)

# The fitted object contains the searchlight results; check your nilearn version
# common attribute names: searchlight.scores_ (3D Nifti image) or similar
scores_img = searchlight.scores_
scores_img.to_filename('searchlight_scores.nii.gz')