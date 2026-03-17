"""
Permutation searchlight script (simplified teaching version)

What it does:
1. Load beta images for happiness vs anger
2. Run the REAL searchlight
3. Shuffle emotion labels
4. Run the searchlight again
5. Save permutation maps

No max-statistic correction here.
This script only generates the maps.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from nilearn.image import load_img, concat_imgs, mean_img, new_img_like
from nilearn.decoding import SearchLight
from nilearn.plotting import plot_stat_map

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# USER PARAMETERS

SUBJECT = "sub-001"

BETAS_ROOT = Path(r"D:\singleN_betas")

OUTPUT_DIR = Path(
r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\Results\Permutation_debug"
)

MODALITY = "audiovisual"

EMOTIONS = ["happiness", "anger"]

N_PERMUTATIONS = 5

SEARCHLIGHT_RADIUS = 8  # mm

N_JOBS = 4


# STEP 1 — LOAD DATA

print("\nLoading data...")

subj_dir = BETAS_ROOT / SUBJECT

rows = []

# loop through sessions
for ses in sorted(subj_dir.glob("ses-*")):

    bas = ses / "BAS2"
    labels_file = bas / "regressor_labels.csv"

    labels = pd.read_csv(labels_file, header=None)[0].tolist()

    for i, label in enumerate(labels, start=1):

        # keep only passive trials
        if "_passive_" not in label:
            continue

        parts = label.split("_")

        run = int(parts[0].split("-")[1])
        emotion = parts[2].lower()
        modality = parts[3].lower()

        if modality != MODALITY:
            continue

        if emotion not in EMOTIONS:
            continue

        beta_path = bas / f"beta_{i:04d}.nii"

        rows.append(
            dict(
                session=ses.name,
                run=run,
                emotion=emotion,
                beta=str(beta_path)
            )
        )

meta = pd.DataFrame(rows)

print(meta.head())


# STEP 2 — BUILD DATA MATRICES

print("\nBuilding dataset...")

# load all beta images as a 4D image
X = concat_imgs(meta.beta.tolist())

# labels
y = meta.emotion.values

# groups for cross-validation (sessions)
groups = meta.session.values

# STEP 3 — LOAD BRAIN MASK

mask = load_img(subj_dir / "ses-01" / "BAS2" / "_mask.nii")

bg_img = mean_img(X)


# STEP 4 — CREATE CLASSIFIER

print("\nCreating classifier...")

classifier = make_pipeline(
    StandardScaler(),
    LinearSVC()
)

# STEP 5 — CREATE SEARCHLIGHT OBJECT

print("\nInitializing searchlight...")

searchlight = SearchLight(
    mask_img=mask,
    process_mask_img=mask,
    radius=SEARCHLIGHT_RADIUS,
    estimator=classifier,
    cv=LeaveOneGroupOut(),
    scoring="accuracy",
    n_jobs=N_JOBS,
    verbose=1
)


# STEP 6 — RUN REAL SEARCHLIGHT

print("\nRunning REAL searchlight...")

searchlight.fit(X, y, groups=groups)

real_scores = searchlight.scores_

# save map
real_img = new_img_like(mask, real_scores)

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

real_img.to_filename(OUTPUT_DIR / "real_searchlight_map.nii.gz")

plot = plot_stat_map(
    real_img,
    bg_img=bg_img,
    threshold=None,
    title="Real searchlight map"
)

plot.savefig(OUTPUT_DIR / "real_map.png")

plot.close()


# STEP 7 — PERMUTATION FUNCTION

def shuffle_labels_within_session(y, groups):

    """
    Shuffle emotion labels inside each session.
    This preserves session structure.
    """

    y_perm = y.copy()

    for s in np.unique(groups):

        idx = np.where(groups == s)[0]

        y_perm[idx] = np.random.permutation(y_perm[idx])

    return y_perm


# STEP 8 — RUN PERMUTATIONS

print("\nRunning permutations...")

for p in range(N_PERMUTATIONS):

    print(f"\nPermutation {p+1}/{N_PERMUTATIONS}")

    # shuffle labels
    y_perm = shuffle_labels_within_session(y, groups)

    # run searchlight
    searchlight.fit(X, y_perm, groups=groups)

    perm_scores = searchlight.scores_

    perm_img = new_img_like(mask, perm_scores)

    perm_img.to_filename(
        OUTPUT_DIR / f"perm_{p+1:03d}_map.nii.gz"
    )

    plot = plot_stat_map(
        perm_img,
        bg_img=bg_img,
        threshold=None,
        title=f"Permutation {p+1}"
    )

    plot.savefig(
        OUTPUT_DIR / f"perm_{p+1:03d}.png"
    )

    plot.close()


print("\nAll permutations finished!")