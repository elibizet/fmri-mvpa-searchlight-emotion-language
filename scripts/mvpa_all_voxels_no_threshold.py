import os, glob, time, json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn.maskers import NiftiMasker
from nilearn import plotting
from nilearn.datasets import load_mni152_template

from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# PARAMETERS
VALID_EMOTIONS = [
    "anger", "disgust", "fear", "sadness",
    "interest", "happiness", "pride", "relief", "neutral"
]

VALID_MODALITIES = ["video", "audio", "audiovisual"]


# 1) SUBJECT MASK
def make_subject_mask(sub_path, output_name="subject_mask.nii.gz"):
    """Create a subject-wide mask by intersecting all session masks."""
    mask_files = glob.glob(
        os.path.join(sub_path, "ses-*", "BAS1", "**", "mask.nii*"),  # change BAS1/BAS2 if needed
        recursive=True
    )
    if not mask_files:
        raise FileNotFoundError(f"No mask.nii found under {sub_path}")

    first_mask = nib.load(mask_files[0])
    mask_data = first_mask.get_fdata() > 0

    for mf in mask_files[1:]:
        mask_data &= (nib.load(mf).get_fdata() > 0)

    out_path = os.path.join(sub_path, output_name)
    nib.save(
        nib.Nifti1Image(mask_data.astype(np.uint8), first_mask.affine, first_mask.header),
        out_path
    )
    print(f"Saved subject mask: {out_path} ({int(mask_data.sum())} voxels)")
    return out_path


# 2) LOAD BETAS + LABELS
def load_betas_and_labels(sub_path, task="passive"):
    """
    Load beta-maps and parse labels.
    Assumes label format: run_task_emotion_modality
    """
    all_rows = []
    pattern = os.path.join(sub_path, "ses-*", "BAS1", "**", "regressor_labels.csv")  # change BAS1/BAS2 if needed

    for csv_path in glob.glob(pattern, recursive=True):
        folder = os.path.dirname(csv_path)
        beta_files = sorted(glob.glob(os.path.join(folder, "beta_*.nii")))
        if not beta_files:
            continue

        df = pd.read_csv(csv_path, header=None, names=["label"])
        df = df.iloc[:len(beta_files)].copy()
        df["beta_file"] = beta_files

        parts = df["label"].astype(str).str.split("_", expand=True)
        if parts.shape[1] < 4:
            continue

        df["run"] = parts.iloc[:, 0]
        df["task"] = parts.iloc[:, 1]
        df["emotion"] = parts.iloc[:, 2]
        df["modality"] = parts.iloc[:, 3]

        df = df[df["task"] == task]
        df = df[df["emotion"].isin(VALID_EMOTIONS)]

        df["session"] = df["beta_file"].apply(
            lambda p: next(
                (part for part in os.path.normpath(p).split(os.sep) if part.startswith("ses-")),
                "unknown"
            )
        )

        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        raise FileNotFoundError(f"No valid beta-maps found for {sub_path}")

    merged = pd.concat(all_rows, ignore_index=True)

    print(f"Loaded {len(merged)} beta-maps for {os.path.basename(sub_path)}")
    print("Modalities:", merged["modality"].value_counts().to_dict())
    print("Sessions:", merged["session"].nunique())
    return merged


# 3) FEATURE EXTRACTION (RAW)
def extract_features_raw(df, mask_path, mask_cache="./mask_cache"):
    """
    Convert beta-maps into voxel feature matrix WITHOUT standardization.
    """
    masker = NiftiMasker(
        mask_img=mask_path,
        standardize=False,
        memory=mask_cache,
        dtype="float32",
        verbose=0,
    )
    print(f"Extracting voxel patterns from {len(df)} beta-maps...")
    start = time.time()
    X = masker.fit_transform(df["beta_file"])
    print(f"Masking complete in {time.time() - start:.1f}s. Shape: {X.shape}")
    return X, masker


# 4) LOGO WITHIN-MODALITY MVPA
def run_logo_within_modality(
    X, emotions, modalities, sessions,
    modality_filter, out_dir, title_prefix
):
    os.makedirs(out_dir, exist_ok=True)

    keep = (modalities == modality_filter)
    Xk = X[keep]
    yk = np.array(emotions)[keep]
    gk = np.array(sessions)[keep]

    if Xk.shape[0] < 2:
        raise ValueError(f"Not enough samples for modality='{modality_filter}' (n={Xk.shape[0]})")

    n_folds = len(np.unique(gk))
    print(f"\n[{modality_filter}] Samples={Xk.shape[0]} | Sessions(folds)={n_folds}")

    le = LabelEncoder()
    y_enc = le.fit_transform(yk)

    clf = LinearSVC(
        penalty="l2",
        class_weight="balanced",
        dual=False,
        tol=1e-3,
        max_iter=50000,
        random_state=42,
    )

    cv = LeaveOneGroupOut()
    splits = cv.split(Xk, y_enc, gk)

    y_pred = cross_val_predict(clf, Xk, y_enc, cv=splits, n_jobs=1)
    acc = float(np.mean(y_pred == y_enc))
    print(f"[{modality_filter}] LOGO accuracy: {acc:.3f} (chance≈{1/len(le.classes_):.3f})")

    cm = confusion_matrix(y_enc, y_pred, labels=np.arange(len(le.classes_)))
    per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    per_class_df = pd.DataFrame({"emotion": le.classes_, "accuracy": per_class_acc})
    per_class_csv = os.path.join(out_dir, f"{title_prefix}_{modality_filter}_per_class_accuracy.csv")
    per_class_df.to_csv(per_class_csv, index=False)

    fig = plt.figure(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        cmap="Blues", ax=plt.gca(), colorbar=False
    )
    plt.title(f"{title_prefix} | {modality_filter} | LOGO")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_png = os.path.join(out_dir, f"{title_prefix}_{modality_filter}_confusion_matrix.png")
    plt.savefig(cm_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "modality": modality_filter,
        "n_samples": int(Xk.shape[0]),
        "n_sessions_folds": int(n_folds),
        "n_classes": int(len(le.classes_)),
        "accuracy": acc,
        "chance": float(1 / len(le.classes_)),
        "classes": le.classes_.tolist()
    }
    summary_json = os.path.join(out_dir, f"{title_prefix}_{modality_filter}_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    return acc, cm, le, (Xk, y_enc, gk), clf


# 5) FULL DESCRIPTIVE WEIGHT MAPS
def save_full_weight_map(masker, fitted_clf, out_png, out_nii=None, title=None):
    """
    Fit classifier on ALL samples for a modality, then save and plot
    the FULL unthresholded weight map.

    For multiclass LinearSVC: use mean(abs(coef)) across classes.
    """
    coef = fitted_clf.coef_

    if coef.ndim == 2 and coef.shape[0] > 1:
        w = np.mean(np.abs(coef), axis=0)
    else:
        w = coef.ravel()

    weight_img = masker.inverse_transform(w)

    if out_nii is not None:
        nib.save(weight_img, out_nii)
        print(f"Saved full weight map NIfTI: {out_nii}")

    bg = load_mni152_template()
    display = plotting.plot_stat_map(
        weight_img,
        bg_img=bg,
        title=title or "SVM full weights",
        display_mode="ortho"
    )
    display.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved full weight map PNG: {out_png}")


# RUNNER
def main():
    sub_path = r"D:\singleN_betas\sub-001"  # change to your subject path
    base_out_dir = r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\Results\MVPA results\MVPA passive task\BAS1"
    out_dir = os.path.join(
        base_out_dir,
        os.path.basename(sub_path),
        "mvpa_within_modality_nine_emotions_passive_task" # change name if needed
    )

    task = "passive"   # change to "recog" if needed
    MAKE_WEIGHT_MAPS = True

    mask_path = os.path.join(sub_path, "subject_mask.nii.gz")
    if not os.path.exists(mask_path):
        mask_path = make_subject_mask(sub_path)

    df = load_betas_and_labels(sub_path, task=task)

    print("Unique modality labels in your data:", sorted(df["modality"].unique().tolist()))

    X, masker = extract_features_raw(df, mask_path)

    emotions = df["emotion"].to_numpy()
    modalities = df["modality"].to_numpy()
    sessions = df["session"].to_numpy()

    title_prefix = f"{os.path.basename(sub_path)}_{task}_within_modality_RAW_LOGO"

    results = {}
    for m in VALID_MODALITIES:
        if m not in np.unique(modalities):
            print(f"Skipping modality '{m}' (not present).")
            continue

        acc, cm, le, packed, base_clf = run_logo_within_modality(
            X, emotions, modalities, sessions,
            modality_filter=m,
            out_dir=out_dir,
            title_prefix=title_prefix
        )
        results[m] = acc

        if MAKE_WEIGHT_MAPS:
            Xk, yk_enc, gk = packed

            clf_full = LinearSVC(
                penalty="l2",
                class_weight="balanced",
                dual=False,
                tol=1e-3,
                max_iter=50000,
                random_state=42,
            )
            clf_full.fit(Xk, yk_enc)

            wm_png = os.path.join(out_dir, f"{title_prefix}_{m}_weightmap_FULL.png")
            wm_nii = os.path.join(out_dir, f"{title_prefix}_{m}_weightmap_FULL.nii.gz")

            save_full_weight_map(
                masker=masker,
                fitted_clf=clf_full,
                out_png=wm_png,
                out_nii=wm_nii,
                title=f"{title_prefix} | {m} | full weights"
            )

    print("\nDONE. Accuracies:", results)
    print("Results folder:", out_dir)


if __name__ == "__main__":
    main()