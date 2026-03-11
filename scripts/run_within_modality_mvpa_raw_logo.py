# run: python run_within_modality_mvpa_raw_logo.py

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


# SETTINGS
MODEL = "BAS2"              # "BAS1" or "BAS2"
TASK = "recog"            # "passive" or "recog"
SUBJECT = "sub-003"         # change to your subject ID (e.g., "sub-001")

VALID_EMOTIONS = ["anger", "disgust", "fear", "sadness", "interest", "happiness", "pride", "relief", "neutral"] # "anger", "disgust", "fear", "sadness", #"interest", "happiness", "pride", "relief", "neutral" adapt if I want to include/exclude certain emotions
VALID_MODALITIES = ["video", "audio", "audiovisual"]

MAKE_WEIGHT_MAPS = True
SAVE_FULL_WEIGHT_MAPS = True
SAVE_TOP_WEIGHT_MAPS = True
WEIGHT_TOP_PERCENT = 1.0

BASE_DATA_DIR = r"D:\singleN_betas"
BASE_OUT_DIR = r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\NEW MVPA results\Recognition task\All nine emotions"


# 1) SUBJECT MASK
def make_subject_mask(sub_path, model, output_name=None):
    """Create a subject-wide mask by intersecting all session masks for one model only."""
    if output_name is None:
        output_name = f"subject_mask_{model}.nii.gz"

    mask_files = glob.glob(
        os.path.join(sub_path, "ses-*", model, "**", "mask.nii*"),
        recursive=True
    )

    if not mask_files:
        raise FileNotFoundError(f"No mask.nii found under {sub_path} for model={model}")

    print(f"\nMasks used to build {output_name}:")
    for mf in mask_files:
        print("  ", mf)

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
def load_betas_and_labels(sub_path, model, task="passive"):
    """
    Load beta-maps and parse labels.
    Assumes label format: run_task_emotion_modality
    """
    all_rows = []
    pattern = os.path.join(sub_path, "ses-*", model, "**", "regressor_labels.csv")

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
        raise FileNotFoundError(
            f"No valid beta-maps found for {sub_path}, model={model}, task={task}"
        )

    merged = pd.concat(all_rows, ignore_index=True)

    print(f"\nLoaded {len(merged)} beta-maps for {os.path.basename(sub_path)}")
    print("Model:", model)
    print("Task:", task)
    print("Modalities:", merged["modality"].value_counts().to_dict())
    print("Sessions:", merged["session"].nunique())
    return merged


# 3) FEATURE EXTRACTION
def extract_features_raw(df, mask_path, mask_cache="./mask_cache"):
    """Convert beta-maps into voxel feature matrix WITHOUT standardization."""
    masker = NiftiMasker(
        mask_img=mask_path,
        standardize=False,
        memory=mask_cache,
        dtype="float32",
        verbose=0,
    )

    print(f"\nExtracting voxel patterns from {len(df)} beta-maps...")
    start = time.time()
    X = masker.fit_transform(df["beta_file"])
    print(f"Masking complete in {time.time() - start:.1f}s. Shape: {X.shape}")
    return X, masker


# 4) LOGO WITHIN-MODALITY MVPA
def run_logo_within_modality(X, emotions, modalities, sessions, modality_filter, out_dir, title_prefix):
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
    per_class_df.to_csv(
        os.path.join(out_dir, f"{title_prefix}_{modality_filter}_per_class_accuracy.csv"),
        index=False
    )

    fig = plt.figure(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        cmap="Blues", ax=plt.gca(), colorbar=False
    )
    plt.title(f"{title_prefix} | {modality_filter} | LOGO")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{title_prefix}_{modality_filter}_confusion_matrix.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    summary = {
        "model": MODEL,
        "task": TASK,
        "modality": modality_filter,
        "n_samples": int(Xk.shape[0]),
        "n_sessions_folds": int(n_folds),
        "n_classes": int(len(le.classes_)),
        "accuracy": acc,
        "chance": float(1 / len(le.classes_)),
        "classes": le.classes_.tolist()
    }
    with open(os.path.join(out_dir, f"{title_prefix}_{modality_filter}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return acc, (Xk, y_enc)


# 5) WEIGHT MAPS
def save_weight_maps(masker, fitted_clf, out_dir, base_name, title_prefix, top_percent=1.0):
    """Save full and optionally thresholded descriptive weight maps."""
    coef = fitted_clf.coef_

    if coef.ndim == 2 and coef.shape[0] > 1:
        w = np.mean(np.abs(coef), axis=0)
    else:
        w = coef.ravel()

    bg = load_mni152_template()

    # FULL MAP
    if SAVE_FULL_WEIGHT_MAPS:
        full_img = masker.inverse_transform(w)

        full_nii = os.path.join(out_dir, f"{base_name}_weightmap_FULL.nii.gz")
        full_png = os.path.join(out_dir, f"{base_name}_weightmap_FULL.png")

        nib.save(full_img, full_nii)

        display = plotting.plot_stat_map(
            full_img,
            bg_img=bg,
            title=f"{title_prefix} | full weights",
            display_mode="ortho"
        )
        display.savefig(full_png, dpi=250, bbox_inches="tight")
        plt.close()

        print(f"Saved full weight map NIfTI: {full_nii}")
        print(f"Saved full weight map PNG:   {full_png}")

    # TOP-X% MAP
    if SAVE_TOP_WEIGHT_MAPS and top_percent is not None and 0 < top_percent < 100:
        thr = np.percentile(np.abs(w), 100 - top_percent)
        w_top = w.copy()
        w_top[np.abs(w_top) < thr] = 0.0

        top_img = masker.inverse_transform(w_top)
        top_label = str(top_percent).replace(".", "p")

        top_nii = os.path.join(out_dir, f"{base_name}_weightmap_top{top_label}pct.nii.gz")
        top_png = os.path.join(out_dir, f"{base_name}_weightmap_top{top_label}pct.png")

        nib.save(top_img, top_nii)

        display = plotting.plot_stat_map(
            top_img,
            bg_img=bg,
            title=f"{title_prefix} | weights (top {top_percent}%)",
            display_mode="ortho"
        )
        display.savefig(top_png, dpi=250, bbox_inches="tight")
        plt.close()

        print(f"Saved top-{top_percent}% weight map NIfTI: {top_nii}")
        print(f"Saved top-{top_percent}% weight map PNG:   {top_png}")


# MAIN
def main():
    sub_path = os.path.join(BASE_DATA_DIR, SUBJECT)

    out_dir = os.path.join(
        BASE_OUT_DIR,
        MODEL,
        SUBJECT,
        f"mvpa_within_modality_{'_'.join(VALID_EMOTIONS)}_{TASK}_task"
    )

    os.makedirs(out_dir, exist_ok=True)

    print("\nSummary of settings:")
    print("Running MVPA")
    print("Subject:", SUBJECT)
    print("Model:", MODEL)
    print("Task:", TASK)
    print("Emotions:", VALID_EMOTIONS)
    #print("========================================")

    # MODEL-SPECIFIC MASK
    mask_filename = f"subject_mask_{MODEL}.nii.gz"
    mask_path = os.path.join(sub_path, mask_filename)

    if not os.path.exists(mask_path):
        mask_path = make_subject_mask(sub_path, model=MODEL, output_name=mask_filename)
    else:
        print(f"\nUsing existing model-specific mask: {mask_path}")

    df = load_betas_and_labels(sub_path, model=MODEL, task=TASK)

    print("\nUnique modality labels in your data:", sorted(df["modality"].unique().tolist()))

    X, masker = extract_features_raw(df, mask_path)

    emotions = df["emotion"].to_numpy()
    modalities = df["modality"].to_numpy()
    sessions = df["session"].to_numpy()

    title_prefix = f"{SUBJECT}_{MODEL}_{TASK}_within_modality_RAW_LOGO"

    results = {}
    for m in VALID_MODALITIES:
        if m not in np.unique(modalities):
            print(f"Skipping modality '{m}' (not present).")
            continue

        acc, packed = run_logo_within_modality(
            X, emotions, modalities, sessions,
            modality_filter=m,
            out_dir=out_dir,
            title_prefix=title_prefix
        )
        results[m] = acc

        if MAKE_WEIGHT_MAPS:
            Xk, yk_enc = packed

            clf_full = LinearSVC(
                penalty="l2",
                class_weight="balanced",
                dual=False,
                tol=1e-3,
                max_iter=50000,
                random_state=42,
            )
            clf_full.fit(Xk, yk_enc)

            base_name = f"{title_prefix}_{m}"
            save_weight_maps(
                masker=masker,
                fitted_clf=clf_full,
                out_dir=out_dir,
                base_name=base_name,
                title_prefix=f"{title_prefix} | {m}",
                top_percent=WEIGHT_TOP_PERCENT
            )

    print("\nDONE. Accuracies:", results)
    print("Results folder:", out_dir)


if __name__ == "__main__":
    main()