"""
This script performs within-modality MVPA 
decoding using a Leave-One-Session-Out cross-validation scheme
within the Master thesis project 27/3/2026:
*selects one modality, e.g. audio
*within that modality only
*trains on all sessions except one
*tests on the left-out session
*repeats for every session
*combines predictions across folds
For the weight maps, the script does this:
* after CV is done, it fits one final LinearSVC on all samples of that modality
* save the weights from that full model
* accuracy = from LOSO CV
* weight map = descriptive map from full-data fit
This way we get a more stable weight map that is not based on just one fold."""


from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from nilearn.maskers import NiftiMasker
from nilearn import plotting
from nilearn.datasets import load_mni152_template

from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#warnings.filterwarnings("ignore", message="NaNs or infinite values are present")

# SETTINGS
MODEL = "BAS2"                  # "BAS1" or "BAS2"
TASK = "passive"                # "passive" or "recog"
SUBJECT = "sub-001"

# Choose only the emotions you want to decode
VALID_EMOTIONS = ["happiness", "anger"] # , "anxiety", "sadness"  , 

# Choose which modalities to run within-modality decoding on
VALID_MODALITIES =  ["audio", "video","audiovisual"] #["audio", "video"] 

BASE_DATA_DIR = Path(r"D:\singleN_betas")
BASE_OUT_DIR = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\NEW MVPA results\happiness vs anger\rest_brain_mask"
)

MASK_PATH = Path( # change mask path here to run with different masks
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\aligned_rest_of_brain_mask.nii.gz"
)


#MASK_PATH = Path(
#    r"D:\singleN_betas\sub-001\ses-01\BAS2\mask.nii"
#)

MAKE_WEIGHT_MAP = True
WEIGHT_TOP_PERCENT = False

# LOAD BETAS + LABELS
def load_betas_and_labels(sub_path: Path, model: str, task: str) -> pd.DataFrame:
    """
    Load beta maps and parse labels from regressor_labels.csv.
    Expected label format: run_task_emotion_modality
    """
    rows = []
    pattern = f"ses-*/{model}/**/regressor_labels.csv"

    for csv_path in sub_path.glob(pattern):
        folder = csv_path.parent
        beta_files = sorted(folder.glob("beta_*.nii"))

        if not beta_files:
            continue

        df = pd.read_csv(csv_path, header=None, names=["label"])
        df = df.iloc[:len(beta_files)].copy()
        df["beta_file"] = [str(p) for p in beta_files]

        parts = df["label"].astype(str).str.split("_", expand=True)
        if parts.shape[1] < 4:
            continue

        df["run"] = parts.iloc[:, 0]
        df["task"] = parts.iloc[:, 1]
        df["emotion"] = parts.iloc[:, 2]
        df["modality"] = parts.iloc[:, 3]

        df = df[df["task"] == task]
        df = df[df["emotion"].isin(VALID_EMOTIONS)]
        df = df[df["modality"].isin(VALID_MODALITIES)]

        df["session"] = df["beta_file"].apply(
            lambda p: next(
                (part for part in Path(p).parts if part.startswith("ses-")),
                "unknown"
            )
        )

        if not df.empty:
            rows.append(df)

    if not rows:
        raise FileNotFoundError(
            f"No valid beta maps found for subject={sub_path.name}, model={model}, task={task}"
        )

    merged = pd.concat(rows, ignore_index=True)

    print(f"\nLoaded {len(merged)} beta maps")
    print("Subject:   ", sub_path.name)
    print("Model:     ", model)
    print("Task:      ", task)
    print("Emotions:  ", merged["emotion"].value_counts().to_dict())
    print("Modalities:", merged["modality"].value_counts().to_dict())
    print("Sessions:  ", merged["session"].nunique())

    return merged

# FEATURE EXTRACTION
def extract_features(df: pd.DataFrame, mask_path: Path):
    masker = NiftiMasker(
        mask_img=str(mask_path),
        standardize=False,
        dtype="float32",
        verbose=0,
    )

    print(f"\nExtracting features from {len(df)} beta maps...")

    beta_files = df["beta_file"].tolist()

    masker.fit()
    X = masker.transform(beta_files)

    X = np.nan_to_num(X)

    print("Feature matrix shape:", X.shape)

    return X, masker

# WITHIN-MODALITY DECODING (Leave-One-Session-Out)
def run_within_modality_decoding(
    X: np.ndarray,
    emotions: np.ndarray,
    modalities: np.ndarray,
    sessions: np.ndarray,
    modality_filter: str,
    out_dir: Path,
    prefix: str,
):
    """
    Within-modality decoding using Leave-One-Session-Out cross-validation.
    Example:
        train on audio from all sessions except one
        test on audio from the left-out session
    """

    keep = modalities == modality_filter

    Xk = X[keep]
    yk = emotions[keep]
    gk = sessions[keep]
    print(f"\n{modality_filter} samples:", Xk.shape)
    
    if len(Xk) == 0:
        raise ValueError(f"No samples found for modality '{modality_filter}'")

    unique_sessions = np.unique(gk)
    if len(unique_sessions) < 2:
        raise ValueError(
            f"Need at least 2 sessions for Leave-One-Session-Out CV, found {len(unique_sessions)}"
        )

    le = LabelEncoder()
    yk_enc = le.fit_transform(yk)

    clf = LinearSVC(
        penalty="l2",
        class_weight="balanced",
        dual=False,
        tol=1e-3,
        max_iter=50000,
        random_state=42,
    )

    cv = LeaveOneGroupOut()

    print(f"\nWithin-modality: {modality_filter}")
    print(f"Samples:   {len(Xk)}")
    print(f"Sessions:  {len(unique_sessions)}")
    print(f"Classes:   {list(le.classes_)}")

    y_pred = cross_val_predict(
        clf,
        Xk,
        yk_enc,
        cv=cv,
        groups=gk,
        n_jobs=1,
    )
    # --- DEBUG CHECK ---
    n_correct = int(np.sum(y_pred == yk_enc))

    print(f"\nDEBUG ({modality_filter})")
    print("Correct:", n_correct, "/", len(yk_enc))
    print("True counts:", pd.Series(yk).value_counts().to_dict())
    print("Pred counts:", pd.Series(le.inverse_transform(y_pred)).value_counts().to_dict())
    
    acc = float(np.mean(y_pred == yk_enc))
    chance = 1.0 / len(le.classes_)

    print(f"Accuracy:  {acc:.3f}")
    print(f"Chance:    {chance:.3f}")

    # confusion matrix
    cm = confusion_matrix(yk_enc, y_pred, labels=np.arange(len(le.classes_)))
    per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    modality_name = modality_filter

    per_class_df = pd.DataFrame({
        "emotion": le.classes_,
        "accuracy": per_class_acc
    })
    per_class_df.to_csv(
        out_dir / f"{prefix}_{modality_name}_per_class_accuracy.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        cmap="Purples",
        ax=ax,
        colorbar=False
    )
    ax.set_title(f"{prefix} | {modality_filter} | LOSO-CV")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{prefix}_{modality_name}_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # predictions
    pred_df = pd.DataFrame({
        "true_emotion": yk,
        "predicted_emotion": le.inverse_transform(y_pred),
        "modality": modality_filter,
        "session": gk
    })
    pred_df.to_csv(
        out_dir / f"{prefix}_{modality_name}_predictions.csv",
        index=False
    )

    # session-wise accuracy
    session_rows = []
    for sess in unique_sessions:
        sess_mask = gk == sess
        sess_acc = float(np.mean(y_pred[sess_mask] == yk_enc[sess_mask]))
        session_rows.append({
            "session": sess,
            "n_samples": int(np.sum(sess_mask)),
            "accuracy": sess_acc
        })

    session_df = pd.DataFrame(session_rows)
    session_df.to_csv(
        out_dir / f"{prefix}_{modality_name}_session_accuracy.csv",
        index=False
    )

    # summary
    summary = {
        "subject": SUBJECT,
        "model": MODEL,
        "task": TASK,
        "modality": modality_filter,
        "n_samples": int(len(Xk)),
        "n_sessions": int(len(unique_sessions)),
        "n_classes": int(len(le.classes_)),
        "classes": le.classes_.tolist(),
        "accuracy": acc,
        "chance": chance,
        "cv": "LeaveOneGroupOut",
        "groups": "session",
    }

    with open(out_dir / f"{prefix}_{modality_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # fit one final model on all samples of this modality
    # only for descriptive weight maps
    clf_full = LinearSVC(
        penalty="l2",
        class_weight="balanced",
        dual=False,
        tol=1e-3,
        max_iter=50000,
        random_state=42,
    )
    clf_full.fit(Xk, yk_enc)

    return acc, clf_full


# WEIGHT MAP
def save_weight_map(
    masker,
    fitted_clf,
    out_dir: Path,
    base_name: str,
    title: str,
    top_percent: float = 1.0
):
    """
    Save descriptive weight maps from a fitted LinearSVC.
    """
    coef = fitted_clf.coef_

    if coef.ndim == 2 and coef.shape[0] > 1:
        weights = np.mean(np.abs(coef), axis=0)
    else:
        weights = coef.ravel()

    bg = load_mni152_template()

    # full map
    full_img = masker.inverse_transform(weights)
    full_nii = out_dir / f"{base_name}_weights_full.nii.gz"
    full_png = out_dir / f"{base_name}_weights_full.png"

    full_img.to_filename(full_nii)

    display = plotting.plot_stat_map(
        full_img,
        bg_img=bg,
        title=f"{title} | full weights",
        display_mode="ortho"
    )
    display.savefig(full_png, dpi=250, bbox_inches="tight")
    display.close()

    # top % map
    if 0 < top_percent < 100:
        thr = np.percentile(np.abs(weights), 100 - top_percent)
        top_weights = weights.copy()
        top_weights[np.abs(top_weights) < thr] = 0.0

        top_img = masker.inverse_transform(top_weights)
        label = str(top_percent).replace(".", "p")

        top_nii = out_dir / f"{base_name}_weights_top{label}pct.nii.gz"
        top_png = out_dir / f"{base_name}_weights_top{label}pct.png"

        top_img.to_filename(top_nii)

        display = plotting.plot_stat_map(
            top_img,
            bg_img=bg,
            title=f"{title} | top {top_percent}%",
            display_mode="ortho"
        )
        display.savefig(top_png, dpi=250, bbox_inches="tight")
        display.close()

   # print(f"Saved weight maps for {base_name}")

# MAIN
def main():
    sub_path = BASE_DATA_DIR / SUBJECT

    out_dir = (
        BASE_OUT_DIR
        / "within_modality_mvpa"
        / MODEL
        / SUBJECT
        / f"{TASK}_task"
        / f"{'_vs_'.join(VALID_EMOTIONS)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n:D")
    print("Running within-modality MVPA")
    print("Subject:   ", SUBJECT)
  #  print("Model:     ", MODEL)
    print("Task:      ", TASK)
    print("Emotions:  ", VALID_EMOTIONS)
    print("Mask:      ", MASK_PATH)
    print("Out dir:   ", out_dir)
    print(":D\n")

    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Mask not found: {MASK_PATH}")

    df = load_betas_and_labels(sub_path, model=MODEL, task=TASK)
    print("\nUnique modality labels found:", sorted(df["modality"].unique().tolist()))

    #print("\nSamples per session and modality:")
    #print(df.groupby(["session", "modality"]).size())

    X, masker = extract_features(df, MASK_PATH)
    print("FULL feature matrix shape:", X.shape)

    emotions = df["emotion"].to_numpy()
    modalities = df["modality"].to_numpy()
    sessions = df["session"].to_numpy()

    prefix = f"{SUBJECT}_{MODEL}_{TASK}_within_modality"

    results = []

    for modality in VALID_MODALITIES:
        if modality not in modalities:
            print(f"Skipping modality '{modality}': not present.")
            continue

        acc, fitted_clf = run_within_modality_decoding(
            X=X,
            emotions=emotions,
            modalities=modalities,
            sessions=sessions,
            modality_filter=modality,
            out_dir=out_dir,
            prefix=prefix,
        )

        results.append({
            "modality": modality,
            "accuracy": acc
        })

        if MAKE_WEIGHT_MAP:
            base_name = f"{prefix}_{modality}"
            save_weight_map(
                masker=masker,
                fitted_clf=fitted_clf,
                out_dir=out_dir,
                base_name=base_name,
                title=f"{modality} | LOSO-CV",
                top_percent=WEIGHT_TOP_PERCENT,
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / f"{prefix}_all_results.csv", index=False)

    print("\nDONE")
    print(results_df if not results_df.empty else "No results were produced.")
    print("\nResults folder:")
    print(out_dir)


if __name__ == "__main__":
    main()