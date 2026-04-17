"""This script performs cross-modal leave-one-session-out (LOSO) MVPA decoding
using a specified ROI mask. It trains a Linear SVM on one modality (e.g. audio)
and tests on another modality (e.g. video) for each session left out."""


from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn.maskers import NiftiMasker
from nilearn import plotting
from nilearn.datasets import load_mni152_template

from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# QUIET WARNINGS
#warnings.filterwarnings("ignore", message=".*Generation of a mask has been requested.*")
#warnings.filterwarnings("ignore", message=".*imgs are being resampled to the mask_img resolution.*")
#warnings.filterwarnings("ignore", message=".*NaNs or infinite values are present.*")


# SETTINGS
MODEL = "BAS2"
TASK = "passive" # "passive" or "recog"
SUBJECT = "sub-001"

VALID_EMOTIONS = ["happiness", "anger"]   # "anxiety", "sadness" "happiness", "anger"
VALID_MODALITIES = ["audio", "video", "audiovisual"]

CROSS_MODAL_PAIRS = [
    ("audio", "video"),
    ("video", "audio"),
]

BASE_DATA_DIR = Path(r"D:\singleN_betas")
BASE_OUT_DIR = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\NEW MVPA results\happiness vs anger\rest_brain_mask"
)

MASK_PATH = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\aligned_rest_of_brain_mask.nii.gz"
)

#MASK_PATH = Path(
#    r"D:\singleN_betas\sub-001\ses-01\BAS2\mask.nii"
#)

MAKE_WEIGHT_MAP = True
WEIGHT_TOP_PERCENT = False


# LOAD BETAS + LABELS
def load_betas_and_labels(sub_path: Path, model: str, task: str) -> pd.DataFrame:
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


# CROSS-MODAL LOSO
def run_cross_modal_loso(
    X: np.ndarray,
    emotions: np.ndarray,
    modalities: np.ndarray,
    sessions: np.ndarray,
    train_modality: str,
    test_modality: str,
    out_dir: Path,
    prefix: str,
):
    """
    Leave-one-session-out cross-modal decoding.

    Example for audio -> video:
    - leave out ses-05
    - train on AUDIO from all sessions except ses-05
    - test on VIDEO from ses-05
    - repeat for all sessions
    """
    unique_sessions = np.unique(sessions)
    pair_name = f"{train_modality}_to_{test_modality}"

    # label encoding over all relevant samples from both modalities
    relevant = np.isin(modalities, [train_modality, test_modality])
    le = LabelEncoder()
    le.fit(emotions[relevant])

    all_true = []
    all_pred = []
    all_test_sessions = []
    fold_rows = []

    last_clf = None

    for test_session in unique_sessions:
        train_mask = (modalities == train_modality) & (sessions != test_session)
        test_mask = (modalities == test_modality) & (sessions == test_session)

        X_train = X[train_mask]
        y_train = emotions[train_mask]

        X_test = X[test_mask]
        y_test = emotions[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Skipping fold {test_session}: missing train/test samples.")
            continue

        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        clf = LinearSVC(
            penalty="l2",
            class_weight="balanced",
            dual=False,
            tol=1e-3,
            max_iter=50000,
            random_state=42,
        )

        clf.fit(X_train, y_train_enc)
        y_pred_enc = clf.predict(X_test)

        fold_acc = float(np.mean(y_pred_enc == y_test_enc))

        fold_rows.append({
            "session": test_session,
            "n_train_samples": int(len(X_train)),
            "n_test_samples": int(len(X_test)),
            "accuracy": fold_acc,
        })

        all_true.extend(y_test_enc.tolist())
        all_pred.extend(y_pred_enc.tolist())
        all_test_sessions.extend([test_session] * len(y_test_enc))

        last_clf = clf

    if len(all_true) == 0:
        raise ValueError(f"No valid LOSO folds produced results for {pair_name}")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    acc = float(np.mean(all_pred == all_true))
    chance = 1.0 / len(le.classes_)

    print(f"\n{train_modality} -> {test_modality} | LOSO")
    print(f"Correct:       {int(np.sum(all_pred == all_true))}/{len(all_true)}")
    print(f"Accuracy:      {acc:.3f}")
    print(f"Chance:        {chance:.3f}")
    print("Pred counts:   ", pd.Series(le.inverse_transform(all_pred)).value_counts().to_dict())

    # confusion matrix
    cm = confusion_matrix(all_true, all_pred, labels=np.arange(len(le.classes_)))
    per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    per_class_df = pd.DataFrame({
        "emotion": le.classes_,
        "accuracy": per_class_acc
    })
    per_class_df.to_csv(
        out_dir / f"{prefix}_{pair_name}_per_class_accuracy.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        cmap="Greens",
        ax=ax,
        colorbar=False
    )
    ax.set_title(f"{prefix} | {train_modality} → {test_modality} | LOSO")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{prefix}_{pair_name}_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # predictions
    pred_df = pd.DataFrame({
        "true_emotion": le.inverse_transform(all_true),
        "predicted_emotion": le.inverse_transform(all_pred),
        "test_modality": test_modality,
        "session": all_test_sessions,
    })
    pred_df.to_csv(
        out_dir / f"{prefix}_{pair_name}_predictions.csv",
        index=False
    )

    # fold accuracies
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(
        out_dir / f"{prefix}_{pair_name}_session_accuracy.csv",
        index=False
    )

    summary = {
        "subject": SUBJECT,
        "model": MODEL,
        "task": TASK,
        "train_modality": train_modality,
        "test_modality": test_modality,
        "n_total_predictions": int(len(all_true)),
        "n_sessions": int(len(unique_sessions)),
        "n_classes": int(len(le.classes_)),
        "classes": le.classes_.tolist(),
        "accuracy": acc,
        "chance": chance,
        "cv": "Leave-one-session-out cross-modal",
    }

    with open(out_dir / f"{prefix}_{pair_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return acc, last_clf


# WEIGHT MAP
def save_weight_map(
    masker,
    fitted_clf,
    out_dir: Path,
    base_name: str,
    title: str,
    top_percent: float = 1.0,
):
    coef = fitted_clf.coef_

    if coef.ndim == 2 and coef.shape[0] > 1:
        weights = np.mean(np.abs(coef), axis=0)
    else:
        weights = coef.ravel()

    bg = load_mni152_template()

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

    print(f"Saved weight maps for {base_name}")


# MAIN
def main():
    sub_path = BASE_DATA_DIR / SUBJECT
    mask_label = MASK_PATH.name.replace(".nii.gz", "").replace(".nii", "")

    out_dir = (
        BASE_OUT_DIR
        / "cross_modal_loso_mvpa"
        / mask_label
        / MODEL
        / SUBJECT
        / f"{TASK}_task"
        / f"{'_vs_'.join(VALID_EMOTIONS)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning cross-modal LOSO MVPA")
    print("Subject:  ", SUBJECT)
    print("Task:     ", TASK)
    print("Mask:     ", MASK_PATH)
    print("Out dir:  ", out_dir)

    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Mask not found: {MASK_PATH}")

    df = load_betas_and_labels(sub_path, model=MODEL, task=TASK)
    print("\nUnique modality labels found:", sorted(df["modality"].unique().tolist()))
    print("\nSamples per session and modality:")
    print(df.groupby(["session", "modality"]).size())

    X, masker = extract_features(df, MASK_PATH)

    emotions = df["emotion"].to_numpy()
    modalities = df["modality"].to_numpy()
    sessions = df["session"].to_numpy()

    prefix = f"{SUBJECT}_{MODEL}_{TASK}_cross_modal_loso"

    results = []

    for train_modality, test_modality in CROSS_MODAL_PAIRS:
        if train_modality not in modalities or test_modality not in modalities:
            print(f"Skipping {train_modality} -> {test_modality}: modality missing.")
            continue

        acc, fitted_clf = run_cross_modal_loso(
            X=X,
            emotions=emotions,
            modalities=modalities,
            sessions=sessions,
            train_modality=train_modality,
            test_modality=test_modality,
            out_dir=out_dir,
            prefix=prefix,
        )

        results.append({
            "train_modality": train_modality,
            "test_modality": test_modality,
            "accuracy": acc
        })

        if MAKE_WEIGHT_MAP and fitted_clf is not None:
            base_name = f"{prefix}_{train_modality}_to_{test_modality}"
            save_weight_map(
                masker=masker,
                fitted_clf=fitted_clf,
                out_dir=out_dir,
                base_name=base_name,
                title=f"{train_modality} → {test_modality} | LOSO",
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