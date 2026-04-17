"""
ROI-level permutation testing within the Master thesis MVPA pipeline.

within-modality:
- LOSO (Leave-One-Session-Out)
- permute labels within the selected modality, keeping session structure intact

cross-modal:
- LOSO by session
- leave one session out
- train on modality A from remaining sessions
- test on modality B in the left-out session
- permute TRAINING labels only, within training sessions
- keep test labels unchanged
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from nilearn.maskers import NiftiMasker

from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import LabelEncoder


# QUIET WARNINGS
#warnings.filterwarnings("ignore", message=".*Generation of a mask has been requested.*")
#warnings.filterwarnings("ignore", message=".*imgs are being resampled to the mask_img resolution.*")
#warnings.filterwarnings("ignore", message=".*NaNs or infinite values are present.*")

# SETTINGS
MODEL = "BAS2"
TASK = "passive" # "passive" or "recog"
SUBJECT = "sub-001"

VALID_EMOTIONS = ["happiness", "anger"] # "anxiety", "sadness" "happiness", "anger"
VALID_MODALITIES = ["audio", "video", "audiovisual"]

CROSS_MODAL_PAIRS = [
    ("audio", "video"),
    ("video", "audio"),
]

BASE_DATA_DIR = Path(r"D:\singleN_betas")
BASE_OUT_DIR = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\NEW MVPA results\happiness vs anger\rest_brain_mask\permutations_passive"
)

# change mask here
MASK_PATH = Path(
    r"C:\Users\Fabian\OneDrive - Stockholm University\Desktop\Eli\Master thesis\masks\aligned_rest_of_brain_mask.nii.gz"
)

#MASK_PATH = Path(
#    r"D:\singleN_betas\sub-001\ses-01\BAS2\mask.nii"
#)

N_PERMUTATIONS = 100
RANDOM_SEED = 42

RUN_WITHIN_MODALITY = True
RUN_CROSS_MODAL = True

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

    beta_files = df["beta_file"].tolist()
    masker.fit()
    X = masker.transform(beta_files)
    X = np.nan_to_num(X)

    print("Feature matrix shape:", X.shape)
    return X, masker


# HELPERS
def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def fit_clf() -> LinearSVC:
    return LinearSVC(
        penalty="l2",
        class_weight="balanced",
        dual=False,
        tol=1e-3,
        max_iter=50000,
        random_state=42,
    )


def permute_within_groups(
    y: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Permute labels separately within each group.
    Useful for respecting session structure.
    """
    y_perm = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        y_perm[idx] = rng.permutation(y_perm[idx])
    return y_perm


# WITHIN-MODALITY: REAL + PERMUTATION
def within_modality_real_and_perm(
    X: np.ndarray,
    emotions: np.ndarray,
    modalities: np.ndarray,
    sessions: np.ndarray,
    modality_filter: str,
    n_permutations: int,
    rng: np.random.Generator,
):
    keep = modalities == modality_filter

    Xk = X[keep]
    yk = emotions[keep]
    gk = sessions[keep]

    if len(Xk) == 0:
        raise ValueError(f"No samples found for modality '{modality_filter}'")

    le = LabelEncoder()
    yk_enc = le.fit_transform(yk)

    clf = fit_clf()
    cv = LeaveOneGroupOut()

    # real accuracy
    y_pred_real = cross_val_predict(
        clf,
        Xk,
        yk_enc,
        cv=cv,
        groups=gk,
        n_jobs=1,
    )
    real_acc = accuracy_score_np(yk_enc, y_pred_real)

    # permutation distribution
    perm_accs = np.zeros(n_permutations, dtype=float)

    for i in range(n_permutations):
        y_perm = permute_within_groups(yk_enc, gk, rng)

        y_pred_perm = cross_val_predict(
            clf,
            Xk,
            y_perm,
            cv=cv,
            groups=gk,
            n_jobs=1,
        )
        perm_accs[i] = accuracy_score_np(y_perm, y_pred_perm)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  {modality_filter} permutation {i + 1}/{n_permutations}")

    p_value = (np.sum(perm_accs >= real_acc) + 1) / (n_permutations + 1)

    return {
        "analysis": "within_modality",
        "modality": modality_filter,
        "real_accuracy": real_acc,
        "chance": 1.0 / len(le.classes_),
        "n_samples": int(len(Xk)),
        "n_sessions": int(len(np.unique(gk))),
        "n_classes": int(len(le.classes_)),
        "classes": le.classes_.tolist(),
        "n_permutations": int(n_permutations),
        "perm_mean_accuracy": float(np.mean(perm_accs)),
        "perm_std_accuracy": float(np.std(perm_accs)),
        "perm_95th_accuracy": float(np.percentile(perm_accs, 95)),
        "p_value": float(p_value),
        "perm_accuracies": perm_accs.tolist(),
    }


# CROSS-MODAL LOSO: REAL + PERMUTATION
def cross_modal_loso_real_and_perm(
    X: np.ndarray,
    emotions: np.ndarray,
    modalities: np.ndarray,
    sessions: np.ndarray,
    train_modality: str,
    test_modality: str,
    n_permutations: int,
    rng: np.random.Generator,
):
    """
    Cross-modal permutation testing with leave-one-session-out.

    Real analysis:
    - for each left-out session:
      train on modality A from all other sessions
      test on modality B from the left-out session

    Permutations:
    - shuffle TRAINING labels only
    - shuffle within training sessions
    - keep test labels unchanged
    """
    unique_sessions = np.unique(sessions)

    relevant = np.isin(modalities, [train_modality, test_modality])
    le = LabelEncoder()
    le.fit(emotions[relevant])

    # REAL ACCURACY
    all_true_real = []
    all_pred_real = []

    for test_session in unique_sessions:
        train_mask = (modalities == train_modality) & (sessions != test_session)
        test_mask = (modalities == test_modality) & (sessions == test_session)

        X_train = X[train_mask]
        y_train = emotions[train_mask]
        g_train = sessions[train_mask]

        X_test = X[test_mask]
        y_test = emotions[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        clf = fit_clf()
        clf.fit(X_train, y_train_enc)
        y_pred_enc = clf.predict(X_test)

        all_true_real.extend(y_test_enc.tolist())
        all_pred_real.extend(y_pred_enc.tolist())

    if len(all_true_real) == 0:
        raise ValueError(
            f"No valid LOSO folds for {train_modality} -> {test_modality}"
        )

    all_true_real = np.array(all_true_real)
    all_pred_real = np.array(all_pred_real)
    real_acc = accuracy_score_np(all_true_real, all_pred_real)

    # PERMUTATION DISTRIBUTION
    perm_accs = np.zeros(n_permutations, dtype=float)

    for i in range(n_permutations):
        all_true_perm = []
        all_pred_perm = []

        for test_session in unique_sessions:
            train_mask = (modalities == train_modality) & (sessions != test_session)
            test_mask = (modalities == test_modality) & (sessions == test_session)

            X_train = X[train_mask]
            y_train = emotions[train_mask]
            g_train = sessions[train_mask]

            X_test = X[test_mask]
            y_test = emotions[test_mask]

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            y_train_enc = le.transform(y_train)
            y_test_enc = le.transform(y_test)

            # permute ONLY training labels, within training sessions
            y_train_perm = permute_within_groups(y_train_enc, g_train, rng)

            clf_perm = fit_clf()
            clf_perm.fit(X_train, y_train_perm)
            y_pred_perm = clf_perm.predict(X_test)

            all_true_perm.extend(y_test_enc.tolist())
            all_pred_perm.extend(y_pred_perm.tolist())

        all_true_perm = np.array(all_true_perm)
        all_pred_perm = np.array(all_pred_perm)

        perm_accs[i] = accuracy_score_np(all_true_perm, all_pred_perm)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  {train_modality}->{test_modality} permutation {i + 1}/{n_permutations}")

    p_value = (np.sum(perm_accs >= real_acc) + 1) / (n_permutations + 1)

    return {
        "analysis": "cross_modal_loso",
        "train_modality": train_modality,
        "test_modality": test_modality,
        "real_accuracy": real_acc,
        "chance": 1.0 / len(le.classes_),
        "n_total_predictions": int(len(all_true_real)),
        "n_sessions": int(len(unique_sessions)),
        "n_classes": int(len(le.classes_)),
        "classes": le.classes_.tolist(),
        "n_permutations": int(n_permutations),
        "perm_mean_accuracy": float(np.mean(perm_accs)),
        "perm_std_accuracy": float(np.std(perm_accs)),
        "perm_95th_accuracy": float(np.percentile(perm_accs, 95)),
        "p_value": float(p_value),
        "perm_accuracies": perm_accs.tolist(),
    }


# SAVE HELPERS
def save_json(obj: dict, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# MAIN
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    sub_path = BASE_DATA_DIR / SUBJECT
    mask_label = MASK_PATH.name.replace(".nii.gz", "").replace(".nii", "")

    out_dir = (
        BASE_OUT_DIR
        / mask_label
        / MODEL
        / SUBJECT
        / f"{TASK}_task"
        / f"{'_vs_'.join(VALID_EMOTIONS)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning ROI permutation MVPA")
    print("Subject:        ", SUBJECT)
    print("Model:          ", MODEL)
    print("Task:           ", TASK)
    print("Emotions:       ", VALID_EMOTIONS)
    print("Mask:           ", MASK_PATH)
    print("Permutations:   ", N_PERMUTATIONS)
    print("Output folder:  ", out_dir)

    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Mask not found: {MASK_PATH}")

    df = load_betas_and_labels(sub_path, model=MODEL, task=TASK)
    X, _ = extract_features(df, MASK_PATH)

    emotions = df["emotion"].to_numpy()
    modalities = df["modality"].to_numpy()
    sessions = df["session"].to_numpy()

    all_rows = []

    # WITHIN-MODALITY
    if RUN_WITHIN_MODALITY:
        print("\n=== WITHIN-MODALITY PERMUTATIONS ===")
        for modality in VALID_MODALITIES:
            if modality not in modalities:
                print(f"Skipping {modality}: not present.")
                continue

            print(f"\nRunning within-modality permutations for {modality}")
            result = within_modality_real_and_perm(
                X=X,
                emotions=emotions,
                modalities=modalities,
                sessions=sessions,
                modality_filter=modality,
                n_permutations=N_PERMUTATIONS,
                rng=rng,
            )

            save_json(
                result,
                out_dir / f"{SUBJECT}_{MODEL}_{TASK}_within_{modality}_permutation_summary.json"
            )

            all_rows.append({
                "analysis": "within_modality",
                "modality_or_pair": modality,
                "real_accuracy": result["real_accuracy"],
                "chance": result["chance"],
                "perm_mean_accuracy": result["perm_mean_accuracy"],
                "perm_95th_accuracy": result["perm_95th_accuracy"],
                "p_value": result["p_value"],
            })

            print(
                f"  {modality}: real={result['real_accuracy']:.3f}, "
                f"perm95={result['perm_95th_accuracy']:.3f}, p={result['p_value']:.4f}"
            )

    # CROSS-MODAL LOSO
    if RUN_CROSS_MODAL:
        print("\n=== CROSS-MODAL LOSO PERMUTATIONS ===")
        for train_modality, test_modality in CROSS_MODAL_PAIRS:
            if train_modality not in modalities or test_modality not in modalities:
                print(f"Skipping {train_modality}->{test_modality}: modality missing.")
                continue

            print(f"\nRunning cross-modal LOSO permutations for {train_modality} -> {test_modality}")
            result = cross_modal_loso_real_and_perm(
                X=X,
                emotions=emotions,
                modalities=modalities,
                sessions=sessions,
                train_modality=train_modality,
                test_modality=test_modality,
                n_permutations=N_PERMUTATIONS,
                rng=rng,
            )

            save_json(
                result,
                out_dir / f"{SUBJECT}_{MODEL}_{TASK}_cross_loso_{train_modality}_to_{test_modality}_permutation_summary.json"
            )

            all_rows.append({
                "analysis": "cross_modal_loso",
                "modality_or_pair": f"{train_modality}_to_{test_modality}",
                "real_accuracy": result["real_accuracy"],
                "chance": result["chance"],
                "perm_mean_accuracy": result["perm_mean_accuracy"],
                "perm_95th_accuracy": result["perm_95th_accuracy"],
                "p_value": result["p_value"],
            })

            print(
                f"  {train_modality}->{test_modality}: real={result['real_accuracy']:.3f}, "
                f"perm95={result['perm_95th_accuracy']:.3f}, p={result['p_value']:.4f}"
            )

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_dir / "all_permutation_results.csv", index=False)

    print("\nDONE")
    print(results_df)
    print("\nSaved to:")
    print(out_dir)


if __name__ == "__main__":
    main()