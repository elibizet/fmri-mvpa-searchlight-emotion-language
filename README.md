# fMRI MVPA Searchlight – Emotion & Language

This repository contains the analysis code for my Master’s thesis:

**“Decoding Modality-Independent Emotional Representations in Language-Related Brain Networks”**
Master’s Programme in Language and AI
Stockholm University

## Project Overview

This project investigates how emotional categories are represented in the human brain, and whether such representations are **modality-independent** across sensory modalities such as vision and audition.

Using **multivariate pattern analysis (MVPA)** and **searchlight decoding**, the study examines whether **language-related brain regions** contribute to abstract representations of emotional meaning beyond sensory-specific processing.

## Main Research Questions

* Do emotional representations generalize across modalities, for example from video to audio?
* Do language-related brain regions encode modality-independent emotional information?
* Does task engagement influence decoding performance and abstraction?

## Methods

* **Searchlight analysis** using `nilearn.decoding.SearchLight`
* **Classifier:** Linear Support Vector Machine (SVM)
* **Cross-validation:** Leave-One-Group-Out (LOGO), session-based
* **Analyses:**

  * Within-modal decoding
  * Cross-modal decoding
* **Statistical validation:** Permutation testing with label shuffling

## Repository Structure

```text
scripts/
figures/
README.md
.gitignore
requirements.txt
```

## Analysis Pipeline

```text
Input fMRI data
      │
      ▼
Mask creation and refinement
(create_binary_mask.py → binary_mask.py → final_masks.py)
      │
      ▼
Quality control
(alignment_checking.py, visual_check.py)
      │
      ├─────────────────────────────┐
      │                             │
      ▼                             ▼
Within-modal MVPA              Cross-modal MVPA
(within_modal_mvpa.py)         (mvpa_cross_modal_LOGO.py)
      │                             │
      └──────────────┬──────────────┘
                     ▼
          Whole-brain searchlight
         (searchlight_raw_values.py)
                     ▼
           Permutation testing
        (mvpa_LOGO_perm_testing.py)
                     ▼
              Statistical results
```

## Core Scripts

### Masking and preprocessing

#### `create_binary_mask.py`

Creates an initial binary mask for a region or image of interest.

**Example**

```bash
python scripts/create_binary_mask.py
```

#### `binary_mask.py`

Processes and refines binary masks used in decoding analyses.

**Example**

```bash
python scripts/binary_mask.py
```

#### `final_masks.py`

Generates the final masks used for MVPA and searchlight analyses.

**Example**

```bash
python scripts/final_masks.py
```

### Quality control and visualization

#### `alignment_checking.py`

Checks whether functional and anatomical images are correctly aligned.

**Example**

```bash
python scripts/alignment_checking.py
```

#### `visual_check.py`

Performs visual inspection of output maps and intermediate results.

**Example**

```bash
python scripts/visual_check.py
```

### Decoding analyses

#### `within_modal_mvpa.py`

Runs within-modal MVPA, for example training and testing within the same modality.

**Example**

```bash
python scripts/within_modal_mvpa.py --subject sub-001
```

#### `mvpa_cross_modal_LOGO.py`

Runs cross-modal MVPA using Leave-One-Group-Out cross-validation.

**Example**

```bash
python scripts/mvpa_cross_modal_LOGO.py --subject sub-001
```

#### `searchlight_raw_values.py`

Runs whole-brain searchlight decoding and saves unthresholded accuracy maps.

**Example**

```bash
python scripts/searchlight_raw_values.py --subject sub-001
```

#### `mvpa_LOGO_perm_testing.py`

Runs permutation testing for MVPA analyses to assess statistical significance.

**Example**

```bash
python scripts/mvpa_LOGO_perm_testing.py --subject sub-001 --n-perms 100
```

## Reproducibility

Due to data privacy and size constraints, raw and processed fMRI data are not included in this repository.
However, all analysis scripts are provided to support reproducibility given appropriate data access.

## System Description 

MVPA framework
Searchlight decoding \cite{etzel2013searchlight} radius/feature space: 8mm (voxels within the searchlight sphere)
Classifier type: Linear SVM (C=1 (default value) and default Scikit-learn settings.
Evaluation method: Leave-One-Group-Out()
Software stack: Python, Nilearn, Scikit-learn, MATLAB, MRIcroGL, SPM152 and NeuroQuery \cite{dockes2020neuroquery}
NumPy, Pandas, NiBabel and Matplotlib.
Infrastructure: Workstation at SUBIC and personal computer. 

## Results Data

Full MVPA results are available in the `results/` folder as CSV files:

- `mvpa_happiness_anger.csv`
- `mvpa_anxiety_sadness.csv`

These files contain all decoding results across tasks, masks, and conditions.

## Notes

Additional scripts for:

* whole-brain permutation searchlight
* max-statistic correction

will be added in future updates.

## Author

**Elisabeth Bizet**
MSc in Language and AI
Stockholm University

Focus: neuroimaging, MVPA, emotion, and language processing


