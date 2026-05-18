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

See the four final masks, all binary (0/1), with matching dimensions and affine to the functional data in the folder * masks

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

* Binary masks created 
* MVPA framework
* Searchlight decoding \cite{etzel2013searchlight} radius/feature space: 8mm (voxels within the searchlight sphere)
* Classifier type: Linear SVM (C=1 (default value) and default Scikit-learn settings.
* Evaluation method: Leave-One-Group-Out()
* Software stack: Python, Nilearn, Scikit-learn, MATLAB, MRIcroGL, SPM152 and NeuroQuery \cite{dockes2020neuroquery}
  NumPy, Pandas, NiBabel and Matplotlib.
* Infrastructure: Workstation at SUBIC and personal computer. 

## Results Data

Full MVPA results are available in the `results/` folder as CSV files:

- `mvpa_happiness_anger.csv`
- `mvpa_anxiety_sadness.csv`

These files contain all decoding results across tasks, masks, and conditions.

## Additional Statistical Scripts

### `whole_brain_searchlight_permutation.py`

Runs whole-brain searchlight decoding using the same settings as the real analysis, but with globally shuffled labels to generate permutation-based null maps.

Main features:

- same searchlight settings as the real analysis
- 8 mm searchlight radius
- Leave-One-Group-Out cross-validation (session-based)
- global label shuffling across sessions
- saves full unthresholded permutation maps
- preserves class counts and CV structure

The script does **not** compute corrected thresholds or p-values.  
Those steps are handled separately by `maxstat.py`.

**Example**

```bash
python scripts/whole_brain_searchlight_permutation.py \
    --subject sub-001 \
    --n-perms 52 \
    --seed 42
```

---

### `maxstat.py`

Performs permutation-based maximum statistic correction using the saved permutation searchlight maps.

The script:

- loads the real searchlight map
- loads all permutation maps
- extracts the maximum accuracy from each permutation
- builds the null max-statistic distribution
- computes the family-wise error corrected threshold
- applies thresholding to the real map
- computes corrected voxelwise p-values

Saved outputs include:

- thresholded searchlight map
- significant voxel mask
- corrected p-value map
- CSV file with max-stat values
- JSON summary file
- voxelwise accuracy histogram

**Example**

```bash
python scripts/maxstat.py
```

## Author

**Elisabeth Bizet**
MSc in Language and AI
Stockholm University

Focus: neuroimaging, MVPA, emotion, and language processing


