# Searchlight Analysis Within Modalities

This repository contains the Python code used for my master’s thesis:

**“[Decoding Modality-Independent Emotional Representations in Language-Related Brain Networks]”**  
Master’s Programme in [Language and AI]  
Stockholm University

## Overview
The code implements searchlight-based multivariate pattern analysis (MVPA)
on fMRI data using Nilearn. Analyses focus on within-modality emotion
classification (e.g., happiness vs. anger) across different sensory modalities.

## Repository contents
- `whole_brain_searchlight.py`  
  Whole-brain searchlight analysis¨.  Within-modality classification script (happiness vs. anger)

- `single_axial_slice_searchlight.py`  
  Searchlight analysis on a single axial slice (for visualization/debugging)

## Requirements
The code was developed and tested with:
- Python ≥ 3.9
- nilearn
- numpy
- scipy
- scikit-learn

(Exact versions may be specified later if needed.)

## Usage
Scripts are intended to be run from the command line or an IDE such as VS Code.
Paths to input data and output directories must be adapted to the local environment.
