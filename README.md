# Searchlight Analysis Within Modalities

This repository contains the Python code used for my master’s thesis:

**“[Decoding Modality-Independent Emotional Representations in Language-Related Brain Networks]”**  
Master’s Programme in [Language and AI]  
Stockholm University

# fMRI Searchlight & MVPA Analysis of Emotion and Language

It contains code developed as part of a Master’s thesis investigating how emotions are represented and decoded in the human brain, with a particular focus on their relationship to language processing systems.

## Project Overview

The project applies multivariate pattern analysis (MVPA) and searchlight decoding techniques to fMRI data to identify brain regions that carry information about emotional states across different sensory modalities (e.g., visual and auditory stimuli).

A key aim is to explore whether emotion representations overlap with or engage classical language-related brain regions, contributing to the understanding of emotions as communicative and cognitively structured signals.

## Methods

* Whole-brain **searchlight analysis** using `nilearn.decoding.SearchLight`
* **Multivariate pattern analysis (MVPA)** using linear Support Vector Machines (SVM)
* Cross-validation with `LeaveOneGroupOut` to ensure generalization across runs
* Modality-specific decoding (e.g., audio vs. video conditions)
* Permutation testing for statistical validation

## Repository Structure

* `scripts/`
  Main analysis pipelines, including searchlight and MVPA workflows

* `example scripts/`
  Supporting scripts for testing, visualization, and data inspection

* `.gitignore`
  Excludes large neuroimaging files and environment-specific folders

## Technologies

* Python
* Nilearn
* scikit-learn
* NumPy, Pandas
* NiBabel
* Matplotlib

## Reproducibility

Due to data privacy and size constraints, raw and processed fMRI data are not included in this repository. However, all analysis scripts are provided to ensure reproducibility given appropriate data access.

## Example Usage

```bash
python whole_brain_searchlight.py --subject sub-001
```

## Research Context

This work contributes to ongoing research in cognitive neuroscience and computational linguistics by combining:

* Brain decoding techniques
* Emotion representation modeling
* Language and communication systems in the brain

The project aligns with current efforts to understand how abstract mental states such as emotions are encoded in distributed neural patterns and how these patterns relate to higher-level cognitive functions like language.

## Key Skills Demonstrated

- Machine learning on neuroimaging data  
- Multivariate decoding (MVPA)  
- Python-based scientific computing  
- Experimental design and analysis  

## Author
Elisabeth Bizet
MSc in Language and AI  
Focus: Neuroimaging, MVPA, emotion, and language processing

