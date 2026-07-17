# Brain Tumor MRI Classifier — with Uncertainty Estimation & Explainability

A deep learning pipeline for 4-class brain tumor classification (Glioma / Meningioma / Pituitary / No Tumor) from MRI scans, built with transfer learning, validated for reproducibility across multiple random seeds, and extended with Monte Carlo Dropout uncertainty estimation and Grad-CAM explainability — deployed as a live, interactive demo.

**🔗 Live Demo:** [https://huggingface.co/spaces/Swati1925/brain-tumor-classifier](https://huggingface.co/spaces/Swati1925/brain-tumor-classifier)

⚠️ **This is a research/educational project, NOT a certified medical device. It must not be used for actual clinical diagnosis.**

---

## Overview

Most student brain-tumor-classification projects stop at "train a model, report accuracy." This project goes further in three directions:

1. **Rigorous validation** — every major result is validated across 5 random seeds (not just a single run), after a mid-project discovery that single-run comparisons can be misleading due to training stochasticity
2. **Uncertainty estimation** — Monte Carlo Dropout flags predictions the model is unsure about, rather than silently returning a possibly-wrong answer
3. **Explainability** — Grad-CAM verifies *why* the model made a prediction, and was used to generate a concrete, testable hypothesis for the model's main weakness

---

## Results Summary

### Architecture comparison (5-seed validated, held-out test set)

| Model | Test Accuracy (mean ± std) | Glioma Recall (mean ± std) |
|---|---|---|
| **ResNet50** (selected as final model) | **93.37% ± 0.58%** | **79.30% ± 2.09%** |
| DenseNet121 | 93.13% ± 0.51% | 77.25% ± 1.62% |
| EfficientNetB0 | 89.55% ± 0.32% | 68.45% ± 2.03% |
| Baseline CNN (from scratch) | 92.86% (single run) | — |

**Key finding:** Single-run validation accuracy initially suggested ResNet50 and EfficientNetB0 performed similarly (96.4% vs 95.9%). Multi-seed test-set validation revealed EfficientNetB0 is actually consistently ~4 points weaker overall and ~11 points weaker on Glioma — demonstrating why architecture selection should rely on multi-seed, held-out test performance rather than single validation numbers.

### The Glioma problem

Across all experiments, Glioma was the consistently hardest class to classify correctly — driven by genuine visual similarity to Meningioma and a subset of cases with no single, clear tumor mass. Four targeted retraining interventions were tried:

| Intervention | Result |
|---|---|
| Class-weighted loss | No improvement (79.5%, down from 80.25%) |
| + Deeper fine-tuning | No improvement (80.0%), increased overfitting |
| Data augmentation | Inconclusive — high run-to-run variance (73–82%) triggered a reproducibility investigation |
| Test-time augmentation | Small improvement (+1 point) |

**None of the retraining-based fixes meaningfully resolved the issue** — this motivated a shift from "make the model always right" to "make the model know when it might be wrong."

### Monte Carlo Dropout — uncertainty estimation

| | Mean Uncertainty |
|---|---|
| Correct predictions | 0.0073 |
| Incorrect predictions | 0.0907 (**12.4x higher**) |
| Glioma incorrect | 0.0815 (**8.0x higher** than Glioma correct) |

Using a simple uncertainty threshold, flagging just **13.9%** of test cases for manual review catches **82%** of all model errors — reducing silent (undetected) errors from 97 to 16. This uncertainty-based approach was far more effective than any retraining-based intervention at reducing dangerous silent misdiagnoses.

### Grad-CAM — explainability

Comparing correct vs. incorrect Glioma predictions revealed a clear qualitative pattern: correct predictions show a single, large, concentrated activation blob resembling an actual tumor mass, while incorrect predictions show diffuse, scattered attention with no comparable signal — suggesting the model learned a genuine "coherent mass" detector, and its failures cluster around small/subtle/diffuse tumors rather than spurious pattern-matching. No evidence of shortcut learning (e.g., attention on scan borders/artifacts) was found.

### External validation

Tested on the SARTAJ Kaggle dataset (a different source not fully overlapping with training data) without any retraining:

| Class | External Accuracy |
|---|---|
| Meningioma | 93.91% |
| No Tumor | 97.14% |
| Pituitary | 93.24% |
| Glioma | 26.00%* |

*Consistent with a documented labeling-quality caveat in this external dataset's Glioma folder, and/or a genuine distribution shift toward ring-enhancing tumor presentations underrepresented in training (visually confirmed via sample inspection). Meningioma/No-Tumor/Pituitary results closely matched internal test performance, indicating genuine generalization for those classes.

---

## Project Structure

brain-tumor-mri-classifier/
├── notebooks/                  # Main analysis notebook (all sections)
├── app.py                      # Gradio deployment app (Hugging Face Spaces)
├── requirements.txt
├── results/                    # Confusion matrices, classification reports,
│                                  multi-seed summaries, Grad-CAM samples
├── DEVLOG.md                   # Day-by-day development journal
└── README.md

---

## Methodology

1. **EDA** — verified class balance, inspected samples across classes, identified mixed MRI planes/sequences as a source of visual variance, confirmed preprocessing requirements (resize, RGB conversion) against published literature on this dataset
2. **Baseline** — simple 3-block CNN trained from scratch, establishing a "before" number
3. **Transfer learning** — ResNet50, EfficientNetB0, and DenseNet121, each fine-tuned via a 2-phase strategy (frozen base → unfrozen top layers at low learning rate)
4. **Full evaluation** — confusion matrix, per-class precision/recall/F1, ROC-AUC, on a held-out test set never touched during training or hyperparameter selection
5. **Reproducibility validation** — after discovering that single-run results could vary significantly due to unseeded training randomness (weight init, dropout, augmentation, shuffling — not just the data split), all three architectures were retrained across 5 seeds, with results reported as mean ± std
6. **Error analysis** — systematic, one-variable-at-a-time debugging of the Glioma weakness (see Results Summary above)
7. **Monte Carlo Dropout** — 30 stochastic forward passes per prediction (dropout kept active at inference, batch norm kept in eval mode), using predictive standard deviation as an uncertainty score
8. **Grad-CAM** — gradient-based class activation mapping on the final convolutional block, compared between correct and incorrect predictions
9. **External validation** — tested on a separate Kaggle dataset (SARTAJ) with care taken to avoid data leakage from datasets that overlapped with training sources
10. **Deployment** — Gradio interface (prediction + confidence + uncertainty flag + Grad-CAM heatmap) on Hugging Face Spaces (ZeroGPU)

---

## Limitations

- **Dataset scale** — ~7,000 images from a combination of three public sources; not validated against a large, single-institution clinical dataset
- **Patient-level leakage was not verified** — unlike some comparable studies, this dataset's metadata does not clearly support patient-level train/test splitting; it's possible (though not confirmed) that multiple slices from the same patient exist across splits
- **Glioma remains the weakest class**, with external validation suggesting the model under-recognizes ring-enhancing tumor presentations specifically
- **Not validated for real clinical use** — this is an educational/portfolio project, not a certified diagnostic tool
- **Augmentation's true effect on Glioma recall remains statistically unresolved** — a single-run improvement could not be reliably reproduced, and full multi-seed validation of this specific intervention was not completed due to time constraints
- **Future work identified but not implemented:** ROI-based cropping combined with higher input resolution (to better capture small/subtle tumors), based on the Grad-CAM finding that failures cluster around cases lacking a large, coherent tumor mass

---

## Tech Stack

Python, PyTorch, torchvision (ResNet50/EfficientNetB0/DenseNet121), scikit-learn, Grad-CAM (custom implementation), Gradio, Hugging Face Spaces (ZeroGPU), Google Colab (T4 GPU), Kaggle API

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle, CC BY 4.0) — combination of Figshare, SARTAJ, and Br35H source datasets, ~7,023 images across 4 classes.

## Development Log

See [DEVLOG.md](./DEVLOG.md) for a detailed, day-by-day account of the build process, including bugs encountered and fixed (a BatchNorm/Dropout interaction bug in MC Dropout, a data-leakage risk avoided in external validation, and the reproducibility issue that reshaped the project's second half).
