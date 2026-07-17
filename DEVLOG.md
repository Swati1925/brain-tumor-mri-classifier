# Development Log

## Day 1 

### What I did
- Set up GitHub repo, Kaggle API access, and Google Colab environment (T4 GPU)
- Downloaded the Brain Tumor MRI Dataset (Kaggle, masoudnickparvar/brain-tumor-mri-dataset) — 7,023 images across 4 classes (Glioma, Meningioma, Pituitary, No Tumor), pre-split into Training/Testing folders
- **EDA:**
  - Verified class distribution — training set is perfectly balanced (1400 images/class). Noted this is unusually clean for a medical dataset; will still implement weighted loss to demonstrate the technique, since real clinical data is rarely this balanced.
  - Visually inspected multiple samples per class. Found the dataset contains mixed MRI planes (axial, coronal, sagittal) and mixed sequences (T1/T2/FLAIR-like contrast differences) — a realistic but harder setting for the model. Flagged this as a potential source of visual variance the model needs to generalize across.
  - One early "notumor" sample had a visually different texture (edge-detection-like) — investigated further with more samples and confirmed it was an isolated case, not a systematic pattern across the class. Noted as a shortcut-learning risk to re-check later with Grad-CAM.
  - Checked image sizes (highly variable, 192x192 to 1335x1302) and color modes (mixed RGB/grayscale) — confirmed resize to 224x224 + forced RGB conversion is required in preprocessing. Cross-checked this against published papers using the same dataset — matches standard practice (224x224, bicubic resize, grayscale-to-RGB, ImageNet normalization).
- **Baseline CNN (from scratch):**
  - Built a simple 3-block CNN (Conv-ReLU-MaxPool x3 + FC classifier with Dropout 0.5), no pretrained weights
  - 80/20 train/validation split from the Training folder (Testing folder held out untouched for final evaluation later)
  - Trained 10 epochs — final result: Train Acc 94.3%, Val Acc 92.9%
  - Train/val loss and accuracy curves tracked closely together — no signs of overfitting yet
  - Saved model weights to `models/baseline_cnn.pth`

### Key learnings
- `model.train()` vs `model.eval()` changes Dropout behavior — this is also the foundation for Monte Carlo Dropout (uncertainty estimation) planned for Week 3
- A small train/val gap is a good early sign of healthy generalization; will keep monitoring as models get more complex

### Next up
- Week 2: Transfer learning with ResNet50 / EfficientNetB0, compare against this baseline
## Day 2 — [Aaj ki date]

### What I did
- **Transfer Learning (ResNet50):**
  - Loaded pretrained ResNet50 (ImageNet weights), froze all base layers, replaced final fc layer for 4 classes
  - Phase 1 (frozen base, only classifier trained, 5 epochs): Val Acc plateaued at ~86.7% — confirmed frozen ImageNet features alone weren't enough for the MRI domain
  - Phase 2 (unfroze layer4, fine-tuned at lr=0.0001, 5 epochs): Val Acc jumped to 96.4% — clear evidence fine-tuning was necessary, not just feature extraction
  - Saved model to `models/resnet50_finetuned.pth`
- **Transfer Learning (EfficientNetB0):**
  - Same two-phase approach. Phase 1: 91.3% val acc (notably better than ResNet50's Phase 1 — EfficientNet's pretrained features generalized better out of the box)
  - Phase 2 (unfroze last MBConv block + projection layer, lr=0.0001): 95.9% val acc, with only 1.13M trainable params vs ResNet50's 14.9M
  - Saved model to `models/efficientnetb0_finetuned.pth`
- Built comparison table (`results/comparison_table.md`) — Baseline vs ResNet50 vs EfficientNetB0, both accuracy and parameter count

### Key learnings
- Domain gap (ImageNet natural photos → grayscale MRI) means frozen-feature transfer learning alone underperforms a well-tuned baseline; fine-tuning top layers is what unlocks the real gain
- Low learning rate (10x smaller) during fine-tuning prevents destroying pretrained weights ("catastrophic forgetting")
- Accuracy alone isn't the full story — EfficientNetB0 nearly matches ResNet50's accuracy with ~13x fewer trainable parameters, an important tradeoff for real deployment scenarios

### Next up
- Week 3: Full multi-class evaluation suite (confusion matrix, per-class precision/recall/F1, ROC-AUC) on the held-out Testing set, then Monte Carlo Dropout for uncertainty estimation


## Day 3

### What I did — Full Evaluation Suite (Section 5)

- Loaded the held-out Testing set (1,600 images, untouched until now) and evaluated the best model (ResNet50, single run) on it for the first time
- **Test accuracy: 92.94%** — notably lower than the validation accuracy (96.43%). Investigated and concluded this gap (~3.5%) was expected and healthy, not a red flag: val accuracy is always somewhat optimistic since hyperparameter choices are indirectly tuned against it, while the test set is genuinely unseen
- Generated confusion matrix, per-class precision/recall/F1, and macro/weighted averages
- **Key finding:** Overall accuracy hid a major class-specific weakness — Glioma recall was only 80.25% (321/400 correct), while all other classes scored 95%+. Precision for Glioma was high (98.5%) but recall was low — meaning the model was highly confident when it did predict Glioma, but frequently failed to recognize real Glioma cases at all
- Broke down the Glioma errors by clinical severity: 45 cases misclassified as Meningioma (less dangerous — still flagged as *some* tumor) vs. 34 cases misclassified as No-Tumor (most dangerous — a real tumor missed entirely, a false negative)
- Computed per-class ROC-AUC (one-vs-rest): all classes scored 0.96+, including Glioma (0.9665) — indicating the model's underlying probability estimates carried useful signal even where hard classification decisions were failing, an early hint that uncertainty estimation (planned for later) could add real value here
- Searched published literature using the same dataset; confirmed Glioma-Meningioma confusion is a widely reported, genuine phenomenon attributed to overlapping signal intensities and boundary distortions in T1-weighted MRI — not unique to this model

### What I did — Error Analysis & Targeted Improvement, Section 6 (single-run experiments)

Investigated the Glioma weakness with a structured, one-variable-at-a-time approach:

- **Root cause investigation:** Visually inspected misclassified Glioma images. Found tumors were not uniformly small/subtle (several were large and clearly visible yet still missed), and the dataset's mixed scan orientations (axial/coronal/sagittal) were present across failures — consistent with the mixed-orientation risk flagged during EDA on Day 1

- **Attempt 1 — Class-weighted loss** (Glioma weighted 2x in CrossEntropyLoss): Glioma recall 80.25% → 79.50%. **No improvement** — the model became more confident on the same limited pattern-recognition, not better at recognizing new cases

- **Attempt 2 — Weighted loss + deeper fine-tuning** (unfroze `layer3` in addition to `layer4`, lr=0.00005): Recall 80.25% → 80.00%, overall accuracy improved to 94.75%, but training accuracy hit 99.6% — clear overfitting. **No improvement** on the target metric despite touching 22M parameters

- **Attempt 3 — Data augmentation** (stronger rotation, translation, color jitter, reverting to a clean pre-fix checkpoint to isolate the variable): First run gave Glioma recall 82.00% and reduced the dangerous Glioma→Notumor errors from 34 to 15 (a 56% reduction) — the best single-run result. **Mistake:** the trained model was not saved immediately after this result; a subsequent Colab session disconnect (GPU quota exhaustion) permanently lost these weights

- **Attempt 4 — Test-Time Augmentation** (averaging predictions over 5 augmented views at inference, no retraining): Applied to the clean (non-augmented) model. Test accuracy 92.94% → 93.44%, Glioma recall 80.25% → 81.25%, Glioma→Notumor errors 34 → 31. Modest improvement, no retraining cost

### The Reproducibility Crisis

While attempting to reproduce Attempt 3 (having lost the original trained weights), the exact same code and configuration produced a very different result on a second run: Glioma recall 82.00% → 73.00%, a 9-point swing — **larger than the ~2-point "improvement" being attributed to augmentation in the first place.**

**Root cause:** `torch.manual_seed(42)` had only been applied to the train/val split, not to the training process itself (weight initialization, dropout, batch shuffling, augmentation randomness). This meant every reported "improvement" from single-run experiments (Attempts 1–4) could not be trusted as a genuine effect versus random variance.

### Multi-Seed Validation (the fix)

- Implemented a comprehensive `set_seed()` covering `random`, `numpy`, `torch`, `torch.cuda`, and forced deterministic cuDNN behavior
- Retrained all three architectures (ResNet50, EfficientNetB0, and a newly added DenseNet121 — included after literature review showed it performs well on this exact dataset) across 5 seeds each (42, 123, 7, 2024, 99), using three parallel Google/Colab accounts to work around free-tier GPU quota limits, saving every seed's model immediately after training this time

**Results (mean ± std across 5 seeds, held-out test set):**

| Model | Test Accuracy | Glioma Recall |
|---|---|---|
| **ResNet50** | **93.37% ± 0.58%** | **79.30% ± 2.09%** |
| DenseNet121 | 93.13% ± 0.51% | 77.25% ± 1.62% |
| EfficientNetB0 | 89.55% ± 0.32% | 68.45% ± 2.03% |

**Key finding:** Single-run validation accuracy (Day 2) had suggested ResNet50 (96.43%) and EfficientNetB0 (95.89%) were nearly identical. Multi-seed test-set validation revealed EfficientNetB0 is actually consistently, meaningfully weaker — ~4 points lower accuracy and ~11 points lower Glioma recall — showing that single-run validation-set comparisons can be misleading, and that architecture selection should be based on held-out test performance across multiple seeds, not a single validation number.

**Decision: ResNet50 selected as the final model** — best on both metrics, with low variance confirming reliable, reproducible performance.

### Glioma Failure-Mode Analysis (across all 5 ResNet50 seeds)

Reloaded each of the 5 saved seed-models and generated confusion matrices to check whether Glioma's two error types (→Meningioma vs →Notumor) were stable or random:

| Seed | Glioma→Meningioma | Glioma→Notumor |
|---|---|---|
| 42 | 37 | 35 |
| 123 | 42 | 44 |
| 7 | 35 | 45 |
| 2024 | 52 | 45 |
| 99 | 44 | 34 |
| **Average** | **~42** | **~41** |

**Finding:** Glioma misclassifications split almost evenly between the two error types, consistently across independently-seeded runs. This ~50/50 stability (rather than random fluctuation) indicates a genuine, systematic model limitation — not training noise — reinforcing that further hyperparameter tuning alone is unlikely to resolve it, and that uncertainty estimation is a more principled next step than continued blind retraining.

### Key learnings
- **Training is stochastic in more ways than just the data split** — every source of randomness (weight init, dropout, shuffling, augmentation) needs seeding before single-run results can be trusted, especially for small effect sizes
- **Save results immediately after training, not after evaluation** — losing the Attempt 3 model cost significant re-work; this is now a standing habit going forward
- **Validation accuracy alone can mislead model selection** — held-out test-set performance, ideally averaged across multiple seeds, is necessary before declaring one architecture "better" than another
- **Aggregate metrics hide clinically important detail** — recall alone didn't distinguish "dangerous" false negatives from "less severe" misclassifications; confusion-matrix-level, error-type-specific analysis was necessary
- Multiple targeted interventions (weighted loss, deeper fine-tuning) can fail to move a metric — this is itself a valid, useful finding, not a wasted effort, if the reasoning and results are documented honestly


Chose not to pursue further TTA/augmentation experiments after reviewing recent literature (Medeiros, 2026) showing TTA degrades accuracy in 11/12 tested medical imaging configurations due to distribution shift — reinforcing that the reproducibility-variance we observed wasn't unique to this project, and that MC Dropout-based uncertainty estimation is a more principled path forward than continued ad-hoc augmentation tuning

### Next up
- Monte Carlo Dropout for uncertainty estimation on the final ResNet50 model — specifically testing whether predictive uncertainty is elevated on the ~83 average Glioma misclassifications identified above, since two targeted retraining strategies (Attempts 1 and 2) and augmentation-based fixes failed to meaningfully close this gap
- Grad-CAM explainability
- Deployment (Gradio + Hugging Face Spaces)
- 
## Day 4 — Monte Carlo Dropout (Section 8)

### What I did

**Architecture issue discovered:** Standard torchvision ResNet50 has no Dropout layers — MC Dropout requires them to generate stochastic predictions. Added a `Dropout(0.3)` layer before the final classifier (`fc`), transferred the existing backbone weights from the Seed-7 checkpoint (chosen earlier as the seed closest to the 5-seed mean), and briefly fine-tuned (3 epochs, very low LR) to let the model adjust to the new layer. Val accuracy after this step: 97.95%.

**Implementation bug (caught and fixed):** Initial MC Dropout implementation used `model.train()` to activate Dropout, which — as an unintended side effect — also switched BatchNorm out of its stable "running statistics" mode and into per-batch statistics mode. Since inference was done one image at a time (batch size 1), BatchNorm statistics became unstable, causing MC Dropout mean-prediction accuracy to collapse to 70.3% (vs. ~94% expected). Fixed by writing an `enable_dropout_only()` helper that calls `model.eval()` (keeping BatchNorm stable) and then manually switches only `nn.Dropout` modules to train mode. After the fix, MC Dropout mean-prediction accuracy matched expected single-pass performance (94.44%), confirming correctness.

**MC Dropout run:** 30 stochastic forward passes per test image (1,600 images), using predictive standard deviation as the uncertainty score.

### Results

**Uncertainty separates correct from incorrect predictions clearly:**

| | Mean Uncertainty |
|---|---|
| Correct predictions | 0.0073 |
| Incorrect predictions | 0.0907 (**12.4x higher**) |
| Glioma correct | 0.0102 |
| Glioma incorrect | 0.0815 (**8.0x higher**) |

**Threshold-based triage analysis:** Flagging predictions with uncertainty > 0.01 for manual review:
- Only 13.9% of all test images would need manual review
- 82.0% of all model errors fall within this flagged set
- Overall "silent" (undetected) errors reduced from 97 → 16

**Glioma-specific:** Of 74 total Glioma errors, 60 (81.1%) were caught by the same threshold. Of the 14 that were missed (confidently wrong), 7 were the most dangerous type — Glioma misclassified as No-Tumor with low uncertainty. This is an important, honestly-reported limitation: uncertainty estimation substantially reduces but does not eliminate the risk of silent, confidently-wrong predictions on the most clinically dangerous error type.

### Key learnings
- MC Dropout requires the model to actually contain Dropout layers — this isn't automatic, and needs to be verified per-architecture rather than assumed (ResNet50 has none by default; the baseline CNN and EfficientNetB0 did)
- `model.train()` affects **all** layers with train/eval-dependent behavior, not just Dropout — BatchNorm's per-batch statistics can silently corrupt predictions when doing single-image inference. The fix is to selectively enable only the desired layer type
- Uncertainty-based triage was far more effective at reducing dangerous silent errors (~80% reduction) than any of the four retraining-based interventions attempted in Section 6/7 (0-2 point recall improvements) — reframing the problem from "make the model always right" to "make the model know when it might be wrong" was a more productive direction
- Even a strong uncertainty signal isn't a complete solution — 7 of 74 Glioma errors remained both confidently wrong *and* the most dangerous error type, underscoring that this should be treated as a complementary safety layer, not a replacement for careful validation

### Next up
- Grad-CAM explainability — specifically checking where the model was "looking" on the 7 confidently-wrong, high-risk Glioma→Notumor cases, to see if they show signs of shortcut learning (e.g., attending to scan artifacts) or genuinely ambiguous/small tumor regions

## Day 4 (continued) — Grad-CAM Explainability (Section 9)

### What I did

Applied Grad-CAM to the final ResNet50 model (targeting `layer4`, the last convolutional block) to visualize what the model attends to when making predictions — specifically investigating the 7 confidently-wrong, high-risk Glioma→Notumor cases identified via MC Dropout in Section 8.

**Comparison: correct vs. dangerous-wrong Glioma predictions**

- **Correct Glioma predictions:** Grad-CAM heatmaps consistently showed a single, large, high-intensity activation blob — closely resembling an actual tumor mass in shape and concentration
- **Confidently-wrong cases (Glioma→Notumor):** Heatmaps showed diffuse, scattered attention across multiple weaker regions, with no comparable concentrated "mass" signal. In 2 of 7 cases, attention concentrated heavily on the ventricles (a normal anatomical structure present in all brains) rather than any tumor-specific region

**Reassuring finding:** No evidence of the shortcut-learning risk flagged during EDA (Day 1) — attention in all 7 dangerous cases fell on genuine brain tissue, never on scan borders, artifacts, or text markers.

### Interpretation

The qualitative difference between correct and incorrect cases suggests the model has learned a genuine "large, coherent mass" detector for Glioma, and its failures cluster around cases where the tumor is likely small, subtle, or diffuse — rather than the model learning spurious, irrelevant patterns. This is a data-driven hypothesis for *why* these specific cases are hard, generated directly from the explainability analysis rather than assumed.

### Future work identified (not implemented — outside current scope)

Based on this finding, two complementary preprocessing changes were identified as the most promising next steps if pursued further:
1. **ROI-based cropping** — removing irrelevant skull/background regions before classification (shown to help in related literature on this dataset)
2. **Higher input resolution** (e.g., 384×384 instead of 224×224)

These were evaluated as synergistic rather than independent: cropping removes wasted background pixels, allowing the subsequent resolution increase to effectively "zoom in" on brain tissue rather than upscaling irrelevant background — giving small/subtle tumors more effective pixel-detail than either change alone. Not pursued in this project due to scope (consistent with the Day 1 decision to exclude segmentation-level work) and to prioritize completing the uncertainty estimation and deployment stages within the available time.

### Key learnings
- Grad-CAM is most useful when compared *between* correct and incorrect predictions of the same class, not viewed in isolation — the qualitative contrast (concentrated blob vs. diffuse attention) was far more informative than either set of images alone
- Explainability findings can generate concrete, testable hypotheses for future improvement (small/subtle tumor detection) even when not pursued immediately — this is more valuable than a vague "more data would help" conclusion

### Next up
- Deployment: Gradio interface combining prediction + confidence + MC Dropout uncertainty + Grad-CAM heatmap, deployed to Hugging Face Spaces
