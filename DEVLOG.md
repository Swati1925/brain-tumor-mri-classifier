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
