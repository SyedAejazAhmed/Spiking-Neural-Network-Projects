# Retinal Classification Project

This folder contains multiple deep learning experiments for diabetic retinopathy classification. Five different model architectures were trained and evaluated on the APTOS 2019 Blindness Detection dataset to predict retinal disease severity (No DR, Mild, Moderate, Severe, Proliferative).

## Quick Navigation

- [Basic CNN](#basic-cnn-convolutional-neural-network) — Highest raw accuracy
- [RetSpike-Net](#retspike-net) — Best balanced performance (recommended for clinical use)
- [CSNN](#csnn-convolutional-spiking-neural-network) — Biologically-inspired; needs tuning
- [Proto k-shot SCNN](#proto-k-shot-scnn-prototype-based-readout) — Few-shot / prototype learning
- [Spike Classifier](#spike-classifier) — Spike-based variant; missing metrics

---

## Model Comparison Summary

| Model | Test Accuracy | Macro F1 | Weighted F1 | Best For | Status |
|---|---:|---:|---:|---|---|
| **Basic CNN** | 0.7270 | 0.3832 | 0.6757 | Raw accuracy; fast iteration | ✅ Complete |
| **RetSpike-Net** | 0.7080 | **0.4985** | 0.7019 | Balanced multi-class; clinical use | ✅ Complete |
| **Proto k-shot SCNN** | 0.6049 | 0.3111 | 0.5825 | Few-shot experimentation | ✅ Complete |
| **CSNN** | 0.3733 | 0.2125 | 0.3445 | Research on spiking dynamics | ⚠️ Needs tuning |
| **Spike Classifier** | — | — | — | (Metrics pending) | ⏳ Incomplete |

---

## Detailed Model Descriptions

### Basic CNN (Convolutional Neural Network)

**Overview**  
A straightforward convolutional neural network trained with standard supervised learning. Serves as a strong baseline for the retinal classification task.

**Architecture**
- Convolutional backbone: 3–5 Conv blocks with BatchNorm and ReLU
- Global pooling + fully-connected readout (2 layers)
- Output: 5-class softmax

**Key Results**
- **Test Accuracy**: 72.70%
- **Test Macro Avg F1**: 0.3832 (indicating imbalance across classes)
- **Test Weighted Avg F1**: 0.6757
- Strong on class 0 (No DR): 96.19% recall, 92.67% precision
- Weak on minority classes (3, 4): ~0% recall

**Advantages**
- Highest overall test accuracy in the workspace
- Simple, interpretable architecture
- Fast training and inference

**Limitations**
- Poor sensitivity on rare disease classes (Severe, Proliferative)
- Not suitable for clinical deployment if balanced sensitivity matters

**Full Details**: See [CNN/README.md](CNN/README.md)

---

### RetSpike-Net

**Overview**  
An advanced architecture combining spiking neural layers with temporal dynamics. Designed for robust, balanced classification across all disease severity classes.

**Architecture**
- Encoding layer: converts images to spike patterns (rate or temporal encoding)
- Spiking backbone: stacked LIF (or similar) layers with temporal dynamics
- Readout: dense classifier over aggregated spike features
- Output: 5-class logits

**Key Results**
- **Test Accuracy**: 70.80%
- **Test Macro Avg F1**: **0.4985** (best balanced performance)
- **Test Weighted Avg F1**: 0.7019
- Train accuracy: 82.98% (strong training signal)
- More even per-class performance (vs. CNN)

**Advantages**
- **Best macro F1 score** → more balanced sensitivity across classes
- Excellent on No DR (93.91% recall) and Moderate (60% recall)
- Biologically-inspired spiking dynamics; can be deployed to neuromorphic hardware
- Stronger overall validation performance (64.96% val acc)

**Limitations**
- Slightly lower raw test accuracy than Basic CNN (70.8% vs 72.7%)
- More complex training (surrogate gradients)

**Clinical Recommendation**  
**Preferred for real-world deployment** because it detects minority disease classes more reliably, which is critical in medical screening.

**Full Details**: See [RetSpike-Net/README.md](RetSpike-Net/README.md)

---

### CSNN (Convolutional Spiking Neural Network)

**Overview**  
A convolutional spiking neural network that simulates event-driven neurons with temporal spike trains. Designed for neuroscience research and neuromorphic hardware deployment.

**Architecture**
- Convolutional feature extractor
- Spiking layers: event-driven LIF or similar units
- Readout: dense layer over spike aggregates
- Output: 5-class logits

**Key Results**
- **Test Accuracy**: 37.33%
- **Test Macro Avg F1**: 0.2125
- **Test Weighted Avg F1**: 0.3445
- Class 2 (Moderate): **0% recall** in test set
- Highly imbalanced predictions across classes

**Advantages**
- Biologically plausible spike-based computation
- Suitable for neuromorphic hardware (e.g., Loihi, Intel chips)
- Lower energy footprint if deployed on specialized hardware

**Limitations**
- **Significantly underperforming** on this dataset
- Many classes show zero or near-zero recall
- Needs additional tuning: better loss weighting, data augmentation, or readout redesign

**Recommendation**  
Use CSNN for **research into spiking dynamics** or **hardware experiments**, not for classification on this dataset without further optimization.

**Full Details**: See [CSNN/README.md](CSNN/README.md)

---

### Proto k-shot SCNN (Prototype-based Spiking Readout)

**Overview**  
An experiment in few-shot learning using prototype-based classifiers and spiking features. Two variants: `proto_full` (full dataset) and `proto_5shot` (simulated 5-shot).

**Architecture**
- Convolutional feature extractor (shared backbone)
- Prototype layer: per-class prototypes computed in feature space
- Distance-based or SVC readout (multiple variants in `outputs/`)
- Output: 5-class predictions

**Key Results (proto_full)**
- **Test Accuracy**: 60.49%
- **Test Macro Avg F1**: 0.3111
- **Test Weighted Avg F1**: 0.5825
- Strong on class 0: 80% recall, 92.31% precision
- Weak/zero on classes 1 and 4

**Key Results (proto_5shot example)**
- **Test Accuracy**: 36.51%
- Highly variable across different episodes

**Advantages**
- Useful for few-shot scenarios where labeled data is scarce
- Interpretable prototype-based decision boundaries
- Can be adapted to online learning

**Limitations**
- Lower overall accuracy than CNN or RetSpike-Net
- Performance highly sensitive to prototype selection
- Few-shot runs show high variance

**Recommendation**  
Use for **research into few-shot learning** or when you need to deploy with minimal labeled examples. For standard multi-class classification with full data, prefer CNN or RetSpike-Net.

**Full Details**: See [proto k-shot SCNN/README.md](proto%20k-shot%20SCNN/README.md)

---

### Spike Classifier

**Overview**  
A spike-based classifier variant. Currently has model weights and test predictions but is missing standard evaluation reports (classification reports and confusion matrices).

**Status**
- ✅ Model checkpoint: `outputs/snn_best.pth`
- ✅ Test predictions: `outputs/test_predictions.csv`
- ❌ Missing: `test_classification_report.csv`, `test_confusion_matrix.csv`, per-split metrics

**How to Complete**
To generate full metrics:
1. Obtain ground-truth labels for the test set
2. Compare predicted labels in `test_predictions.csv` against ground truth
3. Compute and save classification reports and confusion matrices
4. Update this README with full tables and images

**Full Details**: See [Spike Classifier/README.md](Spike%20Classifier/README.md)

---

## Recommendation Summary

### Use **RetSpike-Net** if:
- ✅ You need balanced sensitivity across all disease classes
- ✅ Clinical deployment or medical screening scenarios
- ✅ Missing rare-class detection is costly
- ✅ You want spiking dynamics for neuromorphic hardware

### Use **Basic CNN** if:
- ✅ You prioritize raw overall accuracy
- ✅ You need fast training and inference
- ✅ You are in a research/prototyping phase
- ✅ Your application focuses on detecting the dominant class (No DR)

### Use **CSNN** if:
- ✅ You are researching spiking neural networks
- ✅ You plan to deploy on neuromorphic hardware (and can further optimize the model)
- ✅ You want to explore event-driven computation

### Use **Proto k-shot SCNN** if:
- ✅ You have very limited labeled data
- ✅ You are experimenting with prototype-based meta-learning
- ✅ You need interpretable decision boundaries

---

## Dataset & Preprocessing

- **Source**: APTOS 2019 Blindness Detection (Kaggle)
- **Classes**: 5 (No DR, Mild, Moderate, Severe, Proliferative)
- **Train set**: ~2343 images
- **Val set**: ~733 images
- **Test set**: ~586–733 images (varies by split)
- **Images**: Color fundus photographs, resized and normalized (see individual notebooks)

---

## File Structure

```
Retinal Classification/
├── README.md (this file)
├── CNN/
│   ├── model.ipynb
│   ├── README.md (detailed metrics, tables, architecture)
│   └── outputs/basic_cnn_outputs/
│       ├── train/val/test_classification_report.csv
│       ├── train/val/test_confusion_matrix.csv
│       └── ...
├── RetSpike-Net/
│   ├── model.ipynb
│   ├── README.md (detailed metrics, tables, architecture)
│   └── outputs/
│       ├── train/val/test_classification_report.csv
│       ├── train/val/test_confusion_matrix.png/csv
│       └── ...
├── CSNN/
│   ├── model.ipynb
│   ├── README.md
│   └── outputs/
│       ├── train/val/test_classification_report.csv
│       ├── train/val/test_confusion_matrix.png/csv
│       └── ...
├── proto k-shot SCNN/
│   ├── model.ipynb
│   ├── README.md
│   ├── snn.py
│   ├── utils.py
│   └── outputs/
│       ├── proto_full_* and proto_5shot_* reports
│       └── ...
├── Spike Classifier/
│   ├── model.ipynb
│   ├── README.md
│   └── outputs/
│       ├── snn_best.pth
│       └── test_predictions.csv
└── dataset/
    └── aptos2019-blindness-detection/
        ├── train_images/
        ├── test_images/
        └── train.csv, test.csv, ...
```

---

## How to Explore Each Model

1. **Read the model's individual README** (linked above) for full architecture, tables, and confusion-matrix links.
2. **Open the notebook** (e.g., `CNN/model.ipynb`) to see training code, hyperparameters, and visualizations.
3. **Check the `outputs/` folder** for CSVs and PNGs.

---

## Quick Metrics Comparison (Detailed)

### Per-Class Performance on Test Set

**Basic CNN**
- Class 0 (No DR): 96.19% recall ✅
- Class 1 (Mild): 27.12% recall ⚠️
- Class 2 (Moderate): 82.50% recall ✅
- Class 3 (Severe): 0% recall ❌
- Class 4 (Prolif.): 0% recall ❌

**RetSpike-Net**
- Class 0 (No DR): 93.91% recall ✅
- Class 1 (Mild): 51.35% recall ✅
- Class 2 (Moderate): 60.00% recall ✅
- Class 3 (Severe): 38.46% recall ⚠️
- Class 4 (Prolif.): 11.86% recall ⚠️

**CSNN**
- Class 0: 65.00% recall
- Class 1: 35.14% recall
- Class 2: 0% recall ❌
- Class 3: 20.00% recall
- Class 4: 10.00% recall

**Proto k-shot (proto_full)**
- Class 0: 80.00% recall
- Class 1: 0% recall ❌
- Class 2: 74.00% recall
- Class 3: 20.00% recall
- Class 4: 0% recall ❌

---

## Next Steps

1. **For clinical deployment**: Use **RetSpike-Net** (best macro F1, balanced sensitivity).
2. **For research continuation**: Consider further tuning CSNN with class weighting and stronger data augmentation.
3. **For Spike Classifier**: Generate missing metrics and integrate results into this comparison.
4. **For publication**: Use the detailed per-model READMEs as supplementary material (architecture, full tables, confusion matrices).

---

## References & Resources

- Dataset: [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
- See `../References/Links.txt` for SNN literature and related papers

---

*Last updated: May 2026*  
*For questions or updates, see the individual model READMEs or the main workspace README.md*
