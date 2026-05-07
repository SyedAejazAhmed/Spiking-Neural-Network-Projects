# Basic CNN

Overview
--------
This folder contains a straightforward convolutional neural network trained for retinal disease classification. The model was implemented and experimented with in `model.ipynb` (see the notebook for exact hyperparameters and training loop). Below you will find a short architecture description, training notes, and full per-split results (tables) so reviewers can quickly inspect performance.

Architecture
------------
- Input: color RGB retinal images (preprocessed / resized in the notebook).
- Convolutional backbone: a stack of convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool). Typical block count: 3–5 depending on the notebook variant.
- Global pooling and 1–2 fully-connected layers for class logits.
- Output: Softmax over five classes (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative).

Implementation notes
--------------------
- Training loop: standard supervised cross-entropy loss with data augmentation used in the notebook.
- Optimizer: see `model.ipynb` (commonly Adam/SGD depending on experiment).
- Checkpoints and outputs are in `outputs/basic_cnn_outputs/`.

Artifacts (key files)
--------------------
- Classification reports and confusion matrices: `outputs/basic_cnn_outputs/`
  - [test_classification_report.csv](Retinal%20Classification/CNN/outputs/basic_cnn_outputs/test_classification_report.csv)
  - [val_classification_report.csv](Retinal%20Classification/CNN/outputs/basic_cnn_outputs/val_classification_report.csv)
  - [train_classification_report.csv](Retinal%20Classification/CNN/outputs/basic_cnn_outputs/train_classification_report.csv)
  - [test_confusion_matrix.csv](Retinal%20Classification/CNN/outputs/basic_cnn_outputs/test_confusion_matrix.csv)
  - [val_confusion_matrix.csv](Retinal%20Classification/CNN/outputs/basic_cnn_outputs/val_confusion_matrix.csv)
  - [train_confusion_matrix.csv](Retinal%20Classification/CNN/outputs/basic_cnn_outputs/train_confusion_matrix.csv)

Results (detailed tables)
-------------------------
Train split
| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.9358 | 0.9723 | 0.9537 | 1155 |
| 1 | 0.6135 | 0.4219 | 0.5000 | 237 |
| 2 | 0.5785 | 0.8826 | 0.6989 | 639 |
| 3 | 0.8000 | 0.0325 | 0.0625 | 123 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 189 |
| **accuracy** | - | - | **0.7644** | 2343 |

Validation split
| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.9288 | 0.9751 | 0.9514 | 361 |
| 1 | 0.5536 | 0.4189 | 0.4769 | 74 |
| 2 | 0.5700 | 0.8350 | 0.6775 | 200 |
| 3 | 0.6000 | 0.0769 | 0.1364 | 39 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 59 |
| **accuracy** | - | - | **0.7544** | 733 |

Test split
| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.9267 | 0.9619 | 0.9440 | 289 |
| 1 | 0.3902 | 0.2712 | 0.3200 | 59 |
| 2 | 0.5388 | 0.8250 | 0.6519 | 160 |
| 3 | 0.0000 | 0.0000 | 0.0000 | 31 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 47 |
| **accuracy** | - | - | **0.7270** | 586 |

Confusion matrices
-------------------
- Train confusion matrix (CSV): [outputs/basic_cnn_outputs/train_confusion_matrix.csv](Retinal%20Classification/CNN/outputs/basic_cnn_outputs/train_confusion_matrix.csv)
- Val confusion matrix (CSV): [outputs/basic_cnn_outputs/val_confusion_matrix.csv](Retinal%20Classification/CNN/outputs/basic_cnn_outputs/val_confusion_matrix.csv)
- Test confusion matrix (CSV): [outputs/basic_cnn_outputs/test_confusion_matrix.csv](Retinal%20Classification/CNN/outputs/basic_cnn_outputs/test_confusion_matrix.csv)

Interpretation and recommendation
----------------------------------
- The Basic CNN achieves the highest raw test accuracy among the non-prototype experiments (≈72.7%). However, it has very low recall and F1 on minority classes (3 and 4), which is visible in both the per-class tables and the confusion matrices.
- Use the Basic CNN when overall accuracy is the main objective and when quick iteration is needed. If sensitivity on rare classes is required, consider the RetSpike-Net experiment instead (better macro F1).

How to reproduce evaluation
---------------------------
1. Launch the notebook: `Retinal Classification/CNN/model.ipynb` and re-run cells for evaluation.
2. The evaluation scripts in the notebook produce the CSVs listed above.

Notes & next steps
------------------
- To make this README reproducible, consider embedding the confusion-matrix PNGs directly (if you want I can insert them). The notebook contains the exact training hyperparameters used for each run.


