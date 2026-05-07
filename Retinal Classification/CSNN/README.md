# CSNN (Convolutional Spiking Neural Network)

Overview
--------
This folder contains experiments with a convolutional spiking neural network. CSNNs simulate event-driven spiking dynamics and are a bridge between biologically inspired models and practical classification readouts. Results and artifacts are stored in `outputs/`.

Architecture
------------
- Input: preprocessed retinal images.
- Convolutional front-end: several Conv2d layers producing feature maps.
- Spiking layer(s): event-driven units (LIF or similar) convert activations to spikes; temporal dynamics are handled in the time dimension inside the experiment code.
- Readout: a non-spiking dense readout layer maps spike counts or aggregated features to class logits.

Implementation notes
--------------------
- The CSNN training uses surrogate gradient methods to allow backpropagation through spike non-linearities (see `CSNN/model.ipynb`).
- Checkpoints and reports are in `outputs/` — see links below.

Artifacts
---------
- [train_classification_report.csv](Retinal%20Classification/CSNN/outputs/train_classification_report.csv)
- [val_classification_report.csv](Retinal%20Classification/CSNN/outputs/val_classification_report.csv)
- [test_classification_report.csv](Retinal%20Classification/CSNN/outputs/test_classification_report.csv)
- [train_confusion_matrix.csv](Retinal%20Classification/CSNN/outputs/train_confusion_matrix.csv)
- [val_confusion_matrix.csv](Retinal%20Classification/CSNN/outputs/val_confusion_matrix.csv)
- [test_confusion_matrix.csv](Retinal%20Classification/CSNN/outputs/test_confusion_matrix.csv)
- Visualizations: [train_confusion_matrix.png](Retinal%20Classification/CSNN/outputs/train_confusion_matrix.png), [val_confusion_matrix.png](Retinal%20Classification/CSNN/outputs/val_confusion_matrix.png), [test_confusion_matrix.png](Retinal%20Classification/CSNN/outputs/test_confusion_matrix.png)

Results (selected tables)
-------------------------
Train split
| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.3522 | 0.7073 | 0.4703 | 123 |
| 1 | 0.5800 | 0.4462 | 0.5043 | 130 |
| 2 | 0.0000 | 0.0000 | 0.0000 | 110 |
| 3 | 0.5143 | 0.7143 | 0.5980 | 126 |
| 4 | 0.4688 | 0.3093 | 0.3727 | 97 |
| **accuracy** | - | - | **0.4522** | 586 |

Validation split
| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.6049 | 0.6851 | 0.6425 | 181 |
| 1 | 0.1552 | 0.2432 | 0.1895 | 37 |
| 2 | 0.0000 | 0.0000 | 0.0000 | 100 |
| 3 | 0.1333 | 0.5263 | 0.2128 | 19 |
| 4 | 0.0714 | 0.0690 | 0.0702 | 29 |
| **accuracy** | - | - | **0.3962** | 366 |

Test split
| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.6000 | 0.6500 | 0.6240 | 180 |
| 1 | 0.2000 | 0.3514 | 0.2549 | 37 |
| 2 | 0.0000 | 0.0000 | 0.0000 | 100 |
| 3 | 0.0513 | 0.2000 | 0.0816 | 20 |
| 4 | 0.1034 | 0.1000 | 0.1017 | 30 |
| **accuracy** | - | - | **0.3733** | 367 |

Interpretation
--------------
- The CSNN experiment is primarily useful for research into spiking dynamics and efficient inference on neuromorphic hardware. Current results show the model underperforms compared to conventional CNNs and RetSpike-Net; many classes (notably class 2) have zero recall in some runs. This suggests the readout or training schedule needs attention (class balancing, stronger supervision, or different loss weighting).

Recommendations
---------------
- If the goal is classification performance on this dataset, prefer the Basic CNN or RetSpike-Net runs.
- If the goal is to develop a spiking pipeline for deployment to neuromorphic hardware, continue refining the CSNN readout and consider data augmentation and class-reweighting.

How to reproduce
----------------
Open and run `Retinal Classification/CSNN/model.ipynb`.

Contact / Notes
----------------
- If you want, I can embed the PNG confusion matrices directly into this README. I can also run additional balanced training experiments and produce updated reports.


