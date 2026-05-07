# Prototype / k-shot SCNN (Prototype readout + SCNN backbone)

Overview
--------
This experiment explores prototype-based readouts and few-shot evaluation on the retinal dataset. Two primary variants are present in `outputs/`: a `proto_full` run that uses the full training set with prototype-style readout, and `proto_5shot` experiments that simulate k-shot learning.

Architecture
------------
- Frontend: convolutional feature extractor (shared with other SCNN/CNN experiments).
- Prototype readout: per-class prototypes computed in feature space; classification is based on distance to prototypes or a simple linear readout trained on prototype-augmented features.
- Readout variants: support-vector / balanced SVC and a dense readout are both used in different runs — see notebooks for exact readout selection.

Artifacts
---------
- `outputs/` contains:
	- proto_full_train_classification_report.csv
	- proto_full_val_classification_report.csv
	- proto_full_test_classification_report.csv
	- proto_full_*_confusion_matrix.csv / .png
	- proto_5shot_* reports and confusion matrices

Results (proto_full)
--------------------
Train split (proto_full)
| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.8423 | 0.7849 | 0.8126 | 279 |
| 1 | 0.7734 | 0.3548 | 0.4865 | 279 |
| 2 | 0.3550 | 0.6272 | 0.4534 | 279 |
| 3 | 0.4400 | 0.6308 | 0.5184 | 279 |
| 4 | 0.8070 | 0.3297 | 0.4682 | 279 |
| **accuracy** | - | - | **0.5455** | 1395 |

Validation (proto_full)
| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.9416 | 0.8011 | 0.8657 | 181 |
| 1 | 0.2500 | 0.0270 | 0.0488 | 37 |
| 2 | 0.5127 | 0.8100 | 0.6279 | 100 |
| 3 | 0.2083 | 0.5263 | 0.2985 | 19 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 29 |
| **accuracy** | - | - | **0.6475** | 366 |

Test (proto_full)
| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.9231 | 0.8000 | 0.8571 | 180 |
| 1 | 0.0000 | 0.0000 | 0.0000 | 37 |
| 2 | 0.4625 | 0.7400 | 0.5692 | 100 |
| 3 | 0.0952 | 0.2000 | 0.1290 | 20 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 30 |
| **accuracy** | - | - | **0.6049** | 367 |

proto_5shot (example)
| Metric | Value |
|---|---:|
| Test accuracy (proto_5shot) | 0.3651 |

Confusion matrices and reports
-----------------------------
- Example files (proto_full):
	- [proto_full_train_confusion_matrix.csv](Retinal%20Classification/proto%20k-shot%20SCNN/outputs/proto_full_train_confusion_matrix.csv)
	- [proto_full_val_confusion_matrix.csv](Retinal%20Classification/proto%20k-shot%20SCNN/outputs/proto_full_val_confusion_matrix.csv)
	- [proto_full_test_confusion_matrix.csv](Retinal%20Classification/proto%20k-shot%20SCNN/outputs/proto_full_test_confusion_matrix.csv)

Interpretation
--------------
- Prototype-based readouts can perform well on some classes (class 0 and 2 here) but are sensitive to prototype quality and class balance.
- The `proto_5shot` runs show wide variability; few-shot settings require careful episode construction and robust prototype formation.

Recommendations
---------------
- Use `proto_full` when you want a prototype-style analysis on the full training set.
- For reliable few-shot performance consider: stronger augmentation, more representative support sets per episode, or hybrid readouts that combine prototype distances with learned linear classifiers.

How to reproduce
----------------
Run the prototype notebooks in this folder and inspect `outputs/` for the CSVs and PNGs linked above.


