
# RetSpike-Net

Overview
--------
`RetSpike-Net` is an architecture that leverages spiking layers and temporal dynamics to improve class-balanced performance on the retinal dataset. The repository contains training history, checkpoints and CSV reports under `outputs/`.

Architecture
------------
- Input preprocessing: images normalized and resized as in the notebook.
- Encoding layer(s): converts image features to spikes (rate or temporal encoding depending on the run).
- Spiking backbone: stacked spiking layers (e.g., LIF or similar) that process temporal spike trains and produce robust feature representations.
- Readout: a dense classifier over aggregated spike features producing logits for five classes.

Implementation notes
--------------------
- Training often uses surrogate gradients for spiking layers and hybrid loss schedules.
- Checkpoints named `retspikenet_best.pth` are present in `outputs/`.

Artifacts
---------
- [train_classification_report.csv](Retinal%20Classification/RetSpike-Net/outputs/train_classification_report.csv)
- [val_classification_report.csv](Retinal%20Classification/RetSpike-Net/outputs/val_classification_report.csv)
- [test_classification_report.csv](Retinal%20Classification/RetSpike-Net/outputs/test_classification_report.csv)
- [test_confusion_matrix.csv](Retinal%20Classification/RetSpike-Net/outputs/test_confusion_matrix.csv)
- [train_confusion_matrix.csv](Retinal%20Classification/RetSpike-Net/outputs/train_confusion_matrix.csv)
- [val_confusion_matrix.csv](Retinal%20Classification/RetSpike-Net/outputs/val_confusion_matrix.csv)
- Visualizations (PNG): [test_confusion_matrix.png](Retinal%20Classification/RetSpike-Net/outputs/test_confusion_matrix.png), [train_confusion_matrix.png](Retinal%20Classification/RetSpike-Net/outputs/train_confusion_matrix.png), [val_confusion_matrix.png](Retinal%20Classification/RetSpike-Net/outputs/val_confusion_matrix.png)

Results (detailed tables)
-------------------------
Train split
| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| No DR | 0.9097 | 0.9898 | 0.9481 | 1171 |
| Mild | 0.8121 | 0.8631 | 0.8369 | 1147 |
| Moderate | 0.8275 | 0.6755 | 0.7439 | 1165 |
| Severe | 0.7691 | 0.9080 | 0.8328 | 1163 |
| Prolif. | 0.8352 | 0.7088 | 0.7669 | 1130 |
| **accuracy** | - | - | **0.8298** | 5776 |

Validation split
| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| No DR | 0.8814 | 0.9524 | 0.9155 | 273 |
| Mild | 0.6579 | 0.6734 | 0.6656 | 297 |
| Moderate | 0.5221 | 0.4229 | 0.4673 | 279 |
| Severe | 0.5635 | 0.7580 | 0.6464 | 281 |
| Prolif. | 0.6100 | 0.4682 | 0.5297 | 314 |
| **accuracy** | - | - | **0.6496** | 1444 |

Test split
| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| No DR | 0.9162 | 0.9391 | 0.9275 | 361 |
| Mild | 0.4471 | 0.5135 | 0.4780 | 74 |
| Moderate | 0.6283 | 0.6000 | 0.6138 | 200 |
| Severe | 0.2727 | 0.3846 | 0.3191 | 39 |
| Prolif. | 0.2188 | 0.1186 | 0.1538 | 59 |
| **accuracy** | - | - | **0.7080** | 733 |

Confusion matrices
-------------------
- [test_confusion_matrix.csv](Retinal%20Classification/RetSpike-Net/outputs/test_confusion_matrix.csv)
- [val_confusion_matrix.csv](Retinal%20Classification/RetSpike-Net/outputs/val_confusion_matrix.csv)
- [train_confusion_matrix.csv](Retinal%20Classification/RetSpike-Net/outputs/train_confusion_matrix.csv)

Interpretation and recommendation
----------------------------------
- `RetSpike-Net` achieves strong overall training performance and, importantly, the best balanced per-class performance (macro F1 ≈ 0.498) among available runs. While its raw test accuracy (≈70.8%) is slightly below the Basic CNN, it provides more reliable detection of several minority classes.
- For use-cases that require balanced sensitivity across classes (for example clinical screening), `RetSpike-Net` is the recommended model to prefer over the Basic CNN.

How to reproduce
----------------
Run `Retinal Classification/RetSpike-Net/model.ipynb` and check `outputs/` for the listed artifacts.

Advantages:
- Strong training performance and the best balanced per-class performance (higher macro avg f1) among models here.
- Better recall and f1 across minority classes compared to the basic CNN and prototype models.

Limitations:
- Overall test accuracy is slightly lower than the basic CNN (0.708 vs 0.727), but RetSpike-Net gives more balanced predictions across classes.

Recommendation:
- For clinical/real-world settings where balanced detection across classes matters (sensitivity to minority classes), RetSpike-Net is recommended.

