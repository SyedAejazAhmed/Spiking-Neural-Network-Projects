
# Spike Classifier

Overview
--------
This folder contains a spike-based classifier experiment. Current outputs include a saved model weight and a `test_predictions.csv` file listing image filenames and (predicted) labels. There are no per-split `classification_report.csv` files in the `outputs/` folder right now, so the README includes instructions to compute and add them.

Artifacts present
-----------------
- `outputs/snn_best.pth` (model checkpoint)
- [outputs/test_predictions.csv](Retinal%20Classification/Spike%20Classifier/outputs/test_predictions.csv) (image, label)

Status
------
- No explicit `test_classification_report.csv`, `test_confusion_matrix.csv`, `val_*` or `train_*` reports found in `outputs/` for this run.

How to generate full metrics (quick guide)
----------------------------------------
If you have ground-truth labels for the test set (CSV mapping image -> true label), run a small Python snippet to produce the standard reports. Example (run inside the workspace):

```python
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# load predicted file and ground truth file
pred = pd.read_csv('Retinal Classification/Spike Classifier/outputs/test_predictions.csv')
gt = pd.read_csv('Retinal Classification/Spike Classifier/test_ground_truth.csv')  # create or point to your GT file

# merge on image filename column
df = pred.merge(gt, on='image')
y_pred = df['label']
y_true = df['true_label']

# compute and save
report = classification_report(y_true, y_pred, output_dict=True)
cm = confusion_matrix(y_true, y_pred)
pd.DataFrame(report).T.to_csv('Retinal Classification/Spike Classifier/outputs/test_classification_report.csv')
pd.DataFrame(cm).to_csv('Retinal Classification/Spike Classifier/outputs/test_confusion_matrix.csv')
```

Recommended next steps to complete README
-----------------------------------------
1. Provide (or point me to) the ground-truth CSV for the test set so I can compute metrics and update this README.
2. Alternatively, if the `test_predictions.csv` already contains both `true_label` and `pred_label` columns, I can compute reports directly and embed tables and confusion-matrix images.

When metrics are available I will add full train/val/test tables, confusion-matrix PNGs, and an architecture section similar to the other model READMEs.


