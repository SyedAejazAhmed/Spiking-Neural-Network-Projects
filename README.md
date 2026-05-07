# Spiking Neural Network (SNN) Projects Workspace

This workspace contains local experiments and resources for exploring Spiking Neural Networks (SNNs) with PyTorch.

## Structure

- **Spiking-Neural-Network-SNN-with-PyTorch.../**: Contains the main project implementation and notebooks.
  - `Spiking Neural Networks with PyTorch.ipynb`: The primary notebook for training and visualization.
  - `README.md`: Project-specific documentation.
- **References/**: Contains links to useful resources and papers.
- **requirements.txt**: Python dependencies for the workspace.

## Getting Started

### Prerequisites

Ensure you have Python installed (preferably 3.10+).

### Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. Navigate to the project directory:
   ```bash
   cd Spiking-Neural-Network-SNN-with-PyTorch-where-Backpropagation-engenders-STDP
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open `Spiking Neural Networks with PyTorch.ipynb` and run the cells.

## References

See `References/Links.txt` for a curated list of SNN resources, including:
- [Awesome Spiking Neural Networks](https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks)
- [Awesome SNN Conference Papers](https://github.com/AXYZdong/awesome-snn-conference-paper)

## Notes

- The notebook has been updated to handle device placement (CPU/GPU) correctly for visualization.

## Model comparison (Retinal Classification experiments)

Summary of final metrics (selected key numbers from `Retinal Classification/*/outputs`):
- `Basic CNN` (Retinal Classification/CNN): Test accuracy 0.72696; test macro avg (f1) 0.38316 — highest overall accuracy but weak minority-class recall.
- `RetSpike-Net` (Retinal Classification/RetSpike-Net): Test accuracy 0.70805; test macro avg (f1) 0.49846 — best balanced per-class performance (higher macro f1).
- `Proto k-shot SCNN` (proto_full): Test accuracy 0.60490; proto_5shot runs show lower/variable performance (example proto_5shot test acc 0.36512).
- `CSNN` (Retinal Classification/CSNN): Test accuracy 0.37330 — currently underperforming; needs additional tuning or readout changes.
- `Spike Classifier`: no per-split reports saved; only `outputs/test_predictions.csv` and model weights exist.

Recommendation:
- If you care about raw overall accuracy as a single metric, the `Basic CNN` shows the highest test accuracy.
- For clinical or real-world deployment where balanced sensitivity across classes matters (detecting minority classes is critical), `RetSpike-Net` is the better choice due to substantially higher macro-averaged F1 and more even per-class performance.

Next steps I can take for you:
- Add the exact confusion-matrix images into each README (embed PNGs) — would you like that?
- Compute additional metrics for `Spike Classifier` from `test_predictions.csv` if you can provide predicted labels.

