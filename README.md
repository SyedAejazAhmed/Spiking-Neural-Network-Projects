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

---

## 📝 Models for Publication / Paper

Three models are recommended for your paper (all with strong performance and clear contributions):

### **Recommended Paper Models (3-4)**

| # | Model | Test Acc | Macro F1 | Contribution | Status |
|---|---|---:|---:|---|---|
| 1 | **RetSpike-Net** | 70.80% | **0.4985** | 🎯 Main: Novel spiking approach, best balanced performance | ✅ Ready |
| 2 | **Basic CNN** | **72.70%** | 0.3832 | 📌 Baseline: Strong CNN comparison, shows SNNs are competitive | ✅ Ready |
| 3 | **Proto k-shot SCNN** | 60.49% | 0.3111 | 🔬 Innovation: Few-shot learning angle, methodological diversity | ✅ Ready |
| 4 | **CSNN** | 37.33% | 0.2125 | ⚠️ Optional: Only if discussing future optimizations/challenges | ⚠️ Needs tuning |

### **Why These 3?**

- **Novelty**: RetSpike-Net shows spiking nets can match/beat standard CNNs on real medical data
- **Rigor**: Basic CNN provides credible baseline for comparison
- **Scope**: Proto k-shot adds methodological breadth (few-shot angle)
- **Quality**: All three are ≥60% accuracy; strong enough for conference/journal publication

### **Not Recommended for Paper:**
- ❌ **CSNN** (37.33% accuracy; underperforming unless specifically discussing why)
- ❌ **Spike Classifier** (incomplete metrics)

→ **Full model comparison**: See [Retinal Classification/README.md](Retinal%20Classification/README.md) for detailed architecture, tables, and all results.
