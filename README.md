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
