# Stock Price Prediction: SNN vs LSTM

This project compares **Spiking Neural Networks (SNN)** implemented with SpikingJelly against traditional **LSTM** networks for stock price direction prediction on RELIANCE.NS (Reliance Industries NSE stock).

## 📊 Project Overview

- **Dataset**: RELIANCE.NS stock data (Mar 2025 - Mar 2026, ~252 trading days)
- **Task**: Binary classification (predict next day up/down direction)
- **Features**: OHLCV + 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Models**: 
  - SNN with LIF (Leaky Integrate-and-Fire) neurons using rate coding
  - LSTM with 2 layers and dropout
- **Evaluation**: Accuracy, F1, Precision, Recall, Sharpe Ratio, Win Rate

## 🗂️ Project Structure

```
Stock Price/
├── data_preparation.py          # Data download and preprocessing module
├── SNN_Stock_Prediction.ipynb   # SNN model training and evaluation
├── LSTM_Stock_Prediction.ipynb  # LSTM model training and evaluation
├── requirements.txt             # Python dependencies
├── processed_data/              # Cached processed data (auto-generated)
└── outputs/                     # Results and visualizations (auto-generated)
    ├── snn/                     # SNN results
    │   ├── best_model.pth
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   ├── spike_raster.png
    │   ├── firing_rates.png
    │   ├── backtest_results.png
    │   └── results_summary.txt
    ├── lstm/                    # LSTM results
    │   ├── best_model.pth
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   ├── hidden_states.png
    │   ├── backtest_results.png
    │   └── results_summary.txt
    └── model_comparison.csv     # Side-by-side comparison
```

## 🚀 Quick Start

### 1. Install Dependencies

Using conda (recommended for TA-Lib):
```bash
conda create -n snn_stock python=3.8
conda activate snn_stock
conda install -c conda-forge ta-lib
pip install -r requirements.txt
```

Or using pip only:
```bash
pip install -r requirements.txt
# Note: TA-Lib may require additional system dependencies
# On Ubuntu/Debian: sudo apt-get install libta-lib-dev
# On macOS: brew install ta-lib
```

### 2. Run the Notebooks

**Option A: Run SNN first**
```bash
jupyter notebook SNN_Stock_Prediction.ipynb
```
- Downloads RELIANCE.NS data automatically
- Trains SNN model with SpikingJelly
- Generates spike visualizations
- Performs backtesting

**Option B: Run LSTM for comparison**
```bash
jupyter notebook LSTM_Stock_Prediction.ipynb
```
- Uses the same data as SNN
- Trains LSTM baseline
- Compares results with SNN (if available)
- Generates comparison plots

### 3. View Results

After running both notebooks:
- Check `outputs/snn/` for SNN results
- Check `outputs/lstm/` for LSTM results
- View `outputs/model_comparison.csv` for side-by-side metrics
- View `outputs/model_comparison.png` for visual comparison

## 📈 Expected Results

Based on the architecture and data:

| Metric | SNN | LSTM |
|--------|-----|------|
| Accuracy | 70-85% | 68-82% |
| F1 Score | 0.65-0.80 | 0.65-0.78 |
| Inference Time | ~25-40ms | ~8-15ms |
| Sharpe Ratio | 0.3-1.5 | 0.2-1.3 |

**Key Insights**:
- SNN may excel on volatile periods due to temporal dynamics
- LSTM is faster for inference on standard hardware
- Both models capture momentum patterns in technical indicators
- Performance varies based on market conditions

## 🧠 Model Architectures

### SNN Architecture
```
Input (seq_len × features) → Flatten
→ FC(420) → LIF(128) 
→ FC(128) → LIF(128)
→ FC(64) → LIF(64)
→ FC(2)
```
- **T = 25 timesteps** for rate coding
- **LIF neurons** with tau=2.0
- Total params: ~70K

### LSTM Architecture
```
Input (seq_len, features)
→ LSTM(128, 2 layers, dropout=0.3)
→ FC(64) → ReLU → Dropout(0.3)
→ FC(2)
```
- Total params: ~150K

## 📊 Visualizations

### SNN Outputs
1. **Spike Raster Plot**: Shows spiking activity across layers
2. **Firing Rates**: Neuron-wise spike counts
3. **Training History**: Loss, accuracy, F1 progression
4. **Backtest Results**: Cumulative returns vs buy-and-hold

### LSTM Outputs
1. **Hidden States**: LSTM activations over time
2. **Training History**: Loss, accuracy, F1 progression
3. **Backtest Results**: Cumulative returns vs buy-and-hold
4. **Model Comparison**: SNN vs LSTM metrics

## 🔧 Customization

### Change Stock Ticker
Edit in notebooks:
```python
data_dict = prepare_full_dataset(
    ticker="INFY.NS",  # Change to any NSE stock
    start="2025-03-04",
    end="2026-03-04",
    ...
)
```

### Adjust Model Parameters
**SNN**:
```python
model = SNNStockPredictor(
    hidden_size=256,  # Increase neurons
    T=50,            # More timesteps
    ...
)
```

**LSTM**:
```python
model = LSTMStockPredictor(
    hidden_size=256,
    num_layers=3,
    dropout=0.5,
    ...
)
```

### Change Sequence Length
```python
data_dict = prepare_full_dataset(
    seq_len=30,  # Use 30-day lookback instead of 20
    ...
)
```

## 📚 Key Concepts

### Rate Coding (SNN)
- Input is repeated T times
- Spikes accumulate over timesteps
- Output is average firing rate
- Mimics biological neural encoding

### LIF Neurons
- Membrane potential integrates input current
- Spikes when threshold is reached
- Leaky: voltage decays over time (tau parameter)
- Reset after spiking

### Technical Indicators Used
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility bands
- **SMA/EMA**: Simple/Exponential Moving Averages
- **ATR**: Average True Range (volatility)
- **ROC**: Rate of Change (momentum)

## 🎯 Learning Objectives

1. **Understand SNNs**: How spiking neurons differ from artificial neurons
2. **SpikingJelly Framework**: LIF neurons, rate coding, gradient descent
3. **Time Series Prediction**: Sequential modeling with LSTM vs SNN
4. **Stock Market Features**: Technical indicators for price prediction
5. **Model Comparison**: Accuracy, inference speed, interpretability

## 🔬 Next Steps

### Beginner
- Run both notebooks and compare results
- Modify hyperparameters and observe effects
- Try different stock tickers

### Intermediate
- Add more technical indicators
- Implement temporal coding (instead of rate coding)
- Add validation set for early stopping
- Ensemble multiple models

### Advanced
- Multi-stock prediction
- Add sentiment analysis (news/Twitter)
- Optimize for neuromorphic hardware
- Real-time trading with Streamlit dashboard
- Add attention mechanisms

## 📖 References

- **SpikingJelly**: https://github.com/fangwei123456/spikingjelly
- **LIF Neurons**: Gerstner & Kistler, "Spiking Neuron Models"
- **Stock Prediction**: Fischer & Krauss, "Deep Learning with LSTM for Stock Returns"
- **NSE Data**: Yahoo Finance API

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock prediction models should not be used for actual trading without:
- Extensive backtesting on out-of-sample data
- Risk management strategies
- Understanding of market conditions
- Professional financial advice

**Past performance does not guarantee future results.**

## 🤝 Contributing

Feel free to:
- Add new features (e.g., attention mechanisms)
- Try different SNN architectures
- Implement more backtesting strategies
- Add more visualization tools

## 📝 License

This project is provided as-is for educational purposes.

---

**Happy Learning! 🚀🧠📈**
