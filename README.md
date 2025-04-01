# MST-Former: Multi-Scale Spatio-Temporal Transformer for Traffic Forecasting

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

## Introduction

MST-Former is a novel deep learning architecture that leverages the power of transformers for multi-scale spatio-temporal traffic forecasting. It employs a dual-domain attention mechanism to capture complex dependencies in both spatial and temporal dimensions, resulting in more accurate traffic prediction.

Key innovations include:

- **Dual-Domain Attention**: A specialized attention mechanism that processes spatial and temporal dimensions separately
- **Frequency-Feature Cross Modulation (FFCM)**: A module that enhances feature representation by combining spatial and frequency domain information
- **Residual Cycle Forecasting (RCF)**: A technique that captures cyclical patterns in traffic data at different time scales
- **Hierarchical Partition Strategy**: A method for efficiently handling large-scale road networks

## Datasets

The model supports multiple datasets, including:

- **SD**: San Diego traffic dataset
- **GBA**: Greater Bay Area traffic dataset
- **GLA**: Greater Los Angeles traffic dataset

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy 1.19+
- tqdm 4.62+
- configparser 5.0+
- timm 0.4.12+
- pandas 1.3+
- matplotlib 3.4+
- scikit-learn 0.24+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model on a specific dataset:

```bash
python main.py --config config/SD.conf
```

### Testing

To test a trained model:

```bash
python main.py --config config/SD.conf
```

By default, the code will run in test mode.

## Project Structure

```
MST-Former/
├── config/               # Configuration files
│   ├── SD.conf           # San Diego dataset config
│   ├── GBA.conf          # Greater Bay Area dataset config
│   └── GLA.conf          # Greater Los Angeles dataset config
├── data/                 # Data directory
├── lib/                  # Utility libraries
│   ├── data_loader.py    # Data loading
│   ├── metrics.py        # Evaluation metrics
│   └── data_processing.py # Data processing
├── models/               # Model directory
│   └── mst_former.py     # MST-Former model implementation
├── cpt/                  # Model checkpoints
├── log/                  # Log directory
├── main.py               # Main program
├── requirements.txt      # Dependency list
└── LICENSE               # License file
```

## Contact

For questions or suggestions, please open an issue or contact the author. 