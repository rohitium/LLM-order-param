# A Large Language Model Order Parameter

Unsupervised pre-training of a language model on a tiny shakespeare dataset, following [Andrej Karpathy](https://github.com/karpathy/nanoGPT), and analysis using statistical physics tools

## Directory Structure

- **`scripts/`** - Python scripts for training models and analyzing results
- **`models/`** - Model checkpoint files (10K, 0.1M, 1M, 10M parameters)
- **`text/`** - Input text files and generated outputs
- **`results/`** - Analysis results and plots

## Quick Start

### Training
```bash
cd scripts
python gpt.py
```

### Analysis
```bash
cd scripts
python analysis.py
```