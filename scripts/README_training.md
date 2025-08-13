# GPT Training Script

This directory contains the consolidated GPT training script that can train models of different sizes.

## Files

- **`gpt.py`** - Main training script (consolidated from individual scripts)
- **`analysis.py`** - Analysis script for order parameters and loss curves

## Usage

### Training Models

Train a model of a specific size:

```bash
# Train 10M parameter model
python gpt.py 10M

# Train 1M parameter model  
python gpt.py 1M

# Train 0.1M parameter model
python gpt.py 0.1M

# Train 10K parameter model
python gpt.py 10K
```

### Command Line Options

```bash
python gpt.py MODEL_SIZE [OPTIONS]

Options:
  --data PATH              Path to input text file (default: text/input.txt)
  --output-dir PATH        Directory to save model checkpoints (default: models/)
  --text-output-dir PATH   Directory to save generated text (default: text/)
  --max-iters INT          Override max training iterations
  --eval-interval INT      Override evaluation interval
  --batch-size INT         Override batch size
  --learning-rate FLOAT    Override learning rate
```

### Examples

```bash
# Quick test run (10 iterations, evaluate every 5)
python gpt.py 10K --max-iters 10 --eval-interval 5

# Custom data path
python gpt.py 1M --data /path/to/custom/text.txt

# Custom output directories
python gpt.py 0.1M --output-dir custom_models --text-output-dir custom_text
```

## Model Configurations

| Model Size | n_embd | n_head | n_layer | Parameters |
|------------|--------|--------|---------|------------|
| 10M        | 384    | 6      | 6       | ~10M       |
| 1M         | 128    | 4      | 4       | ~1M        |
| 0.1M       | 64     | 2      | 2       | ~0.1M      |
| 10K        | 32     | 2      | 1       | ~10K       |

## Training Hyperparameters

- **batch_size**: 64
- **block_size**: 256
- **max_iters**: 20000
- **eval_interval**: 500
- **learning_rate**: 3e-4
- **eval_iters**: 200
- **dropout**: 0.2

## Output Files

For each model size `{SIZE}`, the script generates:

- **Checkpoints**: `models/gpt_{SIZE}_step_{ITER}.pth`
- **Final Model**: `models/gpt_{SIZE}_final.pth`
- **Generated Text**: `text/out_{SIZE}.txt`

## Device Selection

The script automatically selects the best available device:
1. **MPS** (Apple Silicon) - if available
2. **CUDA** - if available  
3. **CPU** - as fallback

## Analysis

After training, use the analysis script to generate plots and analyze order parameters:

```bash
python analysis.py
```

This will generate comprehensive analysis plots and CSV files in the `results/` directory.
