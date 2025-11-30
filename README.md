# MLP Applied - Language Classification with Audio & Text

A PyTorch-based neural network that classifies languages using both audio and text input. The project uses pre-trained models (RemBERT for text, XLS-R for audio) to encode multilingual data.

## Supported Languages
- French
- Polish
- Portuguese
- Italian
- Spanish

## Features
- ✅ Dual-modal input (audio + text)
- ✅ GPU acceleration support (CUDA/CPU auto-detection)
- ✅ Pre-trained encoder models (RemBERT + XLS-R)
- ✅ Configurable neural network architecture
- ✅ Early stopping with patience parameter
- ✅ Model checkpointing and loading
- ✅ Easy-to-use CLI interface

## Installation

### Prerequisites
- Python 3.8+
- pip

### Step 1: Install Dependencies

```bash
pip install torch torchaudio
pip install transformers
pip install librosa soundfile
pip install pandas seaborn matplotlib
pip install scikit-learn
```

### Step 2: Optional - CUDA Support
For GPU acceleration (if you have NVIDIA GPU):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Check your CUDA version and install accordingly. Without this, the project runs on CPU automatically.

## Project Structure

```
MLP_Applied/
├── config/
│   └── Network1.json                  # Network architecture configuration
├── neuralNetwork/                     # Main neural network module
│   ├── __pycache__/                   # Python cache
│   ├── encoder.py                     # RemBERT + XLS-R text/audio encoding
│   ├── perceptron.py                  # Neural network model class
│   ├── manager.py                     # Training management & checkpointing
│   ├── dataset_loader.py              # Dataset loading & preprocessing
│   ├── trainer.py                     # CLI interface (create/train/predict)
│   └── main.py                        # Main entry point
├── ploting/
│   └── plot.py                        # Visualization (stacked bar charts)
├── datasetTrain/                      # Training dataset
│   ├── extract_1/
│   │   ├── text.txt                   # Country name + text content (UTF-8)
│   │   └── audio.opus                 # Audio file (mono, 16kHz)
│   ├── extract_2/
│   │   ├── text.txt
│   │   └── audio.opus
│   └── extract_N/
│       ├── text.txt
│       └── audio.opus
├── checkpoints/                       # Saved models & training checkpoints
│   └── model_name_best.pth            # Best model state dict
├── extract_embeddings.py              # Utility for embedding extraction
├── note.txt                           # Development notes
├── LICENSE                            # Project license
└── README.md                          # This file
```

### Key Files

| File | Purpose |
|------|---------|
| `neuralNetwork/main.py` | Main entry point - run with `python -m neuralNetwork.main` |
| `neuralNetwork/trainer.py` | CLI interface for create/train/predict actions |
| `neuralNetwork/encoder.py` | Encodes text (RemBERT) + audio (XLS-R) to embeddings |
| `neuralNetwork/perceptron.py` | PyTorch neural network model |
| `neuralNetwork/manager.py` | Training loop with early stopping & checkpointing |
| `neuralNetwork/dataset_loader.py` | Loads dataset & handles batching |
| `config/Network1.json` | Network architecture & hyperparameter configuration |
| `ploting/plot.py` | Visualization tool for results |

## Dataset Format

Your dataset should be organized as follows:

```
datasetTrain/
├── extract_1/
│   ├── text.txt                   # Format: Line 1 = Country, Line 2+ = Text content
│   └── audio_file.opus            # .opus audio format (mono, 16kHz)
├── extract_2/
│   ├── text.txt
│   └── audio_file.opus
└── extract_3/
    ├── text.txt
    └── audio_file.opus
```

### text.txt Format
```
Polish
Dzień dobry, jak się masz?
```

## Configuration

Edit `config/Network1.json` to customize the network:

```json
{
    "input_size": 4352,
    "output_size": 5,
    "hiddenLayers": [2048, 1024, 512, 256],
    "activationFunction": "gelu",
    "learningRate": 0.0003,
    "optimizer": "adam",
    "lossFunction": "crossentropy",
    "dropoutRate": 0.2
}
```

**Parameters:**
- `input_size`: 4352 (2304 text + 2048 audio embeddings)
- `output_size`: 5 (number of language classes)
- `hiddenLayers`: List of hidden layer sizes
- `activationFunction`: "relu" or "gelu"
- `learningRate`: Optimizer learning rate
- `dropoutRate`: Dropout probability

## Usage

### 1. Create a New Model

```bash
python -m neuralNetwork.main create --config config/Network1.json
```

When prompted, enter a name for your model (e.g., "MyLanguageModel").

The model will be saved to `checkpoints/MyLanguageModel_model.pth`

### 2. Train a Model

**Continue training an existing model:**
```bash
python -m neuralNetwork.main train checkpoints/MyLanguageModel_model.pth --config config/Network1.json --dataset datasetTrain --epochs 50 --patience 5 --batch_size 32
```

**Parameters:**
- `--config`: Path to configuration file (required for training)
- `--dataset`: Path to training data directory
- `--epochs`: Number of training epochs (default: 50)
- `--patience`: Early stopping patience (default: 5)
- `--batch_size`: Batch size for training (default: 32)

### 3. Make Predictions

```bash
python -m neuralNetwork.main predict checkpoints/MyLanguageModel_model.pth --dataset /path/to/audio_text_files
```

## Training Output

During training, you'll see:
```
[Encoder] Using device: cpu
Dataset loaded: 120 samples found
Pre-computing embeddings... (this may take a while)
Starting training...
Epoch 1/50 | Loss: 1.5763 [BEST]
Epoch 2/50 | Loss: 1.4684 [BEST]
Epoch 3/50 | Loss: 1.2722 [BEST]
...
Best model loaded (Loss: 0.1234)
Training completed successfully.
```

## Model Encoding Details

The model uses:

**Text Encoding (RemBERT):**
- Model: `google/rembert`
- Hidden size: 1152 dimensions
- Output: max + mean pooling = 2304 dimensions

**Audio Encoding (XLS-R):**
- Model: `facebook/wav2vec2-xls-r-300m`
- Input: 16kHz mono audio
- Output: max + mean pooling = 2048 dimensions

**Combined Input:** 4352 dimensions (2304 + 2048)

## Expected Performance

With sufficient data (100+ samples per class):
- Training Loss: 0.1 - 0.5
- Validation Accuracy: 85% - 98%

With limited data (3-10 samples):
- Training Loss: 1.0 - 2.0
- Use more data for better results

## Troubleshooting

### ModuleNotFoundError
Make sure you're in the project root directory and run with `-m`:
```bash
cd /path/to/MLP_Applied
python -m neuralNetwork.main train ...
```

### FileNotFoundError for audio files
Ensure your dataset directory structure matches the required format with `.opus` files.

### CUDA OutOfMemory
If running on GPU with limited VRAM:
```bash
# Reduce batch size
python -m neuralNetwork.main train ... --batch_size 8
```

### Slow encoding
First run downloads models (~5GB total). Subsequent runs use cached models. Consider GPU for ~5-10x speedup.

## File Sizes

Expected downloads:
- RemBERT: ~2.3 GB
- XLS-R: ~1.3 GB
- Total: ~3.6 GB

These are cached in `~/.cache/huggingface/` after first use.

## Performance Tips

1. **Use GPU** - 5-10x faster encoding
2. **More data** - Better accuracy (100+ samples per class)
3. **Longer text** - Richer context (minimum 10-20 words)
4. **Clear audio** - Better audio embeddings

## License

See LICENSE file

## Requirements Summary

```bash
# Core
torch==2.0+
torchaudio==2.0+
transformers==4.30+

# Audio processing
librosa==0.10+
soundfile==0.12+

# Data & ML
numpy==1.24+
pandas==2.0+
scikit-learn==1.3+

# Visualization
matplotlib==3.7+
seaborn==0.12+
```

## Quick Start Example

```bash
# 1. Install dependencies
pip install torch torchaudio transformers librosa soundfile pandas numpy scikit-learn

# 2. Create model
python -m neuralNetwork.main create --config config/Network1.json

# 3. Train
python -m neuralNetwork.main train --config config/Network1.json --dataset datasetTrain --epochs 30 --patience 3

# 4. Done! Best model saved in checkpoints/
```

## Support

For issues or questions, check the project structure and ensure:
- ✅ Dataset format is correct
- ✅ Python 3.8+ installed
- ✅ All dependencies installed
- ✅ Text files are UTF-8 encoded
- ✅ Audio files are in .opus format

