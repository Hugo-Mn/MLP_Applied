# Setup Python on Mac

This guide will help you install Python 3.8+ on macOS to run the MLP Applied project.

## Prerequisites

- macOS 10.14 or later
- Internet connection
- Terminal access

## Installation Options

### Option 1: Using Homebrew (Recommended)

Homebrew is the easiest way to install Python on Mac.

#### Step 1: Install Homebrew

If you don't have Homebrew installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Step 2: Install Python

```bash
brew install python@3.11
```

#### Step 3: Verify Installation

```bash
python3 --version
```

You should see output like: `Python 3.11.x`

#### Step 4: Upgrade pip

```bash
python3 -m pip install --upgrade pip
```

---

### Option 2: Using MacPorts

#### Step 1: Install MacPorts

Visit https://www.macports.org/install.php and download the installer for your macOS version.

#### Step 2: Install Python

```bash
sudo port install python311
```

#### Step 3: Set Python 3.11 as default

```bash
sudo port select --set python3 python311
```

#### Step 4: Verify Installation

```bash
python3 --version
```

#### Step 5: Upgrade pip

```bash
python3 -m pip install --upgrade pip
```

---

### Option 3: Download from Python.org

#### Step 1: Download Installer

1. Visit https://www.python.org/downloads/
2. Click "Download Python 3.11.x" (or latest 3.8+)
3. Download the macOS installer

#### Step 2: Install Python

1. Open the downloaded `.pkg` file
2. Follow the installation wizard
3. Complete the installation

#### Step 3: Verify Installation

```bash
python3 --version
```

#### Step 4: Upgrade pip

```bash
python3 -m pip install --upgrade pip
```

---

## Next Steps: Install Project Dependencies

Once Python is installed, navigate to the project directory and install dependencies:

```bash
cd /path/to/MLP_Applied

# Install required packages
pip install torch torchaudio
pip install transformers
pip install librosa soundfile
pip install pandas seaborn matplotlib
pip install scikit-learn
```

Or use the quick install command:

```bash
pip install torch torchaudio transformers librosa soundfile pandas seaborn matplotlib scikit-learn
```

### Optional: GPU Support (CUDA) for Mac

Note: Most Macs use Apple Silicon (M1/M2/M3) or Intel processors. CUDA is primarily for NVIDIA GPUs (not available on Mac). However, PyTorch supports Metal acceleration on Apple Silicon:

For Apple Silicon (M1/M2/M3):
```bash
pip install torch torchvision torchaudio
```

The CPU version works fine and will automatically use Metal acceleration when available.

---

## Verify Setup

Test your Python installation:

```bash
python3 << EOF
import torch
import transformers
import librosa

print("Python version:", torch.__version__)
print("PyTorch installed:", torch.__version__)
print("Transformers installed:", transformers.__version__)
print("All dependencies OK!")
EOF
```

You should see version information printed.

---

## Troubleshooting

### Command not found: python3

If you get "command not found", the installation didn't work. Try:

```bash
# Check if Python is installed
which python3

# Or use full path
/usr/local/bin/python3 --version
```

### Permission Denied

If you see permission errors, try adding `--user`:

```bash
pip install --user package_name
```

### Upgrade pip first

```bash
python3 -m pip install --upgrade pip
```

### Use Python Virtual Environment (Recommended)

Create an isolated environment for this project:

```bash
# Create virtual environment
python3 -m venv mlp_env

# Activate it
source mlp_env/bin/activate

# Now install packages
pip install torch torchaudio transformers librosa soundfile pandas seaborn matplotlib scikit-learn

# When done, deactivate with:
deactivate
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install torch` |
| `pip: command not found` | Use `python3 -m pip install` instead |
| Permission errors | Use `--user` flag or create virtual environment |
| Slow installation | Check internet connection, try `pip install -U` |

---

## Ready to Use!

Once setup is complete, you can run the project:

```bash
cd /path/to/MLP_Applied

# Create a model
python -m neuralNetwork.main create --config config/Network1.json

# Train
python -m neuralNetwork.main train --config config/Network1.json --dataset datasetTrain --epochs 10

# Evaluate
python -m neuralNetwork.main evaluate checkpoints/default_model.pth --config config/Network1.json --dataset datasetTrain
```

Enjoy! 🚀
