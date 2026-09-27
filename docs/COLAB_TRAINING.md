# Google Colab Training Guide

This guide explains how to fine-tune the AI-Powered Contract Intelligence model using Google Colab. The pipeline is optimized for a Tesla T4 GPU (the standard free-tier Colab GPU).

## 1. Setting Up Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new Notebook.
3. In the menu, go to **Runtime > Change runtime type** and select **T4 GPU**.

## 2. Mounting Google Drive & Environment

Run the following cell to mount your drive and navigate to the project root:

```python
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Set the Project Root (adjust if your folder name is different)
PROJECT_ROOT = '/content/drive/MyDrive/Colab Notebooks/Ai Powered Contract Intelligence Analyzer/AI-Powered Contract Intelligence & Risk Scoring'
os.chdir(PROJECT_ROOT)

print("Current Directory:", os.getcwd())
```

## 3. Installing Dependencies

Install only the required packages for training:

```bash
!pip install -r requirements.txt
```

*(Note: We do not install `requirements-dev.txt` in Colab unless you need to run pytest).*

## 4. Running the Smoke Test

Before committing to a long training run, ensure everything is working (data loading, GPU usage, model forward/backward pass) by running a smoke test. The smoke test uses a tiny subset of the data and runs for only 1 epoch.

```bash
!python scripts/train_colab.py --smoke-test --project-root .
```

*Expected output: You should see the model downloading, data loading, and the loss decreasing over a few steps. The script will save outputs to `artifacts/colab_training/`.*

## 5. Full Training

Once the smoke test passes, you can launch the full training. 

**Pro-Tip**: For faster data loading, use the `--copy-to-local` flag. This will copy the ~145MB JSONL dataset from Google Drive to the fast local `/content/temp_data/` directory before training starts. Checkpoints and logs will still be saved directly to your Google Drive.

```bash
!python scripts/train_colab.py --copy-to-local --project-root .
```

### Configuration Details (Tesla T4 Optimized)
- **Mixed Precision (FP16)**: Enabled by default to halve memory usage and speed up training.
- **Batch Size**: 16 (optimized for T4 with FP16).
- **Gradient Accumulation**: 2 steps (effective batch size = 32).
- **Epochs**: 3 (default).

## 6. Output & Checkpoint Locations

The training script automatically creates organized directories within the project root:

- **Checkpoints (per epoch)**: `artifacts/colab_training/checkpoints/`
- **Best Model**: `artifacts/colab_training/best_model/` (saved when validation loss improves)
- **Final Model**: `artifacts/colab_training/final_model/`
- **Metrics/Logs**: `artifacts/colab_training/metrics/` and `artifacts/colab_training/reports/`

## 7. Storage Requirements & Limitations

- **GPU RAM**: The script uses approximately 10-12GB of VRAM with the default settings (Sequence Length 512, Batch Size 16, FP16).
- **Disk Storage**: DistilBERT models are ~260MB each. Saving per-epoch checkpoints plus best/final models will consume around 1.5GB of Google Drive space. Ensure you have enough storage.
- **Session Disconnects**: If Colab disconnects, you can resume by loading the weights from `artifacts/colab_training/checkpoints/epoch_X/` (though standard resuming logic via `trainer` is currently simplified; you would pass the checkpoint directory to `from_pretrained` if modifying the script to resume).
