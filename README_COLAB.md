# Self-Pruning Neural Network: Google Colab Guide

This guide is a Colab-first walkthrough to run the project end-to-end on CIFAR-10.

## 1) Create a New Colab Notebook

1. Open Google Colab.
2. Create a new notebook.
3. Enable GPU:
   Runtime > Change runtime type > Hardware accelerator > GPU.

## 2) Run Cells in This Exact Order

### Cell 1: Clone the Repository

```python
!git clone https://github.com/potnuruyaswanth/Self-pruning-neural-network.git
%cd Self-pruning-neural-network
```

### Cell 2: Install Dependencies

```python
!pip install -r requirements.txt
```

### Cell 3: Quick Sanity Run (Short)

Use this to verify everything works before a long training run.

```python
!python main.py --arch cnn --epochs 3 --batch-size 128 --lambdas 1e-5,1e-4,1e-3 --temperature 1.0 --export-pruned
```

### Cell 4: Full Experiment Run (Recommended)

```python
!python main.py --arch cnn --epochs 20 --batch-size 128 --lambdas 1e-5,1e-4,1e-3 --temperature 1.0 --export-pruned
```

### Cell 5: Optional MLP-only PrunableLinear Experiment

```python
!python main.py --arch mlp --hidden-sizes 1024,512 --epochs 20 --lambdas 1e-5,1e-4,1e-3 --temperature 1.0
```

### Cell 6: Inspect Generated Outputs

```python
!ls -R outputs
!cat outputs/reports/report.md
```

### Cell 7: Visualize Plots Inline

```python
from IPython.display import Image, display

display(Image(filename='outputs/plots/sparsity_vs_accuracy.png'))
```

```python
import glob
from IPython.display import Image, display

for p in sorted(glob.glob('outputs/plots/gate_hist_lambda_*.png')):
    print(p)
    display(Image(filename=p))
```

### Cell 8: Download Artifacts to Local Machine

```python
from google.colab import files

files.download('outputs/results.csv')
files.download('outputs/results.json')
```

Download best exported model (if `--export-pruned` was used):

```python
from google.colab import files

files.download('outputs/best_pruned_model.pt')
```

## 3) Resume Training from Checkpoint (Optional)

If your runtime disconnects:

```python
!python main.py --arch cnn --epochs 20 --batch-size 128 --lambdas 1e-4 --resume outputs/checkpoints/lambda_1e-04/last.pt
```

## 4) Recommended Colab Settings

- For reliability in Colab, set `--num-workers 0` if dataloader worker issues appear.
- If you see CUDA out-of-memory:
  - reduce `--batch-size` to 64 or 32.
  - reduce model complexity (use `--arch mlp` or fewer epochs).
- Start with a short run (`--epochs 3`) to validate setup.

## 5) What Outputs Mean

- `outputs/results.csv`: table containing lambda, accuracy, sparsity, and model sizes.
- `outputs/plots/sparsity_vs_accuracy.png`: trade-off curve across lambda values.
- `outputs/plots/gate_hist_lambda_*.png`: gate-value distributions for each lambda.
- `outputs/reports/report.md`: auto-generated summary and results table.

## 6) Typical Command Variants

High sparsity push:

```python
!python main.py --arch cnn --epochs 20 --lambdas 1e-4,1e-3,3e-3 --temperature 1.0
```

Higher accuracy bias (lighter pruning):

```python
!python main.py --arch cnn --epochs 20 --lambdas 1e-6,5e-6,1e-5 --temperature 1.0
```

Sharper gate behavior via temperature:

```python
!python main.py --arch cnn --epochs 20 --lambdas 1e-5,1e-4,1e-3 --temperature 0.5
```

## 7) Colab Checklist

- GPU enabled before training
- Dependencies installed
- Ran short sanity run first
- Full run executed
- Results and plots downloaded
