# Self-Pruning Neural Network (CIFAR-10)

A production-style PyTorch project where each weight has a learnable gate. The model learns to prune itself during training by minimizing:

- Classification loss: CrossEntropy
- Sparsity loss: sum of sigmoid(gate_scores) over all prunable layers

Total loss:

\[
\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \sum_i g_i
\]

## Features

- Custom `PrunableLinear` layer with full gradient flow
- Bonus: `PrunableConv2d` for CNN-based prunable model
- Baseline vs prunable comparison
- Multi-lambda experiments (trade-off study)
- Temperature scaling for gates
- Hard-pruning after training
- Checkpoint save/load
- Plotting: gate histogram and sparsity-vs-accuracy
- Optional TorchScript export of best pruned model

## Project Structure

- `model.py` - Layers and model definitions
- `train.py` - Train/eval loops
- `utils.py` - Data, plotting, metrics, checkpoint utilities
- `main.py` - Experiment orchestration
- `requirements.txt` - Dependencies

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run default experiments (baseline + 3 lambda values)

```bash
python main.py --arch cnn --epochs 20 --batch-size 128 --lambdas 1e-5,1e-4,1e-3 --temperature 1.0 --export-pruned
```

### 3) Use MLP (only PrunableLinear in prunable variant)

```bash
python main.py --arch mlp --hidden-sizes 1024,512 --epochs 20 --lambdas 1e-5,1e-4,1e-3
```

### 4) Resume from checkpoint

```bash
python main.py --resume outputs/checkpoints/lambda_1e-04/last.pt
```

## Outputs

By default, generated in `outputs/`:

- `results.csv`, `results.json`
- `checkpoints/` (best + last for each run)
- `plots/gate_hist_lambda_*.png`
- `plots/sparsity_vs_accuracy.png`
- `reports/report.md`
- `best_pruned_model.pt` (if `--export-pruned`)

## Typical Experiment Interpretation

- Small lambda: better accuracy, lower sparsity
- Large lambda: higher sparsity, stronger pruning pressure, possible accuracy drop

Choose lambda according to deployment constraints (memory/FLOPs budget).
