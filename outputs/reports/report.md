# Self-Pruning Neural Network Report

## Why L1 on Sigmoid Gates Encourages Sparsity

Each gate is defined as g = sigmoid(s), where s is the learnable gate score. The sparsity penalty adds the sum of all gate values to the loss. Because this term is minimized when each gate moves toward 0, and because L1-like penalties prefer exact zeros over small dense values, many gate activations collapse near zero, effectively shutting off their associated weights.

## Results Table

| Model | Lambda | Test Accuracy (%) | Sparsity Level (%) | Test Acc After Hard Prune (%) |
|---|---:|---:|---:|---:|
| Baseline | 0.0 | 67.00 | 0.00 | 67.00 |
| Prunable | 1.0e-05 | 59.99 | 0.00 | 59.99 |
| Prunable | 1.0e-04 | 60.12 | 0.00 | 60.12 |
| Prunable | 1.0e-03 | 56.18 | 0.00 | 56.18 |

## Best Model Summary

Best lambda: 1.0e-04, accuracy: 60.12%, sparsity: 0.00%

## Gate Distribution Plot

![Gate Histogram](./outputs/plots/gate_hist_lambda_1e-04.png)

## Sparsity vs Accuracy Plot

![Sparsity vs Accuracy](./outputs/plots/sparsity_vs_accuracy.png)