import csv
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import collect_gate_values, iter_prunable_modules


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cifar10_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    use_pin_memory = torch.cuda.is_available()

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, test_loader


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    nonzero = 0
    for p in model.parameters():
        nonzero += int((p != 0).sum().item())
    return nonzero


def estimate_model_size_mb(model: nn.Module, nonzero_only: bool = False) -> float:
    if nonzero_only:
        total_params = count_nonzero_parameters(model)
    else:
        total_params = count_total_parameters(model)

    # 4 bytes per fp32 parameter
    return (total_params * 4) / (1024**2)


def estimate_dense_deployable_size_mb(model: nn.Module) -> float:
    """Dense deployable size: excludes gate score tensors from prunable layers."""
    total_params = 0
    gate_param_ids = set()
    for module in iter_prunable_modules(model):
        gate_param_ids.add(id(module.gate_scores))

    for p in model.parameters():
        if id(p) in gate_param_ids:
            continue
        total_params += p.numel()

    return (total_params * 4) / (1024**2)


def estimate_effective_pruned_size_mb(
    model: nn.Module,
    threshold: float = 1e-2,
    temperature: float = 1.0,
) -> float:
    """Effective deployable size after pruning based on active gates."""
    total_params = 0
    prunable_weight_ids = set()
    gate_param_ids = set()

    for module in iter_prunable_modules(model):
        gates = module.get_gates(temperature)
        active_weights = int((gates >= threshold).sum().item())
        total_params += active_weights

        if module.bias is not None:
            total_params += module.bias.numel()

        prunable_weight_ids.add(id(module.weight))
        gate_param_ids.add(id(module.gate_scores))
        if module.bias is not None:
            prunable_weight_ids.add(id(module.bias))

    for p in model.parameters():
        pid = id(p)
        if pid in gate_param_ids or pid in prunable_weight_ids:
            continue
        total_params += p.numel()

    return (total_params * 4) / (1024**2)


def compute_gate_sparsity(
    model: nn.Module, threshold: float = 1e-2, temperature: float = 1.0
) -> float:
    gates = collect_gate_values(model, temperature=temperature)
    if gates.numel() == 0:
        return 0.0

    sparse = (gates < threshold).float().mean().item()
    return sparse * 100.0


def save_checkpoint(path: str, state: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_results_json(results: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_results_csv(results: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not results:
        return

    keys = []
    seen = set()
    for row in results:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in keys})


def plot_gate_histogram(
    model: nn.Module,
    save_path: str,
    temperature: float = 1.0,
    bins: int = 80,
    title: str = "Gate Value Distribution",
) -> None:
    gates = collect_gate_values(model, temperature=temperature).cpu().numpy()
    if gates.size == 0:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=bins, alpha=0.85, edgecolor="black")
    plt.title(title)
    plt.xlabel("Gate value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_sparsity_vs_accuracy(results: List[Dict], save_path: str) -> None:
    if not results:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = [row["sparsity_percent"] for row in results]
    y = [row["test_accuracy"] for row in results]
    labels = [str(row["lambda"]) for row in results]

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    for xi, yi, lbl in zip(x, y, labels):
        plt.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Sparsity vs Accuracy Trade-off")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def write_report_markdown(
    report_path: str,
    baseline_result: Optional[Dict],
    prunable_results: List[Dict],
    best_row: Optional[Dict],
    gate_hist_path: Optional[str],
    sparsity_plot_path: Optional[str],
) -> None:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = []
    lines.append("# Self-Pruning Neural Network Report\n")
    lines.append("## Why L1 on Sigmoid Gates Encourages Sparsity\n")
    lines.append(
        "Each gate is defined as g = sigmoid(s), where s is the learnable gate score. "
        "The sparsity penalty adds the sum of all gate values to the loss. "
        "Because this term is minimized when each gate moves toward 0, and because L1-like "
        "penalties prefer exact zeros over small dense values, many gate activations collapse "
        "near zero, effectively shutting off their associated weights.\n"
    )

    lines.append("## Results Table\n")
    lines.append("| Model | Lambda | Test Accuracy (%) | Sparsity Level (%) | Test Acc After Hard Prune (%) |")
    lines.append("|---|---:|---:|---:|---:|")

    if baseline_result is not None:
        lines.append(
            f"| Baseline | 0.0 | {baseline_result['test_accuracy']:.2f} | 0.00 | {baseline_result['test_accuracy']:.2f} |"
        )

    for row in prunable_results:
        lines.append(
            f"| Prunable | {row['lambda']:.1e} | {row['test_accuracy']:.2f} | "
            f"{row['sparsity_percent']:.2f} | {row['test_accuracy_hard_pruned']:.2f} |"
        )

    if best_row is not None:
        lines.append("\n## Best Model Summary\n")
        lines.append(
            f"Best lambda: {best_row['lambda']:.1e}, "
            f"accuracy: {best_row['test_accuracy']:.2f}%, "
            f"sparsity: {best_row['sparsity_percent']:.2f}%"
        )

    if gate_hist_path:
        gate_hist_path_norm = gate_hist_path.replace("\\", "/")
        lines.append("\n## Gate Distribution Plot\n")
        lines.append(f"![Gate Histogram]({gate_hist_path_norm})")

    if sparsity_plot_path:
        sparsity_plot_path_norm = sparsity_plot_path.replace("\\", "/")
        lines.append("\n## Sparsity vs Accuracy Plot\n")
        lines.append(f"![Sparsity vs Accuracy]({sparsity_plot_path_norm})")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_pruned_model(model: nn.Module, export_path: str, device: torch.device) -> None:
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    model.eval()
    sample = torch.randn(1, 3, 32, 32, device=device)
    traced = torch.jit.trace(model, sample)
    traced.save(export_path)
