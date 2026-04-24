import argparse
import os
from typing import Dict, List, Optional

import torch

from model import build_model, hard_prune_model, model_has_prunable_layers
from train import evaluate, fit
from utils import (
    compute_gate_sparsity,
    estimate_dense_deployable_size_mb,
    estimate_effective_pruned_size_mb,
    export_pruned_model,
    get_cifar10_dataloaders,
    get_device,
    load_checkpoint,
    plot_gate_histogram,
    plot_sparsity_vs_accuracy,
    save_results_csv,
    save_results_json,
    set_seed,
    write_report_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-pruning neural network on CIFAR-10")

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--arch",
        type=str,
        default="mlp",
        choices=["mlp", "cnn"],
        help="Backbone family for baseline and prunable comparison",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="1024,512",
        help="Comma-separated hidden sizes for MLP architecture",
    )

    parser.add_argument(
        "--lambdas",
        type=str,
        default="1e-5,1e-4,1e-3",
        help="Comma-separated sparsity lambdas",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sparsity-threshold", type=float, default=1e-2)
    parser.add_argument("--hard-prune-threshold", type=float, default=1e-2)

    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="If set, baseline training is skipped",
    )
    parser.add_argument(
        "--export-pruned",
        action="store_true",
        help="Export best pruned model as TorchScript",
    )

    return parser.parse_args()


def parse_hidden_sizes(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def resolve_model_names(arch: str) -> Dict[str, str]:
    if arch == "mlp":
        return {"baseline": "standard_mlp", "prunable": "prunable_mlp"}
    if arch == "cnn":
        return {"baseline": "standard_cnn", "prunable": "prunable_cnn"}
    raise ValueError(f"Unsupported arch: {arch}")


def run_single_training(
    model_name: str,
    hidden_sizes: List[int],
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    lambda_sparsity: float,
    temperature: float,
    checkpoint_dir: str,
    grad_clip: Optional[float],
    resume_path: Optional[str],
):
    model = build_model(model_name=model_name, hidden_sizes=hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if resume_path is not None and os.path.exists(resume_path):
        print(f"Loading checkpoint: {resume_path}")
        load_checkpoint(resume_path, model=model, optimizer=optimizer, map_location=str(device))

    history, best_acc = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        lambda_sparsity=lambda_sparsity,
        temperature=temperature,
        checkpoint_dir=checkpoint_dir,
        grad_clip=grad_clip,
    )

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        temperature=temperature,
    )

    return model, history, best_acc, test_loss, test_acc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_root = os.path.join(args.output_dir, "checkpoints")
    plots_root = os.path.join(args.output_dir, "plots")
    reports_root = os.path.join(args.output_dir, "reports")
    os.makedirs(checkpoints_root, exist_ok=True)
    os.makedirs(plots_root, exist_ok=True)
    os.makedirs(reports_root, exist_ok=True)

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    lambda_values = [float(v.strip()) for v in args.lambdas.split(",") if v.strip()]
    model_names = resolve_model_names(args.arch)

    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
    )

    baseline_result = None

    if not args.skip_baseline:
        print("\n=== Training baseline model ===")
        baseline_ckpt_dir = os.path.join(checkpoints_root, "baseline")
        baseline_model, _, _, _, baseline_test_acc = run_single_training(
            model_name=model_names["baseline"],
            hidden_sizes=hidden_sizes,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lambda_sparsity=0.0,
            temperature=args.temperature,
            checkpoint_dir=baseline_ckpt_dir,
            grad_clip=args.grad_clip,
            resume_path=None,
        )

        baseline_size = estimate_dense_deployable_size_mb(baseline_model)
        baseline_result = {
            "model": model_names["baseline"],
            "lambda": 0.0,
            "test_accuracy": baseline_test_acc,
            "sparsity_percent": 0.0,
            "test_accuracy_hard_pruned": baseline_test_acc,
            "size_mb_before": baseline_size,
            "size_mb_after": baseline_size,
        }

        print(
            f"Baseline test accuracy: {baseline_test_acc:.2f}% | "
            f"Model size: {baseline_size:.2f} MB"
        )

    prunable_rows: List[Dict] = []
    best_model_obj = None
    best_row = None

    for lam in lambda_values:
        print(f"\n=== Training prunable model with lambda={lam:.1e} ===")
        run_name = f"lambda_{lam:.0e}".replace("-", "neg")
        run_ckpt_dir = os.path.join(checkpoints_root, run_name)

        model, history, best_acc, test_loss, test_acc = run_single_training(
            model_name=model_names["prunable"],
            hidden_sizes=hidden_sizes,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lambda_sparsity=lam,
            temperature=args.temperature,
            checkpoint_dir=run_ckpt_dir,
            grad_clip=args.grad_clip,
            resume_path=args.resume,
        )

        del history, best_acc, test_loss

        sparsity_percent = compute_gate_sparsity(
            model,
            threshold=args.sparsity_threshold,
            temperature=args.temperature,
        )

        size_before = estimate_dense_deployable_size_mb(model)
        pruned_weights = 0
        if model_has_prunable_layers(model):
            pruned_weights = hard_prune_model(
                model,
                threshold=args.hard_prune_threshold,
                temperature=args.temperature,
            )

        criterion = torch.nn.CrossEntropyLoss()
        _, test_acc_hard = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            temperature=args.temperature,
        )

        size_after = estimate_effective_pruned_size_mb(
            model,
            threshold=args.hard_prune_threshold,
            temperature=args.temperature,
        )

        row = {
            "model": model_names["prunable"],
            "lambda": lam,
            "test_accuracy": test_acc,
            "sparsity_percent": sparsity_percent,
            "test_accuracy_hard_pruned": test_acc_hard,
            "size_mb_before": size_before,
            "size_mb_after": size_after,
            "hard_pruned_weights": pruned_weights,
        }
        prunable_rows.append(row)

        gate_plot_path = os.path.join(plots_root, f"gate_hist_lambda_{lam:.0e}.png")
        plot_gate_histogram(
            model,
            save_path=gate_plot_path,
            temperature=args.temperature,
            title=f"Gate Distribution (lambda={lam:.1e})",
        )

        print(
            f"lambda={lam:.1e} | acc={test_acc:.2f}% | sparsity={sparsity_percent:.2f}% | "
            f"hard-pruned acc={test_acc_hard:.2f}% | "
            f"size before={size_before:.2f}MB | size after={size_after:.2f}MB | "
            f"newly pruned weights={pruned_weights}"
        )

        if best_row is None or row["test_accuracy"] > best_row["test_accuracy"]:
            best_row = row
            best_model_obj = model

    sparsity_plot_path = os.path.join(plots_root, "sparsity_vs_accuracy.png")
    plot_sparsity_vs_accuracy(prunable_rows, save_path=sparsity_plot_path)

    if args.export_pruned and best_model_obj is not None:
        export_path = os.path.join(args.output_dir, "best_pruned_model.pt")
        export_pruned_model(best_model_obj, export_path=export_path, device=device)
        print(f"Exported TorchScript pruned model: {export_path}")

    all_rows = []
    if baseline_result is not None:
        all_rows.append(baseline_result)
    all_rows.extend(prunable_rows)

    save_results_json(all_rows, os.path.join(args.output_dir, "results.json"))
    save_results_csv(all_rows, os.path.join(args.output_dir, "results.csv"))

    gate_hist_path = None
    if best_row is not None:
        gate_hist_path = os.path.join(
            plots_root,
            f"gate_hist_lambda_{best_row['lambda']:.0e}.png",
        )

    write_report_markdown(
        report_path=os.path.join(reports_root, "report.md"),
        baseline_result=baseline_result,
        prunable_results=prunable_rows,
        best_row=best_row,
        gate_hist_path=gate_hist_path,
        sparsity_plot_path=sparsity_plot_path,
    )

    print("\n=== Experiment complete ===")
    print(f"Results: {os.path.join(args.output_dir, 'results.csv')}")
    print(f"Report : {os.path.join(reports_root, 'report.md')}")


if __name__ == "__main__":
    main()
