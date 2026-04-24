from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model import compute_sparsity_loss, model_has_prunable_layers
from utils import save_checkpoint


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    temperature: float = 1.0,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, temperature=temperature)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lambda_sparsity: float,
    temperature: float,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    model.train()

    running_loss = 0.0
    running_cls = 0.0
    running_sparse = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images, temperature=temperature)
        cls_loss = criterion(logits, labels)

        sparse_loss = torch.tensor(0.0, device=device)
        if lambda_sparsity > 0 and model_has_prunable_layers(model):
            sparse_loss = compute_sparsity_loss(model, temperature=temperature)

        loss = cls_loss + lambda_sparsity * sparse_loss
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_cls += cls_loss.item() * batch_size
        running_sparse += sparse_loss.item() * batch_size

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    return {
        "loss": running_loss / total,
        "cls_loss": running_cls / total,
        "sparse_loss": running_sparse / total,
        "accuracy": 100.0 * correct / total,
    }


def fit(
    model: nn.Module,
    train_loader,
    test_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    lambda_sparsity: float,
    temperature: float,
    checkpoint_dir: Optional[str] = None,
    grad_clip: Optional[float] = None,
) -> Tuple[list, float]:
    criterion = nn.CrossEntropyLoss()
    history = []
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            lambda_sparsity=lambda_sparsity,
            temperature=temperature,
            grad_clip=grad_clip,
        )

        test_loss, test_acc = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            temperature=temperature,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_cls_loss": train_metrics["cls_loss"],
            "train_sparse_loss": train_metrics["sparse_loss"],
            "train_acc": train_metrics["accuracy"],
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | "
            f"Sparse Loss: {train_metrics['sparse_loss']:.4f}"
        )

        if checkpoint_dir is not None:
            save_checkpoint(
                path=f"{checkpoint_dir}/last.pt",
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                },
            )

        if test_acc > best_acc:
            best_acc = test_acc
            if checkpoint_dir is not None:
                save_checkpoint(
                    path=f"{checkpoint_dir}/best.pt",
                    state={
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_acc": best_acc,
                    },
                )

    return history, best_acc
