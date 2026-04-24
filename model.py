import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """Linear layer with learnable gates on each weight."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_scores)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def get_gates(self, temperature: float = 1.0) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores / temperature)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        gates = self.get_gates(temperature)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)


class PrunableConv2d(nn.Module):
    """Conv2d layer with learnable gates on each kernel weight."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.gate_scores = nn.Parameter(
            torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_scores)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def get_gates(self, temperature: float = 1.0) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores / temperature)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        gates = self.get_gates(temperature)
        pruned_weight = self.weight * gates
        return F.conv2d(
            x,
            pruned_weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class PrunableMLP(nn.Module):
    """Fully-connected network built using only PrunableLinear layers."""

    def __init__(self, hidden_sizes: List[int], num_classes: int = 10):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain at least one value")

        layers: List[nn.Module] = [nn.Flatten()]
        in_features = 3 * 32 * 32

        for hidden in hidden_sizes:
            layers.append(PrunableLinear(in_features, hidden))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden

        layers.append(PrunableLinear(in_features, num_classes))
        self.net = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        out = x
        for layer in self.net:
            if isinstance(layer, PrunableLinear):
                out = layer(out, temperature=temperature)
            else:
                out = layer(out)
        return out


class StandardMLP(nn.Module):
    """Baseline MLP for comparison."""

    def __init__(self, hidden_sizes: List[int], num_classes: int = 10):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain at least one value")

        layers: List[nn.Module] = [nn.Flatten()]
        in_features = 3 * 32 * 32

        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_features, hidden), nn.ReLU(inplace=True)])
            in_features = hidden

        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        del temperature
        return self.net(x)


class PrunableCNN(nn.Module):
    """CNN with prunable convolution and linear layers."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.ModuleList(
            [
                PrunableConv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                PrunableConv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        )

        self.classifier = nn.ModuleList(
            [
                nn.Flatten(),
                PrunableLinear(64 * 8 * 8, 256),
                nn.ReLU(inplace=True),
                PrunableLinear(256, num_classes),
            ]
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        out = x
        for layer in self.features:
            if isinstance(layer, PrunableConv2d):
                out = layer(out, temperature=temperature)
            else:
                out = layer(out)

        for layer in self.classifier:
            if isinstance(layer, PrunableLinear):
                out = layer(out, temperature=temperature)
            else:
                out = layer(out)
        return out


class StandardCNN(nn.Module):
    """Baseline CNN for comparison."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        del temperature
        out = self.features(x)
        return self.classifier(out)


def iter_prunable_modules(model: nn.Module) -> Iterable[nn.Module]:
    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            yield module


def model_has_prunable_layers(model: nn.Module) -> bool:
    return any(True for _ in iter_prunable_modules(model))


def compute_sparsity_loss(model: nn.Module, temperature: float = 1.0) -> torch.Tensor:
    penalties = []
    for module in iter_prunable_modules(model):
        penalties.append(module.get_gates(temperature).sum())

    if penalties:
        return torch.stack(penalties).sum()

    device = next(model.parameters()).device
    return torch.tensor(0.0, device=device)


def collect_gate_values(model: nn.Module, temperature: float = 1.0) -> torch.Tensor:
    all_gates = []
    for module in iter_prunable_modules(model):
        all_gates.append(module.get_gates(temperature).detach().flatten())

    if not all_gates:
        return torch.empty(0)

    return torch.cat(all_gates)


def hard_prune_model(
    model: nn.Module,
    threshold: float = 1e-2,
    temperature: float = 1.0,
) -> int:
    """Zero out weights whose gates are below threshold.

    Returns number of newly pruned weights.
    """

    pruned_count = 0
    for module in iter_prunable_modules(model):
        with torch.no_grad():
            gates = module.get_gates(temperature)
            mask = gates < threshold
            pruned_count += int(mask.sum().item())

            module.weight.data[mask] = 0.0
            # Push gate scores deep into negative region so gates remain near zero.
            module.gate_scores.data[mask] = -20.0

    return pruned_count


def build_model(model_name: str, hidden_sizes: List[int]) -> nn.Module:
    if model_name == "prunable_mlp":
        return PrunableMLP(hidden_sizes=hidden_sizes)
    if model_name == "standard_mlp":
        return StandardMLP(hidden_sizes=hidden_sizes)
    if model_name == "prunable_cnn":
        return PrunableCNN()
    if model_name == "standard_cnn":
        return StandardCNN()

    raise ValueError(f"Unsupported model_name: {model_name}")
