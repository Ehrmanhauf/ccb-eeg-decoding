"""A small EEGNet CNN baseline (Lawhern et al. 2018) for the hardware-efficiency benchmark.

EEGNet is the standard compact convolutional network for EEG decoding; here it is the
*compute-heavy comparator* that makes the lightweight claim for the fixed pipelines and the
CCB concrete (latency / memory / energy), NOT a decoding contributor to the thesis.

Interface mirrors :class:`thesis.baselines.fbcsp.FBCSP` --- ``fit(X, y)`` / ``predict(X)`` over
``X`` of shape ``(n_trials, n_channels, n_samples)`` and a string-label ``y`` --- so the
benchmark can call all systems identically. torch is an optional dependency
(``uv sync --extra benchmark``); importing this module without torch is fine, but constructing
``EEGNet`` then raises a clear error.

Reference: Lawhern et al., "EEGNet: a compact convolutional neural network for EEG-based
brain-computer interfaces", J. Neural Eng. 15(5):056013, 2018 (the EEGNet-8,2 configuration).
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn

    _TORCH = True
except ImportError:  # torch is an optional (benchmark) dependency
    _TORCH = False


if _TORCH:

    class _EEGNetModule(nn.Module):
        """EEGNet-8,2: temporal conv -> depthwise spatial conv -> separable conv -> linear."""

        def __init__(self, *, n_channels: int, n_samples: int, n_classes: int,
                     sfreq: float, f1: int, depth: int, f2: int, dropout: float):
            super().__init__()
            kern = max(2, int(sfreq // 2))  # ~half-second temporal kernel
            self.block1 = nn.Sequential(
                nn.Conv2d(1, f1, (1, kern), padding=(0, kern // 2), bias=False),
                nn.BatchNorm2d(f1),
                # depthwise spatial filter over all channels -> collapses the channel axis
                nn.Conv2d(f1, f1 * depth, (n_channels, 1), groups=f1, bias=False),
                nn.BatchNorm2d(f1 * depth),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout),
            )
            self.block2 = nn.Sequential(
                # separable conv = depthwise (1,16) then pointwise (1,1)
                nn.Conv2d(f1 * depth, f1 * depth, (1, 16), padding=(0, 8),
                          groups=f1 * depth, bias=False),
                nn.Conv2d(f1 * depth, f2, (1, 1), bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout),
            )
            with torch.no_grad():  # infer the flattened dimension
                dummy = torch.zeros(1, 1, n_channels, n_samples)
                flat = self.block2(self.block1(dummy)).flatten(1).shape[1]
            self.classify = nn.Linear(flat, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.block2(self.block1(x))
            return self.classify(x.flatten(1))


class EEGNet:
    """scikit-learn-style EEGNet wrapper (``fit`` / ``predict``), CPU, deterministic."""

    def __init__(self, *, sfreq: float = 250.0, f1: int = 8, depth: int = 2, f2: int = 16,
                 dropout: float = 0.25, lr: float = 1e-3, epochs: int = 50,
                 batch_size: int = 32, seed: int = 42, device: str = "cpu"):
        if not _TORCH:
            raise ImportError(
                "EEGNet requires torch (an optional dependency): uv sync --extra benchmark"
            )
        self.sfreq = sfreq
        self.f1, self.depth, self.f2 = f1, depth, f2
        self.dropout = dropout
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.seed = seed
        self.device = device

    def fit(self, X: np.ndarray, y: np.ndarray) -> EEGNet:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        n_trials, n_channels, n_samples = X.shape

        # per-trial standardisation (EEGNet trains on normalised inputs); store nothing extra
        Xn = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-7)

        self.model_ = _EEGNetModule(
            n_channels=n_channels, n_samples=n_samples, n_classes=len(self.classes_),
            sfreq=self.sfreq, f1=self.f1, depth=self.depth, f2=self.f2, dropout=self.dropout,
        ).to(self.device)

        xb = torch.from_numpy(Xn[:, None, :, :]).to(self.device)
        yb = torch.from_numpy(y_idx.astype(np.int64)).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model_.train()
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            perm = rng.permutation(n_trials)
            for i in range(0, n_trials, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(xb[idx]), yb[idx])
                loss.backward()
                opt.step()
        self.model_.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xn = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-7)
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(Xn[:, None, :, :]).to(self.device))
            idx = logits.argmax(1).cpu().numpy()
        return self.classes_[idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())
