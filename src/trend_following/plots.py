"""Matplotlib plotting helpers."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trend_following.metrics import drawdown_series
from trend_following.utils import ensure_directory


def _save(fig: plt.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    ensure_directory(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equity_curves(
    equity_curves: dict[str, pd.Series],
    output_path: str | Path,
    title: str = "Equity Curve",
) -> Path:
    """Plot one or more equity curves."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, series in equity_curves.items():
        series.dropna().plot(ax=ax, label=name, linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save(fig, output_path)


def plot_drawdowns(
    returns: dict[str, pd.Series],
    output_path: str | Path,
    title: str = "Drawdown",
) -> Path:
    """Plot drawdown curves for one or more return streams."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, series in returns.items():
        drawdown_series(series.dropna()).plot(ax=ax, label=name, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save(fig, output_path)


def plot_rolling_sharpe(
    returns: pd.Series,
    output_path: str | Path,
    window: int = 252,
    annualization: int = 252,
) -> Path:
    """Plot rolling annualized Sharpe ratio."""
    rolling_mean = returns.rolling(window=window, min_periods=window).mean()
    rolling_std = returns.rolling(window=window, min_periods=window).std(ddof=0)
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(annualization)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    rolling_sharpe.plot(ax=ax, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    ax.grid(True, alpha=0.25)
    return _save(fig, output_path)


def plot_rolling_volatility(
    returns: pd.Series,
    output_path: str | Path,
    window: int = 252,
    annualization: int = 252,
) -> Path:
    """Plot rolling annualized volatility."""
    rolling_vol = returns.rolling(window=window, min_periods=window).std(ddof=0) * np.sqrt(
        annualization
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    rolling_vol.plot(ax=ax, linewidth=1.5)
    ax.set_title(f"Rolling {window}-Day Annualized Volatility")
    ax.set_ylabel("Volatility")
    ax.grid(True, alpha=0.25)
    return _save(fig, output_path)


def plot_positions(
    weights: pd.DataFrame,
    output_path: str | Path,
    ticker: str | None = None,
) -> Path:
    """Plot position weights for one example asset."""
    if weights.empty:
        raise ValueError("Cannot plot empty weights")
    selected = ticker if ticker in weights.columns else weights.columns[0]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    weights[selected].plot(ax=ax, linewidth=1.2)
    ax.set_title(f"Position Weight: {selected}")
    ax.set_ylabel("Weight")
    ax.set_ylim(-0.05, max(1.05, float(weights[selected].max()) + 0.05))
    ax.grid(True, alpha=0.25)
    return _save(fig, output_path)


def plot_heatmap(
    matrix: pd.DataFrame,
    output_path: str | Path,
    title: str,
    cbar_label: str = "Sharpe Ratio",
) -> Path:
    """Plot a simple heatmap without seaborn."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    values = matrix.astype(float).values
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(x) for x in matrix.columns])
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([str(y) for y in matrix.index])
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = values[i, j]
            label = "" if np.isnan(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    return _save(fig, output_path)
