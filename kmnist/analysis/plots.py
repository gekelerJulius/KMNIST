import os
from pathlib import Path

from CONFIG import ANALYSIS

os.environ.setdefault("MPLCONFIGDIR", ANALYSIS.matplotlib_config_dir)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_projection(
    projection: np.ndarray,
    split_names: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
    show_unlabeled: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))

    unlabeled_mask = split_names == "unlabeled"
    labeled_mask = split_names == "labeled"

    if show_unlabeled:
        ax.scatter(
            projection[unlabeled_mask, 0],
            projection[unlabeled_mask, 1],
            s=10,
            c="#b0b0b0",
            alpha=0.35,
            linewidths=0,
            label="Unlabeled",
        )

    labeled_projection = projection[labeled_mask]
    labeled_values = labels[labeled_mask]
    cmap = plt.get_cmap("tab10")

    for class_id in sorted(np.unique(labeled_values)):
        class_mask = labeled_values == class_id
        ax.scatter(
            labeled_projection[class_mask, 0],
            labeled_projection[class_mask, 1],
            s=18,
            c=[cmap(int(class_id))],
            alpha=0.9,
            linewidths=0,
            label=f"Label {int(class_id)}",
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", markerscale=1.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_labeled_prototype_distances(
    distance_summary: dict[int, dict[str, float]],
    output_path: Path,
) -> None:
    classes = np.array(sorted(distance_summary), dtype=np.int64)
    means = np.array([distance_summary[int(class_id)]["mean"] for class_id in classes])
    p90s = np.array([distance_summary[int(class_id)]["p90"] for class_id in classes])
    maxes = np.array([distance_summary[int(class_id)]["max"] for class_id in classes])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    ax.bar(x - 0.25, means, width=0.25, label="Mean")
    ax.bar(x, p90s, width=0.25, label="P90")
    ax.bar(x + 0.25, maxes, width=0.25, label="Max")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(class_id)) for class_id in classes])
    ax.set_xlabel("Label")
    ax.set_ylabel("Cosine distance to class prototype")
    ax.set_title("Labeled Prototype Compactness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
