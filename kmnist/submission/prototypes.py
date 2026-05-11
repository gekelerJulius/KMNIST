import numpy as np
import torch

from kmnist.data import LabeledImageFolderDataset
from kmnist.data.loaders import build_loader
from kmnist.data.transforms import build_test_transform
from kmnist.models import Autoencoder
from kmnist.submission.embeddings import compute_embeddings, normalize_rows
from kmnist.utils.paths import labeled_dir, labels_csv_path


def compute_labeled_prototypes(
    model: Autoencoder,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    labeled_dataset = LabeledImageFolderDataset(
        labeled_dir(),
        labels_csv_path(),
        transform=build_test_transform(),
    )
    labeled_loader = build_loader(labeled_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    labeled_embeddings, labels = compute_embeddings(model, labeled_loader, device, "Embedding labeled reference")
    normalized_embeddings = normalize_rows(labeled_embeddings)
    labels = np.asarray(labels, dtype=np.int64)

    classes = np.array(sorted(np.unique(labels)), dtype=np.int64)
    prototypes = []
    for class_id in classes:
        prototype = normalized_embeddings[labels == class_id].mean(axis=0)
        prototype = prototype / np.clip(np.linalg.norm(prototype), a_min=1e-12, a_max=None)
        prototypes.append(prototype)

    return classes, np.stack(prototypes, axis=0)
