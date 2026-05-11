import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kmnist.models import Autoencoder


def compute_embeddings(
    model: Autoencoder,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = []
    labels = []

    model.eval()
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc=f"Embedding {split_name}", unit="batch"):
            images = images.to(device, non_blocking=True)
            batch_embeddings = model.encode(images).cpu().numpy()
            embeddings.append(batch_embeddings)
            labels.append(batch_labels.cpu().numpy())

    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)
