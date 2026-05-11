import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kmnist.models import Autoencoder
from kmnist.submission.tta import tta_image_batches


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, a_min=1e-12, a_max=None)


def compute_embeddings(
    model: Autoencoder,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> tuple[np.ndarray, list]:
    embeddings = []
    metadata = []

    model.eval()
    with torch.no_grad():
        for images, batch_metadata in tqdm(loader, desc=desc, unit="batch"):
            images = images.to(device, non_blocking=True)
            batch_embeddings = model.encode(images).cpu().numpy()
            embeddings.append(batch_embeddings)

            if isinstance(batch_metadata, torch.Tensor):
                metadata.extend(batch_metadata.cpu().numpy().tolist())
            else:
                metadata.extend(list(batch_metadata))

    return np.concatenate(embeddings, axis=0), metadata


def compute_embeddings_and_logits(
    model: Autoencoder,
    loader: DataLoader,
    device: torch.device,
    desc: str,
    use_tta: bool = False,
) -> tuple[np.ndarray, np.ndarray, list]:
    embeddings = []
    logits = []
    metadata = []

    model.eval()
    with torch.no_grad():
        for images, batch_metadata in tqdm(loader, desc=desc, unit="batch"):
            images = images.to(device, non_blocking=True)
            _, batch_logits, batch_embeddings = model(images)
            if use_tta:
                logit_sum = batch_logits
                tta_batches = tta_image_batches(images, use_tta=True)
                for augmented_images in tta_batches[1:]:
                    _, augmented_logits, _ = model(augmented_images)
                    logit_sum = logit_sum + augmented_logits
                batch_logits = logit_sum / len(tta_batches)
            embeddings.append(batch_embeddings.cpu().numpy())
            logits.append(batch_logits.cpu().numpy())

            if isinstance(batch_metadata, torch.Tensor):
                metadata.extend(batch_metadata.cpu().numpy().tolist())
            else:
                metadata.extend(list(batch_metadata))

    return np.concatenate(embeddings, axis=0), np.concatenate(logits, axis=0), metadata
