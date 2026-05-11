from pathlib import Path

import numpy as np

from CONFIG import SUBMISSION
from kmnist.data.loaders import build_loader
from kmnist.models import Autoencoder
from kmnist.submission.dataset import SubmissionImageDataset
from kmnist.submission.embeddings import compute_embeddings_and_logits
from kmnist.submission.writer import write_json, write_submission
from kmnist.utils.device import get_device
from kmnist.utils.paths import sample_submission_path


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def ensemble_classifier_probabilities(
    checkpoint_paths: list[Path],
    sample_submission: Path | None = None,
    batch_size: int = SUBMISSION.batch_size,
    num_workers: int = SUBMISSION.num_workers,
    use_tta: bool = SUBMISSION.use_tta,
) -> tuple[np.ndarray, list[str]]:
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required for ensemble prediction.")

    sample_path = sample_submission.resolve() if sample_submission is not None else sample_submission_path()
    submission_dataset = SubmissionImageDataset(sample_path)
    submission_loader = build_loader(
        submission_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    device = get_device()
    probability_sum = None
    image_paths = None
    for checkpoint_path in checkpoint_paths:
        model = Autoencoder.load_from_checkpoint(checkpoint_path, map_location=device)
        model.to(device)
        _, logits, current_image_paths = compute_embeddings_and_logits(
            model,
            submission_loader,
            device,
            f"Embedding submission images ({checkpoint_path.name})",
            use_tta=use_tta,
        )
        probabilities = softmax(logits)
        probability_sum = probabilities if probability_sum is None else probability_sum + probabilities
        if image_paths is None:
            image_paths = current_image_paths
        elif image_paths != current_image_paths:
            raise ValueError("Submission image order changed while computing ensemble predictions.")

    return probability_sum / len(checkpoint_paths), image_paths or []


def write_ensemble_submission(
    checkpoint_paths: list[Path],
    output_path: Path,
    summary_path: Path,
    sample_submission: Path | None = None,
    batch_size: int = SUBMISSION.batch_size,
    num_workers: int = SUBMISSION.num_workers,
    checkpoint_metadata: list[dict] | None = None,
    use_tta: bool = SUBMISSION.use_tta,
) -> dict:
    probabilities, image_paths = ensemble_classifier_probabilities(
        checkpoint_paths,
        sample_submission=sample_submission,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tta=use_tta,
    )
    predicted_labels = probabilities.argmax(axis=1).astype(np.int64)
    write_submission(output_path, image_paths, predicted_labels)
    unique_labels, label_counts = np.unique(predicted_labels, return_counts=True)
    summary = {
        "method": "classifier_probability_average",
        "use_tta": use_tta,
        "tta_views": len(SUBMISSION.tta_rotation_degrees) if use_tta else 1,
        "checkpoints": [str(path) for path in checkpoint_paths],
        "checkpoint_metadata": checkpoint_metadata or [],
        "rows": int(len(predicted_labels)),
        "label_counts": {str(int(label)): int(count) for label, count in zip(unique_labels, label_counts)},
    }
    write_json(summary_path, summary)
    return summary
