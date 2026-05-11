import math
import random
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset

from CONFIG import PSEUDO_LABELS, TRAINING
from kmnist.data.datasets import ConsistencyImageFolderDataset, LabeledImageFolderDataset, PseudoLabeledImageDataset
from kmnist.data.transforms import build_strong_train_transform, build_test_transform, build_train_transform, build_weak_train_transform
from kmnist.utils.device import get_num_workers, pin_memory_enabled
from kmnist.utils.paths import data_dir, labeled_dir, labels_csv_path, project_root, unlabeled_dir

try:
    from lightning.pytorch.utilities.combined_loader import CombinedLoader
except ImportError:  # pragma: no cover - kept for older Lightning installs.
    from lightning.pytorch.utilities.combined_loader import CombinedLoader


def build_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory_enabled(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def stratified_labeled_split(
    dataset: LabeledImageFolderDataset,
    validation_fraction: float = TRAINING.validation_fraction,
    seed: int = TRAINING.validation_seed,
) -> tuple[Subset, Subset]:
    indices_by_label: dict[int, list[int]] = defaultdict(list)
    for index, image_path in enumerate(dataset.image_paths):
        label = dataset.labels_by_name[image_path.name]
        indices_by_label[label].append(index)

    rng = random.Random(seed)
    train_indices = []
    validation_indices = []
    total_validation_count = int(round(len(dataset) * validation_fraction))
    validation_counts = {}
    remainders = []
    for label in sorted(indices_by_label):
        label_indices = indices_by_label[label]
        exact_count = len(label_indices) * validation_fraction
        validation_count = int(math.floor(exact_count))
        if len(label_indices) > 1:
            validation_count = min(max(validation_count, 1), len(label_indices) - 1)
        validation_counts[label] = validation_count
        remainders.append((exact_count - validation_count, label))

    remaining_validation_count = total_validation_count - sum(validation_counts.values())
    for _, label in sorted(remainders, reverse=True):
        if remaining_validation_count <= 0:
            break
        label_count = len(indices_by_label[label])
        if validation_counts[label] >= label_count - 1:
            continue
        validation_counts[label] += 1
        remaining_validation_count -= 1

    for label in sorted(indices_by_label):
        label_indices = indices_by_label[label]
        rng.shuffle(label_indices)
        validation_count = validation_counts[label]
        validation_indices.extend(label_indices[:validation_count])
        train_indices.extend(label_indices[validation_count:])

    return Subset(dataset, sorted(train_indices)), Subset(dataset, sorted(validation_indices))


def _resolve_optional_project_path(path):
    if path is None:
        return None
    resolved_path = Path(path)
    if resolved_path.is_absolute():
        return resolved_path
    return project_root() / resolved_path


def build_dataloaders(
    num_workers: int | None = None,
    pseudo_labels_csv: Path | str | None = PSEUDO_LABELS.labels_csv,
    use_consistency_views: bool = True,
) -> tuple[CombinedLoader, DataLoader]:
    train_transform = build_train_transform()
    weak_transform = build_weak_train_transform()
    strong_transform = build_strong_train_transform()
    test_transform = build_test_transform()
    unlabeled_dataset = (
        ConsistencyImageFolderDataset(unlabeled_dir(), weak_transform, strong_transform)
        if use_consistency_views
        else ConsistencyImageFolderDataset(unlabeled_dir(), train_transform, train_transform)
    )
    labeled_train_dataset = LabeledImageFolderDataset(labeled_dir(), labels_csv_path(), transform=train_transform)
    labeled_validation_dataset = LabeledImageFolderDataset(labeled_dir(), labels_csv_path(), transform=test_transform)
    train_labeled_split, validation_labeled_split = stratified_labeled_split(labeled_train_dataset)
    train_labeled_dataset = Subset(labeled_train_dataset, train_labeled_split.indices)
    validation_labeled_dataset = Subset(labeled_validation_dataset, validation_labeled_split.indices)
    pseudo_labels_csv = _resolve_optional_project_path(pseudo_labels_csv)
    pseudo_dataset = (
        PseudoLabeledImageDataset(data_dir(), pseudo_labels_csv, transform=train_transform)
        if pseudo_labels_csv is not None
        else None
    )

    if num_workers is None:
        num_workers = get_num_workers(TRAINING.max_num_workers)

    unlabeled_loader = build_loader(
        unlabeled_dataset,
        batch_size=TRAINING.batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    labeled_loader = build_loader(
        train_labeled_dataset,
        batch_size=TRAINING.batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    validation_loader = build_loader(
        validation_labeled_dataset,
        batch_size=TRAINING.reference_batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    loaders = {
        "unlabeled": unlabeled_loader,
        "labeled": labeled_loader,
    }
    if pseudo_dataset is not None:
        loaders["pseudo"] = build_loader(
            pseudo_dataset,
            batch_size=TRAINING.batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
    train_loader = CombinedLoader(loaders, mode="max_size_cycle")
    return train_loader, validation_loader


def build_labeled_reference_loader(
    batch_size: int = TRAINING.reference_batch_size,
    num_workers: int | None = None,
) -> DataLoader:
    labeled_dataset = LabeledImageFolderDataset(labeled_dir(), labels_csv_path(), transform=build_test_transform())

    if num_workers is None:
        num_workers = get_num_workers(TRAINING.max_num_workers)

    return build_loader(
        labeled_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
