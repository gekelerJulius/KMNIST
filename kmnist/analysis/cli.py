import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from CONFIG import ANALYSIS, SUBMISSION
from kmnist.analysis.embeddings import compute_embeddings, normalize_embeddings
from kmnist.analysis.method_comparison import (
    select_best_method,
    summarize_predictions,
    validation_predictions,
    validation_split_arrays,
    write_method_comparison,
)
from kmnist.analysis.plots import plot_labeled_prototype_distances, plot_projection
from kmnist.analysis.projections import run_tsne, run_umap
from kmnist.analysis.prototypes import prototype_assignment_metrics
from kmnist.data import FlatImageFolderDataset, LabeledImageFolderDataset
from kmnist.data.transforms import build_test_transform
from kmnist.data.loaders import build_loader
from kmnist.models import Autoencoder
from kmnist.submission.embeddings import compute_embeddings_and_logits
from kmnist.utils.checkpoints import resolve_checkpoint
from kmnist.utils.device import get_device
from kmnist.utils.paths import analysis_output_dir, labeled_dir, labels_csv_path, unlabeled_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create autoencoder embeddings for KMNIST data and visualize them with TSNE and UMAP."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a Lightning checkpoint, checkpoint directory, or staged training run. Defaults to the best available checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=analysis_output_dir(),
        help="Directory where embeddings and plots will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=ANALYSIS.batch_size,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=ANALYSIS.num_workers,
        help="DataLoader worker count. Use 0 if multiprocessing is problematic.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable deterministic classifier test-time augmentation in validation method comparison.",
    )
    return parser.parse_args()


def _analysis_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return build_loader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


def build_analysis_datasets():
    test_transform = build_test_transform()
    unlabeled_dataset = FlatImageFolderDataset(unlabeled_dir(), transform=test_transform)
    labeled_dataset = LabeledImageFolderDataset(labeled_dir(), labels_csv_path(), transform=test_transform)
    return unlabeled_dataset, labeled_dataset


def run_embedding_analysis(
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
    batch_size: int = ANALYSIS.batch_size,
    num_workers: int = ANALYSIS.num_workers,
    use_tta: bool = SUBMISSION.use_tta,
) -> Path:
    checkpoint_path = resolve_checkpoint(checkpoint, checkpoint_kind="embedding")
    output_dir = (output_dir or analysis_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = Autoencoder.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)

    unlabeled_dataset, labeled_dataset = build_analysis_datasets()
    unlabeled_loader = _analysis_loader(unlabeled_dataset, batch_size, num_workers)
    labeled_loader = _analysis_loader(labeled_dataset, batch_size, num_workers)

    unlabeled_embeddings, unlabeled_labels = compute_embeddings(model, unlabeled_loader, device, "unlabeled")
    labeled_embeddings, labeled_logits, labeled_labels = compute_embeddings_and_logits(
        model,
        labeled_loader,
        device,
        "labeled",
        use_tta=use_tta,
    )
    labeled_labels = np.asarray(labeled_labels, dtype=np.int64)

    all_embeddings = np.concatenate([unlabeled_embeddings, labeled_embeddings], axis=0)
    normalized_embeddings = normalize_embeddings(all_embeddings)
    all_labels = np.concatenate(
        [
            np.full(len(unlabeled_labels), -1, dtype=np.int64),
            labeled_labels.astype(np.int64),
        ],
        axis=0,
    )
    split_names = np.concatenate(
        [
            np.full(len(unlabeled_embeddings), "unlabeled", dtype=object),
            np.full(len(labeled_embeddings), "labeled", dtype=object),
        ],
        axis=0,
    )
    labeled_mask = split_names == "labeled"
    split = validation_split_arrays(labeled_dataset, labeled_embeddings, labeled_logits, labeled_labels)
    method_predictions = validation_predictions(split)
    method_summaries = summarize_predictions(method_predictions, split.validation_labels)
    best_method = select_best_method(method_summaries)
    write_method_comparison(output_dir, method_summaries, best_method)

    nearest_labels, nearest_distances, nearest_gaps, distance_summary = prototype_assignment_metrics(
        normalized_embeddings,
        all_labels,
    )

    np.savez_compressed(
        output_dir / "embeddings.npz",
        embeddings=all_embeddings.astype(np.float32),
        normalized_embeddings=normalized_embeddings.astype(np.float32),
        labels=all_labels,
        split_names=split_names,
        nearest_prototype_label=nearest_labels.astype(np.int64),
        nearest_prototype_distance=nearest_distances.astype(np.float32),
        nearest_prototype_gap=nearest_gaps.astype(np.float32),
        checkpoint_path=str(checkpoint_path),
        use_tta=use_tta,
        tta_views=len(SUBMISSION.tta_rotation_degrees) if use_tta else 1,
    )

    tsne_projection = run_tsne(normalized_embeddings)
    np.save(output_dir / "tsne_projection.npy", tsne_projection)
    plot_projection(
        tsne_projection,
        split_names,
        all_labels,
        title="Autoencoder Embeddings - TSNE",
        output_path=output_dir / "tsne.png",
    )
    labeled_tsne_projection = run_tsne(normalized_embeddings[labeled_mask])
    np.save(output_dir / "labeled_tsne_projection.npy", labeled_tsne_projection)
    plot_projection(
        labeled_tsne_projection,
        split_names[labeled_mask],
        all_labels[labeled_mask],
        title="Labeled Autoencoder Embeddings - TSNE",
        output_path=output_dir / "labeled_tsne.png",
        show_unlabeled=False,
    )

    umap_projection = run_umap(normalized_embeddings)
    np.save(output_dir / "umap_projection.npy", umap_projection)
    plot_projection(
        umap_projection,
        split_names,
        all_labels,
        title="Autoencoder Embeddings - UMAP",
        output_path=output_dir / "umap.png",
    )
    labeled_umap_projection = run_umap(normalized_embeddings[labeled_mask])
    np.save(output_dir / "labeled_umap_projection.npy", labeled_umap_projection)
    plot_projection(
        labeled_umap_projection,
        split_names[labeled_mask],
        all_labels[labeled_mask],
        title="Labeled Autoencoder Embeddings - UMAP",
        output_path=output_dir / "labeled_umap.png",
        show_unlabeled=False,
    )
    plot_labeled_prototype_distances(
        distance_summary,
        output_path=output_dir / "labeled_prototype_distances.png",
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved embeddings to: {output_dir / 'embeddings.npz'}")
    print(f"Saved TSNE plot to: {output_dir / 'tsne.png'}")
    print(f"Saved UMAP plot to: {output_dir / 'umap.png'}")
    print(f"Saved labeled-only TSNE plot to: {output_dir / 'labeled_tsne.png'}")
    print(f"Saved labeled-only UMAP plot to: {output_dir / 'labeled_umap.png'}")
    print(f"Saved prototype distance plot to: {output_dir / 'labeled_prototype_distances.png'}")
    print(f"Saved validation method comparison to: {output_dir / 'validation_method_comparison.json'}")
    print(f"Best validation method: {best_method}")
    return output_dir


def main() -> None:
    args = parse_args()
    run_embedding_analysis(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_tta=not args.no_tta,
    )


if __name__ == "__main__":
    main()
