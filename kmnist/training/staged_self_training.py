import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping

from CONFIG import LOSSES, PSEUDO_LABELS, STAGED_TRAINING, SUBMISSION, TRAINING
from kmnist.analysis.cli import run_embedding_analysis
from kmnist.analysis.method_comparison import (
    repeated_validation_summary,
    select_best_method,
    summarize_predictions,
    validation_predictions,
    validation_split_arrays,
)
from kmnist.data import LabeledImageFolderDataset, build_dataloaders
from kmnist.data.loaders import build_loader
from kmnist.data.transforms import build_test_transform
from kmnist.models import Autoencoder
from kmnist.pseudo_labels import PseudoLabelArtifacts, generate_pseudo_labels
from kmnist.submission.embeddings import compute_embeddings_and_logits
from kmnist.submission.ensemble import write_ensemble_submission
from kmnist.submission.cli import write_checkpoint_submission
from kmnist.submission.writer import write_json
from kmnist.training.train import build_trainer
from kmnist.utils.checkpoints import _filename_glob
from kmnist.utils.device import get_device
from kmnist.utils.paths import labeled_dir, labels_csv_path, timestamped_staged_training_dir


@dataclass(frozen=True)
class StageConfig:
    index: int
    pseudo_label_budget: int
    pseudo_labels_csv: Path | None
    pseudo_loss_weight: float
    warm_start_checkpoint: Path | None
    stage_dir: Path
    checkpoint_dir: Path
    pseudo_label_dir: Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run staged semi-supervised KMNIST self-training.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(STAGED_TRAINING.pseudo_label_budgets))
    parser.add_argument("--max-epochs", type=int, default=STAGED_TRAINING.max_epochs)
    parser.add_argument("--patience", type=int, default=STAGED_TRAINING.patience)
    parser.add_argument("--min-delta", type=float, default=STAGED_TRAINING.min_delta)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=float, default=TRAINING.limit_train_batches)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(STAGED_TRAINING.seeds))
    parser.add_argument("--repeated-validation-splits", type=int, default=STAGED_TRAINING.repeated_validation_splits)
    parser.add_argument("--ensemble-size", type=int, default=STAGED_TRAINING.ensemble_size)
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--skip-embedding-analysis", action="store_true")
    parser.add_argument("--skip-submission", action="store_true")
    return parser.parse_args()


def pseudo_loss_weight_for_stage(stage_index: int, stage_count: int) -> float:
    if stage_index <= 0:
        return 0.0
    pseudo_stage_count = max(1, stage_count - 2)
    progress = min(1.0, (stage_index - 1) / pseudo_stage_count)
    return STAGED_TRAINING.pseudo_loss_start_weight + progress * (
        STAGED_TRAINING.pseudo_loss_end_weight - STAGED_TRAINING.pseudo_loss_start_weight
    )


def make_stage_config(
    experiment_dir: Path,
    stage_index: int,
    budget: int,
    stage_count: int,
    pseudo_labels_csv: Path | None,
    warm_start_checkpoint: Path | None,
) -> StageConfig:
    stage_dir = experiment_dir / f"stage_{stage_index:02d}_budget_{budget}"
    return StageConfig(
        index=stage_index,
        pseudo_label_budget=budget,
        pseudo_labels_csv=pseudo_labels_csv,
        pseudo_loss_weight=pseudo_loss_weight_for_stage(stage_index, stage_count),
        warm_start_checkpoint=warm_start_checkpoint,
        stage_dir=stage_dir,
        checkpoint_dir=stage_dir / "checkpoints",
        pseudo_label_dir=stage_dir / "pseudo_labels",
    )


def best_embedding_checkpoint(checkpoint_dir: Path) -> Path:
    pattern = _filename_glob(TRAINING.embedding_checkpoint_filename)
    matches = sorted(checkpoint_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    if matches:
        return matches[-1]
    matches = sorted(checkpoint_dir.glob("best-*.ckpt"), key=lambda path: path.stat().st_mtime)
    if matches:
        return matches[-1]
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.exists():
        return last_checkpoint
    raise FileNotFoundError(f"No checkpoint found in stage checkpoint directory: {checkpoint_dir}")


def best_classifier_checkpoint(checkpoint_dir: Path) -> Path:
    pattern = _filename_glob(TRAINING.classifier_checkpoint_filename)
    matches = sorted(checkpoint_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    if matches:
        return matches[-1]
    return best_embedding_checkpoint(checkpoint_dir)


def final_checkpoint(checkpoint_dir: Path) -> Path | None:
    last_checkpoint = checkpoint_dir / "last.ckpt"
    return last_checkpoint if last_checkpoint.exists() else None


def checkpoint_kind(checkpoint_path: Path) -> str:
    if checkpoint_path.name.startswith(TRAINING.classifier_checkpoint_filename.split("{", 1)[0]):
        return "classifier"
    if checkpoint_path.name.startswith(TRAINING.embedding_checkpoint_filename.split("{", 1)[0]):
        return "embedding"
    if checkpoint_path.name.startswith("last"):
        return "final_or_swa"
    return "unknown"


def monitored_score(trainer: L.Trainer) -> float | None:
    for callback in trainer.callbacks:
        if getattr(callback, "monitor", None) != STAGED_TRAINING.monitor:
            continue
        best_model_score = getattr(callback, "best_model_score", None)
        if best_model_score is None:
            continue
        if isinstance(best_model_score, torch.Tensor):
            return float(best_model_score.detach().cpu().item())
        return float(best_model_score)

    value = trainer.callback_metrics.get(STAGED_TRAINING.monitor)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def stage_improved(score: float | None, best_score: float | None, min_delta: float) -> bool:
    if score is None:
        return False
    if best_score is None:
        return True
    if STAGED_TRAINING.mode == "max":
        return score >= best_score + min_delta
    return score <= best_score - min_delta


def paired_validation_delta(stage_summary: dict, baseline_summary: dict | None) -> dict:
    if baseline_summary is None:
        return {
            "paired_delta_mean_accuracy": None,
            "paired_delta_std_accuracy": None,
            "paired_delta_win_count": None,
            "paired_delta_tie_count": None,
            "paired_delta_loss_count": None,
            "paired_split_deltas": [],
        }

    stage_splits = {
        row["seed"]: float(row["best_accuracy"])
        for row in stage_summary.get("repeated_validation", {}).get("splits", [])
    }
    baseline_splits = {
        row["seed"]: float(row["best_accuracy"])
        for row in baseline_summary.get("repeated_validation", {}).get("splits", [])
    }
    shared_seeds = sorted(set(stage_splits) & set(baseline_splits))
    deltas = [
        {
            "seed": seed,
            "delta_accuracy": stage_splits[seed] - baseline_splits[seed],
            "stage_accuracy": stage_splits[seed],
            "baseline_accuracy": baseline_splits[seed],
        }
        for seed in shared_seeds
    ]
    delta_values = np.array([row["delta_accuracy"] for row in deltas], dtype=np.float64)
    if delta_values.size == 0:
        return {
            "paired_delta_mean_accuracy": None,
            "paired_delta_std_accuracy": None,
            "paired_delta_win_count": None,
            "paired_delta_tie_count": None,
            "paired_delta_loss_count": None,
            "paired_split_deltas": [],
        }
    return {
        "paired_delta_mean_accuracy": float(delta_values.mean()),
        "paired_delta_std_accuracy": float(delta_values.std(ddof=0)),
        "paired_delta_win_count": int((delta_values > 0).sum()),
        "paired_delta_tie_count": int((delta_values == 0).sum()),
        "paired_delta_loss_count": int((delta_values < 0).sum()),
        "paired_split_deltas": deltas,
    }


def ranked_stage_summaries(stage_summaries: list[dict]) -> list[dict]:
    scored_stages = [stage for stage in stage_summaries if stage.get("score") is not None]
    ranked_stages = sorted(
        scored_stages,
        key=lambda stage: (
            stage["score"],
            stage.get(
                "repeated_validation_mean_balanced_accuracy",
                stage.get("final_method_balanced_accuracy", 0.0),
            ),
        ),
        reverse=True,
    )
    return [
        {
            "rank": rank,
            "stage": stage["stage"],
            "pseudo_label_budget": stage["pseudo_label_budget"],
            "score": stage["score"],
            "selected_method": stage.get("selected_method"),
            "final_method_accuracy": stage.get("final_method_accuracy"),
            "final_method_balanced_accuracy": stage.get("final_method_balanced_accuracy"),
            "repeated_validation_mean_accuracy": stage.get("repeated_validation_mean_accuracy", stage["score"]),
            "repeated_validation_mean_balanced_accuracy": stage.get(
                "repeated_validation_mean_balanced_accuracy",
                stage.get("final_method_balanced_accuracy"),
            ),
            "training_monitor_score": stage.get("training_monitor_score"),
            "delta_vs_consistency_only": stage.get("delta_vs_consistency_only"),
            "selected_checkpoint_kind": stage.get("selected_checkpoint_kind"),
            "selected_pseudo_labels": stage["selected_pseudo_labels"],
            "pseudo_loss_weight": stage["pseudo_loss_weight"],
            "best_embedding_checkpoint": stage["best_embedding_checkpoint"],
            "repeated_validation_splits": stage.get("repeated_validation_splits"),
            "repeated_validation_std_accuracy": stage.get("repeated_validation_std_accuracy"),
            "repeated_validation_std_balanced_accuracy": stage.get("repeated_validation_std_balanced_accuracy"),
            "paired_delta_mean_accuracy": stage.get("paired_delta_mean_accuracy"),
            "paired_delta_std_accuracy": stage.get("paired_delta_std_accuracy"),
            "paired_delta_win_count": stage.get("paired_delta_win_count"),
            "paired_delta_tie_count": stage.get("paired_delta_tie_count"),
            "paired_delta_loss_count": stage.get("paired_delta_loss_count"),
        }
        for rank, stage in enumerate(ranked_stages, start=1)
    ]


def write_validation_summary(experiment_dir: Path, validation_ranking: list[dict]) -> None:
    csv_path = experiment_dir / "validation_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "stage",
        "pseudo_label_budget",
        "score",
        "selected_method",
        "final_method_accuracy",
        "final_method_balanced_accuracy",
        "repeated_validation_mean_accuracy",
        "repeated_validation_mean_balanced_accuracy",
        "training_monitor_score",
        "delta_vs_consistency_only",
        "selected_checkpoint_kind",
        "selected_pseudo_labels",
        "pseudo_loss_weight",
        "best_embedding_checkpoint",
        "repeated_validation_splits",
        "repeated_validation_std_accuracy",
        "repeated_validation_std_balanced_accuracy",
        "paired_delta_mean_accuracy",
        "paired_delta_std_accuracy",
        "paired_delta_win_count",
        "paired_delta_tie_count",
        "paired_delta_loss_count",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(validation_ranking)

    markdown_path = experiment_dir / "validation_summary.md"
    with markdown_path.open("w") as markdown_file:
        markdown_file.write("# Validation Summary\n\n")
        markdown_file.write("Primary score: repeated-split validation mean accuracy.\n\n")
        markdown_file.write("| Rank | Stage | Budget | Method | Mean Accuracy | Std Accuracy | Delta vs Consistency | Paired Delta | W/T/L | Mean Balanced Accuracy | Checkpoint | Selected Pseudo Labels |\n")
        markdown_file.write("| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |\n")
        for row in validation_ranking:
            paired_delta = row.get("paired_delta_mean_accuracy")
            paired_delta_text = "" if paired_delta is None else f"{paired_delta:.6f}"
            win_count = row.get("paired_delta_win_count")
            tie_count = row.get("paired_delta_tie_count")
            loss_count = row.get("paired_delta_loss_count")
            paired_record_text = (
                ""
                if win_count is None or tie_count is None or loss_count is None
                else f"{win_count}/{tie_count}/{loss_count}"
            )
            markdown_file.write(
                "| "
                f"{row['rank']} | "
                f"{row['stage']} | "
                f"{row['pseudo_label_budget']} | "
                f"{row['selected_method']} | "
                f"{row['repeated_validation_mean_accuracy']:.6f} | "
                f"{row['repeated_validation_std_accuracy']:.6f} | "
                f"{row['delta_vs_consistency_only']:.6f} | "
                f"{paired_delta_text} | "
                f"{paired_record_text} | "
                f"{row['repeated_validation_mean_balanced_accuracy']:.6f} | "
                f"{row['selected_checkpoint_kind']} | "
                f"{row['selected_pseudo_labels']} |\n"
            )


def evaluate_checkpoint(
    checkpoint_path: Path,
    batch_size: int = SUBMISSION.batch_size,
    num_workers: int = SUBMISSION.num_workers,
    repeated_validation_splits: int = STAGED_TRAINING.repeated_validation_splits,
    use_tta: bool = SUBMISSION.use_tta,
) -> dict:
    device = get_device()
    model = Autoencoder.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    labeled_dataset = LabeledImageFolderDataset(labeled_dir(), labels_csv_path(), transform=build_test_transform())
    labeled_loader = build_loader(labeled_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    labeled_embeddings, labeled_logits, labeled_labels = compute_embeddings_and_logits(
        model,
        labeled_loader,
        device,
        f"Evaluating {checkpoint_path.name}",
        use_tta=use_tta,
    )
    split = validation_split_arrays(
        labeled_dataset,
        labeled_embeddings,
        labeled_logits,
        np.asarray(labeled_labels, dtype=np.int64),
    )
    method_predictions = validation_predictions(split)
    method_summaries = summarize_predictions(method_predictions, split.validation_labels)
    selected_method = select_best_method(method_summaries)
    selected_summary = method_summaries[selected_method]
    repeated_summary = repeated_validation_summary(
        labeled_dataset,
        labeled_embeddings,
        labeled_logits,
        np.asarray(labeled_labels, dtype=np.int64),
        split_count=repeated_validation_splits,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "selected_method": selected_method,
        "final_method_accuracy": selected_summary["accuracy"],
        "final_method_balanced_accuracy": selected_summary["balanced_accuracy"],
        "method_summaries": method_summaries,
        "repeated_validation": repeated_summary,
        "repeated_validation_mean_accuracy": repeated_summary["mean_accuracy"],
        "repeated_validation_std_accuracy": repeated_summary["std_accuracy"],
        "repeated_validation_mean_balanced_accuracy": repeated_summary["mean_balanced_accuracy"],
        "repeated_validation_std_balanced_accuracy": repeated_summary["std_balanced_accuracy"],
        "repeated_validation_selected_method": repeated_summary["best_method"],
        "use_tta": use_tta,
        "tta_views": len(SUBMISSION.tta_rotation_degrees) if use_tta else 1,
    }


def select_final_method_checkpoint(
    checkpoint_dir: Path,
    num_workers: int,
    repeated_validation_splits: int = STAGED_TRAINING.repeated_validation_splits,
) -> tuple[Path, dict]:
    candidates = []
    checkpoint_paths = {
        best_embedding_checkpoint(checkpoint_dir),
        best_classifier_checkpoint(checkpoint_dir),
    }
    last_checkpoint = final_checkpoint(checkpoint_dir)
    if last_checkpoint is not None:
        checkpoint_paths.add(last_checkpoint)
    for checkpoint_path in checkpoint_paths:
        evaluation = evaluate_checkpoint(
            checkpoint_path,
            num_workers=num_workers,
            repeated_validation_splits=repeated_validation_splits,
        )
        candidates.append((checkpoint_path, evaluation))
    return max(
        candidates,
        key=lambda item: (
            item[1]["repeated_validation_mean_accuracy"],
            item[1]["repeated_validation_mean_balanced_accuracy"],
        ),
    )


def train_stage(
    stage: StageConfig,
    max_epochs: int,
    patience: int,
    min_delta: float,
    num_workers: int | None,
    limit_train_batches: float,
) -> tuple[Path, float | None]:
    torch.set_float32_matmul_precision(TRAINING.float32_matmul_precision)
    stage.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if stage.warm_start_checkpoint is not None:
        model = Autoencoder.load_from_checkpoint(stage.warm_start_checkpoint, map_location="cpu")
        model.pseudo_loss_weight = stage.pseudo_loss_weight
        model.consistency_loss_weight = LOSSES.consistency_loss_weight
        model.consistency_confidence_threshold = LOSSES.consistency_confidence_threshold
        model.max_epochs = max_epochs
    else:
        model = Autoencoder(
            pseudo_loss_weight=stage.pseudo_loss_weight,
            consistency_loss_weight=LOSSES.consistency_loss_weight,
            consistency_confidence_threshold=LOSSES.consistency_confidence_threshold,
            max_epochs=max_epochs,
        )
    train_loader, validation_loader = build_dataloaders(
        num_workers=num_workers,
        pseudo_labels_csv=stage.pseudo_labels_csv,
    )
    early_stopping = EarlyStopping(
        monitor=STAGED_TRAINING.monitor,
        mode=STAGED_TRAINING.mode,
        patience=patience,
        min_delta=min_delta,
    )
    trainer = build_trainer(
        max_epochs=max_epochs,
        checkpoint_dirpath=stage.checkpoint_dir,
        early_stopping=early_stopping,
        limit_train_batches=limit_train_batches,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    return best_embedding_checkpoint(stage.checkpoint_dir), monitored_score(trainer)


def run_staged_training(
    output_dir: Path | None = None,
    budgets: list[int] | None = None,
    max_epochs: int = STAGED_TRAINING.max_epochs,
    patience: int = STAGED_TRAINING.patience,
    min_delta: float = STAGED_TRAINING.min_delta,
    num_workers: int | None = None,
    limit_train_batches: float = TRAINING.limit_train_batches,
    seed: int | None = None,
    repeated_validation_splits: int = STAGED_TRAINING.repeated_validation_splits,
) -> Path:
    if seed is not None:
        L.seed_everything(seed, workers=True)
    experiment_dir = output_dir.resolve() if output_dir is not None else timestamped_staged_training_dir()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    budgets = list(budgets or STAGED_TRAINING.pseudo_label_budgets)
    if not budgets or budgets[0] != 0:
        raise ValueError("Staged self-training budgets must start with 0.")
    if STAGED_TRAINING.warm_start_mode != "weights_only":
        raise ValueError(f"Unsupported staged warm-start mode: {STAGED_TRAINING.warm_start_mode}")

    stage_summaries = []
    best_score = None
    baseline_score = None
    baseline_stage_summary = None
    previous_checkpoint = None
    next_pseudo_labels_csv = None

    for stage_index, budget in enumerate(budgets):
        pseudo_artifacts: PseudoLabelArtifacts | None = None
        if stage_index > 0:
            if previous_checkpoint is None:
                raise RuntimeError("Cannot generate pseudo labels before stage 0 has produced a checkpoint.")
            pseudo_artifacts = generate_pseudo_labels(
                checkpoint=previous_checkpoint,
                output_dir=experiment_dir / f"stage_{stage_index:02d}_budget_{budget}" / "pseudo_labels",
                class_cap=PSEUDO_LABELS.class_cap,
                total_cap=budget,
                previous_pseudo_labels=next_pseudo_labels_csv,
            )
            next_pseudo_labels_csv = pseudo_artifacts.pseudo_labels_path

        stage = make_stage_config(
            experiment_dir,
            stage_index,
            budget,
            len(budgets),
            next_pseudo_labels_csv if stage_index > 0 else None,
            previous_checkpoint
            if stage_index > 0 and STAGED_TRAINING.warm_start_pseudo_stages
            else None,
        )
        embedding_checkpoint_path, training_monitor_score = train_stage(
            stage,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            num_workers=num_workers,
            limit_train_batches=limit_train_batches,
        )
        checkpoint_path, evaluation = select_final_method_checkpoint(
            stage.checkpoint_dir,
            num_workers=0 if num_workers is None else num_workers,
            repeated_validation_splits=repeated_validation_splits,
        )
        score = evaluation["repeated_validation_mean_accuracy"]
        if stage_index == 0:
            baseline_score = score
        improved = stage_improved(score, best_score, min_delta)
        if improved:
            best_score = score

        stage_summary = {
            "stage": stage_index,
            "pseudo_label_budget": budget,
            "pseudo_labels_csv": str(stage.pseudo_labels_csv) if stage.pseudo_labels_csv else None,
            "pseudo_loss_weight": stage.pseudo_loss_weight,
            "warm_start_checkpoint": str(stage.warm_start_checkpoint) if stage.warm_start_checkpoint else None,
            "warm_start_mode": STAGED_TRAINING.warm_start_mode if stage.warm_start_checkpoint else None,
            "best_embedding_checkpoint": str(checkpoint_path),
            "training_embedding_checkpoint": str(embedding_checkpoint_path),
            "selected_checkpoint_kind": checkpoint_kind(checkpoint_path),
            "monitor": STAGED_TRAINING.monitor,
            "score": score,
            "delta_vs_consistency_only": None if baseline_score is None else score - baseline_score,
            "training_monitor_score": training_monitor_score,
            "selected_method": evaluation["selected_method"],
            "final_method_accuracy": evaluation["final_method_accuracy"],
            "final_method_balanced_accuracy": evaluation["final_method_balanced_accuracy"],
            "validation_method_comparison": evaluation["method_summaries"],
            "repeated_validation": evaluation["repeated_validation"],
            "repeated_validation_splits": repeated_validation_splits,
            "repeated_validation_mean_accuracy": evaluation["repeated_validation_mean_accuracy"],
            "repeated_validation_std_accuracy": evaluation["repeated_validation_std_accuracy"],
            "repeated_validation_mean_balanced_accuracy": evaluation["repeated_validation_mean_balanced_accuracy"],
            "repeated_validation_std_balanced_accuracy": evaluation["repeated_validation_std_balanced_accuracy"],
            "repeated_validation_selected_method": evaluation["repeated_validation_selected_method"],
            "improved": improved,
            "selected_pseudo_labels": pseudo_artifacts.selected_rows if pseudo_artifacts else 0,
        }
        stage_summary.update(paired_validation_delta(stage_summary, baseline_stage_summary))
        write_json(stage.stage_dir / "stage_summary.json", stage_summary)
        stage_summaries.append(stage_summary)
        if stage_index == 0:
            baseline_stage_summary = stage_summary
        previous_checkpoint = checkpoint_path

    validation_ranking = ranked_stage_summaries(stage_summaries)
    best_stage = validation_ranking[0] if validation_ranking else None
    write_validation_summary(experiment_dir, validation_ranking)

    write_json(
        experiment_dir / "experiment_summary.json",
        {
            "budgets": budgets,
            "seed": seed,
            "max_epochs": max_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "repeated_validation_splits": repeated_validation_splits,
            "monitor": STAGED_TRAINING.monitor,
            "mode": STAGED_TRAINING.mode,
            "best_score": best_score,
            "best_stage": best_stage,
            "validation_ranking": validation_ranking,
            "stages": stage_summaries,
        },
    )
    return experiment_dir


def aggregate_seed_summaries(seed_dirs: list[Path]) -> dict:
    rows_by_budget: dict[int, list[dict]] = {}
    for seed_dir in seed_dirs:
        summary_path = seed_dir / "experiment_summary.json"
        if not summary_path.exists():
            continue
        import json

        summary = json.load(summary_path.open())
        for stage in summary["stages"]:
            rows_by_budget.setdefault(int(stage["pseudo_label_budget"]), []).append(stage)

    aggregate_rows = []
    for budget, rows in sorted(rows_by_budget.items()):
        accuracies = np.array(
            [row.get("repeated_validation_mean_accuracy", row["final_method_accuracy"]) for row in rows],
            dtype=np.float64,
        )
        balanced = np.array(
            [row.get("repeated_validation_mean_balanced_accuracy", row["final_method_balanced_accuracy"]) for row in rows],
            dtype=np.float64,
        )
        deltas = np.array([row["delta_vs_consistency_only"] for row in rows], dtype=np.float64)
        paired_deltas = np.array(
            [
                row["paired_delta_mean_accuracy"]
                for row in rows
                if row.get("paired_delta_mean_accuracy") is not None
            ],
            dtype=np.float64,
        )
        aggregate_rows.append(
            {
                "pseudo_label_budget": budget,
                "runs": len(rows),
                "mean_accuracy": float(accuracies.mean()),
                "std_accuracy": float(accuracies.std(ddof=0)),
                "mean_delta_vs_consistency_only": float(deltas.mean()),
                "mean_paired_delta_accuracy": (
                    None if paired_deltas.size == 0 else float(paired_deltas.mean())
                ),
                "mean_balanced_accuracy": float(balanced.mean()),
                "std_balanced_accuracy": float(balanced.std(ddof=0)),
            }
        )
    ranked = sorted(
        aggregate_rows,
        key=lambda row: (row["mean_accuracy"], row["mean_balanced_accuracy"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return {"validation_ranking": ranked}


def write_aggregate_summary(output_dir: Path, aggregate: dict) -> None:
    write_json(output_dir / "aggregate_summary.json", aggregate)
    rows = aggregate["validation_ranking"]
    with (output_dir / "aggregate_validation_summary.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "rank",
                "pseudo_label_budget",
                "runs",
                "mean_accuracy",
                "std_accuracy",
                "mean_delta_vs_consistency_only",
                "mean_paired_delta_accuracy",
                "mean_balanced_accuracy",
                "std_balanced_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "aggregate_validation_summary.md").open("w") as markdown_file:
        markdown_file.write("# Aggregate Validation Summary\n\n")
        markdown_file.write("| Rank | Budget | Runs | Mean Accuracy | Delta vs Consistency | Paired Delta | Std Accuracy | Mean Balanced Accuracy |\n")
        markdown_file.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            paired_delta = row.get("mean_paired_delta_accuracy")
            paired_delta_text = "" if paired_delta is None else f"{paired_delta:.6f}"
            markdown_file.write(
                f"| {row['rank']} | {row['pseudo_label_budget']} | {row['runs']} | "
                f"{row['mean_accuracy']:.6f} | {row['mean_delta_vs_consistency_only']:.6f} | "
                f"{paired_delta_text} | "
                f"{row['std_accuracy']:.6f} | "
                f"{row['mean_balanced_accuracy']:.6f} |\n"
            )


def staged_checkpoint_rows(seed_dirs: list[Path]) -> list[dict]:
    rows = []
    for seed_dir in seed_dirs:
        summary_path = seed_dir / "experiment_summary.json"
        if not summary_path.exists():
            continue
        import json

        summary = json.load(summary_path.open())
        seed = summary.get("seed")
        for stage in summary.get("stages", []):
            rows.append(
                {
                    "seed": seed,
                    "stage": stage["stage"],
                    "pseudo_label_budget": stage["pseudo_label_budget"],
                    "checkpoint": stage["best_embedding_checkpoint"],
                    "score": stage["score"],
                    "mean_accuracy": stage.get("repeated_validation_mean_accuracy", stage["final_method_accuracy"]),
                    "std_accuracy": stage.get("repeated_validation_std_accuracy"),
                    "mean_balanced_accuracy": stage.get(
                        "repeated_validation_mean_balanced_accuracy",
                        stage["final_method_balanced_accuracy"],
                    ),
                    "std_balanced_accuracy": stage.get("repeated_validation_std_balanced_accuracy"),
                    "selected_method": stage.get("repeated_validation_selected_method", stage.get("selected_method")),
                    "selected_checkpoint_kind": stage.get("selected_checkpoint_kind"),
                    "paired_delta_mean_accuracy": stage.get("paired_delta_mean_accuracy"),
                    "paired_delta_win_count": stage.get("paired_delta_win_count"),
                    "paired_delta_tie_count": stage.get("paired_delta_tie_count"),
                    "paired_delta_loss_count": stage.get("paired_delta_loss_count"),
                }
            )
    ranked = sorted(
        rows,
        key=lambda row: (row["mean_accuracy"], row["mean_balanced_accuracy"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def run_staged_postprocess(
    output_dir: Path,
    seed_dirs: list[Path],
    ensemble_size: int = STAGED_TRAINING.ensemble_size,
    run_analysis: bool = STAGED_TRAINING.postprocess_embedding_analysis,
    run_submission: bool = STAGED_TRAINING.postprocess_submission,
    num_workers: int | None = None,
) -> None:
    checkpoint_rows = staged_checkpoint_rows(seed_dirs)
    if not checkpoint_rows:
        return
    postprocess_dir = output_dir / "postprocess"
    write_json(
        postprocess_dir / "repeated_validation_summary.json",
        {
            "selection_metric": "mean_accuracy",
            "default_submission_mode": STAGED_TRAINING.staged_submission_mode,
            "ensemble_size": ensemble_size,
            "checkpoints": checkpoint_rows,
            "best_checkpoint": checkpoint_rows[0],
            "ensemble_checkpoints": checkpoint_rows[:ensemble_size],
        },
    )

    best_checkpoint = Path(checkpoint_rows[0]["checkpoint"])
    if run_analysis:
        run_embedding_analysis(
            checkpoint=best_checkpoint,
            output_dir=postprocess_dir / "embedding_analysis",
            num_workers=0 if num_workers is None else num_workers,
        )

    if run_submission:
        best_submission_dir = postprocess_dir / "submission_best_single"
        write_checkpoint_submission(
            best_checkpoint,
            output_path=best_submission_dir / SUBMISSION.output_filename,
            diagnostics_path=best_submission_dir / "diagnostics.csv",
            summary_path=best_submission_dir / "summary.json",
            num_workers=0 if num_workers is None else num_workers,
            checkpoint_metadata=checkpoint_rows[0],
        )

        ensemble_rows = checkpoint_rows[:ensemble_size]
        submission_dir = postprocess_dir / f"submission_ensemble_top{ensemble_size}"
        write_ensemble_submission(
            [Path(row["checkpoint"]) for row in ensemble_rows],
            output_path=submission_dir / SUBMISSION.output_filename,
            summary_path=submission_dir / "summary.json",
            num_workers=0 if num_workers is None else num_workers,
            checkpoint_metadata=ensemble_rows,
        )


def run_multi_seed_staged_training(
    output_dir: Path | None = None,
    seeds: list[int] | None = None,
    budgets: list[int] | None = None,
    max_epochs: int = STAGED_TRAINING.max_epochs,
    patience: int = STAGED_TRAINING.patience,
    min_delta: float = STAGED_TRAINING.min_delta,
    num_workers: int | None = None,
    limit_train_batches: float = TRAINING.limit_train_batches,
    repeated_validation_splits: int = STAGED_TRAINING.repeated_validation_splits,
    postprocess: bool = STAGED_TRAINING.postprocess,
    ensemble_size: int = STAGED_TRAINING.ensemble_size,
    postprocess_embedding_analysis: bool = STAGED_TRAINING.postprocess_embedding_analysis,
    postprocess_submission: bool = STAGED_TRAINING.postprocess_submission,
) -> Path:
    experiment_dir = output_dir.resolve() if output_dir is not None else timestamped_staged_training_dir()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds or STAGED_TRAINING.seeds)
    seed_dirs = []
    for seed in seeds:
        seed_dir = experiment_dir / f"seed_{seed}"
        run_staged_training(
            output_dir=seed_dir,
            budgets=budgets,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            num_workers=num_workers,
            limit_train_batches=limit_train_batches,
            seed=seed,
            repeated_validation_splits=repeated_validation_splits,
        )
        seed_dirs.append(seed_dir)
    write_aggregate_summary(experiment_dir, aggregate_seed_summaries(seed_dirs))
    if postprocess:
        run_staged_postprocess(
            experiment_dir,
            seed_dirs,
            ensemble_size=ensemble_size,
            run_analysis=postprocess_embedding_analysis,
            run_submission=postprocess_submission,
            num_workers=num_workers,
        )
    return experiment_dir


def main() -> None:
    args = parse_args()
    experiment_dir = run_multi_seed_staged_training(
        output_dir=args.output_dir,
        seeds=args.seeds,
        budgets=args.budgets,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        num_workers=args.num_workers,
        limit_train_batches=args.limit_train_batches,
        repeated_validation_splits=args.repeated_validation_splits,
        postprocess=not args.skip_postprocess,
        ensemble_size=args.ensemble_size,
        postprocess_embedding_analysis=not args.skip_embedding_analysis,
        postprocess_submission=not args.skip_submission,
    )
    print(f"Wrote staged training experiment: {experiment_dir}")


if __name__ == "__main__":
    main()
