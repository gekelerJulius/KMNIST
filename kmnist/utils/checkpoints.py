import json
from pathlib import Path

from CONFIG import TRAINING
from kmnist.utils.paths import checkpoint_dir


def resolve_checkpoint(
    checkpoint_arg: Path | None = None,
    checkpoint_kind: str = "embedding",
) -> Path:
    if checkpoint_arg is not None:
        checkpoint_path = checkpoint_arg.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if checkpoint_path.is_dir():
            return resolve_checkpoint_directory(checkpoint_path, checkpoint_kind=checkpoint_kind)
        return checkpoint_path

    checkpoints_dir = checkpoint_dir()
    for pattern in _checkpoint_patterns(checkpoint_kind):
        best_checkpoints = sorted(
            checkpoints_dir.glob(pattern),
            key=lambda checkpoint_path: checkpoint_path.stat().st_mtime,
        )
        if best_checkpoints:
            return best_checkpoints[-1]

    last_checkpoint = checkpoints_dir / "last.ckpt"
    if last_checkpoint.exists():
        return last_checkpoint

    legacy_checkpoints_dir = checkpoint_dir().parent.parent / "checkpoints"
    for pattern in _checkpoint_patterns(checkpoint_kind):
        legacy_best_checkpoints = sorted(
            legacy_checkpoints_dir.glob(pattern),
            key=lambda checkpoint_path: checkpoint_path.stat().st_mtime,
        )
        if legacy_best_checkpoints:
            return legacy_best_checkpoints[-1]

    legacy_last_checkpoint = legacy_checkpoints_dir / "last.ckpt"
    if legacy_last_checkpoint.exists():
        return legacy_last_checkpoint

    staged_checkpoint = latest_staged_training_checkpoint(checkpoint_kind=checkpoint_kind)
    if staged_checkpoint is not None:
        return staged_checkpoint

    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint or place a checkpoint in outputs/checkpoints."
    )


def resolve_checkpoint_directory(checkpoint_dir_arg: Path, checkpoint_kind: str = "embedding") -> Path:
    postprocess_summary = checkpoint_dir_arg / "postprocess" / "repeated_validation_summary.json"
    if postprocess_summary.exists():
        return checkpoint_from_postprocess_summary(postprocess_summary)

    if (checkpoint_dir_arg / "stage_summary.json").exists():
        return checkpoint_from_stage_summary(checkpoint_dir_arg / "stage_summary.json")

    if (checkpoint_dir_arg / "experiment_summary.json").exists():
        return checkpoint_from_experiment_summary(checkpoint_dir_arg / "experiment_summary.json")

    seed_summaries = sorted(checkpoint_dir_arg.glob("seed_*/experiment_summary.json"))
    if seed_summaries:
        return checkpoint_from_best_seed_summary(seed_summaries)

    for pattern in _checkpoint_patterns(checkpoint_kind):
        matches = sorted(checkpoint_dir_arg.glob(pattern), key=lambda path: path.stat().st_mtime)
        if matches:
            return matches[-1]

    last_checkpoint = checkpoint_dir_arg / "last.ckpt"
    if last_checkpoint.exists():
        return last_checkpoint

    raise FileNotFoundError(f"No checkpoint found in directory: {checkpoint_dir_arg}")


def latest_staged_training_checkpoint(checkpoint_kind: str = "embedding") -> Path | None:
    staged_parent = checkpoint_dir().parent / "staged_training"
    if not staged_parent.exists():
        return None

    runs = sorted(
        (path for path in staged_parent.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    for run_dir in reversed(runs):
        try:
            return resolve_checkpoint_directory(run_dir, checkpoint_kind=checkpoint_kind)
        except FileNotFoundError:
            continue
    return None


def checkpoint_from_stage_summary(summary_path: Path) -> Path:
    summary = _read_json(summary_path)
    checkpoint = summary.get("best_embedding_checkpoint") or summary.get("training_embedding_checkpoint")
    if not checkpoint:
        raise FileNotFoundError(f"Stage summary does not contain a checkpoint path: {summary_path}")
    return _existing_checkpoint_path(checkpoint, summary_path)


def checkpoint_from_postprocess_summary(summary_path: Path) -> Path:
    summary = _read_json(summary_path)
    best_checkpoint = summary.get("best_checkpoint") or {}
    checkpoint = best_checkpoint.get("checkpoint")
    if not checkpoint:
        raise FileNotFoundError(f"Postprocess summary does not contain a best checkpoint: {summary_path}")
    return _existing_checkpoint_path(checkpoint, summary_path)


def ensemble_checkpoints_from_staged_run(run_dir: Path, ensemble_size: int | None = None) -> tuple[list[Path], list[dict]]:
    summary_path = run_dir / "postprocess" / "repeated_validation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Staged run does not contain repeated validation metadata: {summary_path}")
    summary = _read_json(summary_path)
    rows = summary.get("ensemble_checkpoints") or summary.get("checkpoints") or []
    if ensemble_size is not None:
        rows = rows[:ensemble_size]
    checkpoint_paths = [_existing_checkpoint_path(row["checkpoint"], summary_path) for row in rows]
    return checkpoint_paths, rows


def checkpoint_from_experiment_summary(summary_path: Path) -> Path:
    summary = _read_json(summary_path)
    best_stage = summary.get("best_stage")
    if not best_stage:
        stages = summary.get("stages") or []
        if not stages:
            raise FileNotFoundError(f"Experiment summary does not contain stages: {summary_path}")
        best_stage = max(stages, key=lambda stage: float(stage.get("score", float("-inf"))))

    checkpoint = best_stage.get("best_embedding_checkpoint") or best_stage.get("training_embedding_checkpoint")
    if not checkpoint:
        raise FileNotFoundError(f"Best stage does not contain a checkpoint path: {summary_path}")
    return _existing_checkpoint_path(checkpoint, summary_path)


def checkpoint_from_best_seed_summary(seed_summaries: list[Path]) -> Path:
    best: tuple[float, Path] | None = None
    for summary_path in seed_summaries:
        summary = _read_json(summary_path)
        best_stage = summary.get("best_stage")
        if not best_stage:
            continue
        score = float(best_stage.get("score", summary.get("best_score", float("-inf"))))
        if best is None or score > best[0]:
            best = (score, summary_path)

    if best is None:
        raise FileNotFoundError("No best stage found in staged training seed summaries.")
    return checkpoint_from_experiment_summary(best[1])


def _read_json(path: Path) -> dict:
    with path.open() as json_file:
        return json.load(json_file)


def _existing_checkpoint_path(checkpoint: str, summary_path: Path) -> Path:
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = summary_path.parent / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint from {summary_path} not found: {checkpoint_path}")
    return checkpoint_path


def _checkpoint_patterns(checkpoint_kind: str) -> list[str]:
    if checkpoint_kind == "embedding":
        return [_filename_glob(TRAINING.embedding_checkpoint_filename), "best-*.ckpt"]
    if checkpoint_kind == "classifier":
        return [_filename_glob(TRAINING.classifier_checkpoint_filename), "best-*.ckpt"]
    if checkpoint_kind == "latest_best":
        return ["best-*.ckpt"]
    raise ValueError(f"Unsupported checkpoint kind: {checkpoint_kind}")


def _filename_glob(filename: str) -> str:
    return filename.split("{", 1)[0] + "*.ckpt"
