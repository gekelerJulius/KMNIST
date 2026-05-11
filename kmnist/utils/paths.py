from datetime import datetime
from pathlib import Path

from CONFIG import ANALYSIS, DATA, PATHS, PSEUDO_LABELS, STAGED_TRAINING, SUBMISSION, TENSORBOARD, TRAINING


def project_root() -> Path:
    return PATHS.project_root


def data_dir() -> Path:
    return PATHS.data_dir


def outputs_dir() -> Path:
    return PATHS.outputs_dir


def labeled_dir() -> Path:
    return data_dir() / DATA.labeled_dir_name


def unlabeled_dir() -> Path:
    return data_dir() / DATA.unlabeled_dir_name


def labels_csv_path() -> Path:
    return data_dir() / DATA.labels_csv_name


def sample_submission_path() -> Path:
    return data_dir() / DATA.sample_submission_name


def checkpoint_dir() -> Path:
    return outputs_dir() / TRAINING.checkpoint_dir_name


def logs_dir() -> Path:
    return outputs_dir() / TENSORBOARD.log_dir_name


def analysis_output_dir() -> Path:
    return outputs_dir() / ANALYSIS.output_dir_name


def submission_output_path() -> Path:
    return outputs_dir() / SUBMISSION.output_dir_name / SUBMISSION.output_filename


def timestamped_output_dir(parent_name: str, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return outputs_dir() / parent_name / f"{prefix}_{timestamp}"


def timestamped_submission_dir() -> Path:
    return timestamped_output_dir(SUBMISSION.output_dir_name, "submission")


def timestamped_pseudo_label_dir() -> Path:
    return timestamped_output_dir(PSEUDO_LABELS.output_dir_name, "pseudo_labels")


def timestamped_staged_training_dir() -> Path:
    return timestamped_output_dir(STAGED_TRAINING.output_dir_name, "staged_training")
