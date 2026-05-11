import csv
import json
from pathlib import Path

import numpy as np

from CONFIG import DATA


def write_submission(output_path: Path, image_paths: list[str], labels: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[DATA.image_path_column, DATA.label_column])
        writer.writeheader()
        for image_path, label in zip(image_paths, labels):
            writer.writerow({DATA.image_path_column: image_path, DATA.label_column: int(label)})


def write_json(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)


def write_diagnostics(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        DATA.image_path_column,
        "Label",
        "PrototypeLabel",
        "PrototypeDistance",
        "PrototypeMargin",
        "ClassifierLabel",
        "ClassifierConfidence",
        "DecisionReason",
        "MaxDistance",
        "MinMargin",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
