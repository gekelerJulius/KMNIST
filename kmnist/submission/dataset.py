import csv
from pathlib import Path, PureWindowsPath

from PIL import Image
from torch.utils.data import Dataset

from CONFIG import DATA
from kmnist.data.transforms import build_test_transform
from kmnist.utils.paths import data_dir


class SubmissionImageDataset(Dataset):
    def __init__(self, sample_submission_path: Path):
        self.sample_submission_path = sample_submission_path
        self.transform = build_test_transform()
        self.rows = self._load_rows()

    def _load_rows(self) -> list[dict[str, str]]:
        with self.sample_submission_path.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            expected_fields = [DATA.image_path_column, DATA.label_column]
            if reader.fieldnames != expected_fields:
                raise ValueError(f"Expected sample submission columns {expected_fields}, got {reader.fieldnames}")
            rows = list(reader)

        if not rows:
            raise ValueError(f"Sample submission is empty: {self.sample_submission_path}")
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_path_text = self.rows[index][DATA.image_path_column]
        relative_path = Path(*PureWindowsPath(image_path_text).parts)
        image_path = data_dir() / relative_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in sample submission not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("L")
            return self.transform(image), image_path_text
