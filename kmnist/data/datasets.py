import csv
from pathlib import Path, PureWindowsPath
from typing import Union

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from CONFIG import DATA, PSEUDO_LABELS


class FlatImageFolderDataset(Dataset):
    def __init__(self, root: Union[str, Path], transform=None):
        self.root = Path(root)
        self.transform = transform or ToTensor()
        self.image_paths: list[Path] = sorted(self.root.glob(DATA.image_glob))

        if not self.image_paths:
            raise ValueError(f"No PNG images found in {self.root}")

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, image_path: Path):
        with Image.open(image_path) as image:
            image = image.convert("L")
            return self.transform(image)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.load_image(image_path)
        return image, 0


class ConsistencyImageFolderDataset(FlatImageFolderDataset):
    def __init__(self, root: Union[str, Path], weak_transform, strong_transform):
        super().__init__(root, transform=None)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("L")
            return (self.weak_transform(image), self.strong_transform(image)), 0


class LabeledImageFolderDataset(FlatImageFolderDataset):
    def __init__(self, root: Union[str, Path], labels_csv: Union[str, Path], transform=None):
        super().__init__(root, transform=transform)
        self.labels_csv = Path(labels_csv)
        self.labels_by_name = self._load_labels()

    def _load_labels(self) -> dict[str, int]:
        if not self.labels_csv.exists():
            raise ValueError(f"Labels CSV not found: {self.labels_csv}")

        with self.labels_csv.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(f"Labels CSV is empty: {self.labels_csv}")

            required_columns = {DATA.image_path_column, DATA.label_column}
            missing_columns = required_columns - set(reader.fieldnames)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"Labels CSV missing required columns: {missing}")

            labels_by_name: dict[str, int] = {}
            for row in reader:
                image_name = PureWindowsPath(row[DATA.image_path_column]).name
                if image_name in labels_by_name:
                    raise ValueError(f"Duplicate label entry for {image_name} in {self.labels_csv}")
                labels_by_name[image_name] = int(row[DATA.label_column])

        image_names = {path.name for path in self.image_paths}
        missing_labels = sorted(image_names - labels_by_name.keys())
        extra_labels = sorted(labels_by_name.keys() - image_names)

        if missing_labels:
            missing_preview = ", ".join(missing_labels[:5])
            raise ValueError(f"Missing labels for images in {self.root}: {missing_preview}")
        if extra_labels:
            extra_preview = ", ".join(extra_labels[:5])
            raise ValueError(f"Labels CSV contains entries not present in {self.root}: {extra_preview}")

        return labels_by_name

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.load_image(image_path)
        label = self.labels_by_name[image_path.name]
        return image, label


class PseudoLabeledImageDataset(Dataset):
    def __init__(self, data_root: Union[str, Path], labels_csv: Union[str, Path], transform=None):
        self.data_root = Path(data_root)
        self.labels_csv = Path(labels_csv)
        self.transform = transform or ToTensor()
        self.rows = self._load_rows()

    def _load_rows(self) -> list[tuple[Path, int, float]]:
        if not self.labels_csv.exists():
            raise ValueError(f"Pseudo-label CSV not found: {self.labels_csv}")

        rows = []
        with self.labels_csv.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(f"Pseudo-label CSV is empty: {self.labels_csv}")
            required_columns = {DATA.image_path_column, DATA.label_column}
            missing_columns = required_columns - set(reader.fieldnames)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"Pseudo-label CSV missing required columns: {missing}")

            for row in reader:
                relative_path = Path(*PureWindowsPath(row[DATA.image_path_column]).parts)
                image_path = self.data_root / relative_path
                if not image_path.exists():
                    raise FileNotFoundError(f"Pseudo-labeled image not found: {image_path}")
                weight = float(row.get(PSEUDO_LABELS.weight_column, 1.0) or 1.0)
                rows.append((image_path, int(row[DATA.label_column]), weight))

        if not rows:
            raise ValueError(f"Pseudo-label CSV has no rows: {self.labels_csv}")
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_path, label, weight = self.rows[index]
        with Image.open(image_path) as image:
            image = image.convert("L")
            return self.transform(image), label, weight
