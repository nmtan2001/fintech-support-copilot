"""Load the official BANKING77 train and test CSVs."""

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/"
    "task-specific-datasets/master/banking_data/{split}.csv"
)
SPLITS = ("train", "test")


def download_data(data_dir: Path) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        destination = data_dir / f"{split}.csv"
        if destination.exists():
            continue
        try:
            with urlopen(DATA_URL.format(split=split), timeout=30) as response:
                payload = response.read()
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not download {split}.csv; place the official CSV at "
                f"{destination} or retry with network access"
            ) from exc
        pending = destination.with_suffix(".csv.part")
        pending.write_bytes(payload)
        pending.replace(destination)


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"Unknown BANKING77 split: {split}")
    path = Path(data_dir) / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare_data.py first")
    frame = pd.read_csv(path)
    if not {"text", "category"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain 'text' and 'category' columns")
    frame = frame[["text", "category"]].copy()
    if frame.isna().any().any():
        raise ValueError(f"{path} contains missing text or category values")
    for column in ("text", "category"):
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{path} contains blank or invalid {column} values")
        frame[column] = frame[column].str.strip()
    return frame


def training_split(
    frame: pd.DataFrame, *, validation_size: float = 0.1, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    train, validation = train_test_split(
        frame, test_size=validation_size, random_state=seed, stratify=frame["category"]
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def label_maps(train: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(train["category"].unique().tolist())
    label2id = {name: index for index, name in enumerate(labels)}
    return label2id, {index: name for name, index in label2id.items()}


def ensure_known_labels(frame: pd.DataFrame, label2id: dict[str, int]) -> None:
    unknown = set(frame["category"]) - set(label2id)
    if unknown:
        raise ValueError(f"Unknown intent labels: {sorted(unknown)}")
