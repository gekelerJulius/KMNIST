import os

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pin_memory_enabled() -> bool:
    return torch.cuda.is_available()


def get_num_workers(max_workers: int) -> int:
    return min(max_workers, os.cpu_count() or 0)
