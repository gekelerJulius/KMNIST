"""KMNIST training, analysis, and submission package."""

import os

from CONFIG import ANALYSIS

os.environ.setdefault("MPLCONFIGDIR", ANALYSIS.matplotlib_config_dir)
