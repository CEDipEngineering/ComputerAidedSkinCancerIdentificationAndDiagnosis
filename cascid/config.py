# pylint: disable=missing-docstring
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / 'data' / 'PAD-UFES'

DATA_FILE = DATA_DIR / 'metadata.csv'

IMAGE_DIR = DATA_DIR / 'images'
