# pylint: disable=missing-docstring
from pathlib import Path
from os import path
import sys

DATA_DIR = Path.home() / ".cascid_data" # Dir to install dataset, based on python install location
DATA_DIR.mkdir(exist_ok=True, parents=True)