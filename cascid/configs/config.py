# pylint: disable=missing-docstring
from pathlib import Path
from os import path
import sys

DATA_DIR = Path.home() / ".cascid_data" # Dir to install dataset, based on user home dir
DATA_DIR.mkdir(exist_ok=True, parents=True)