# pylint: disable=missing-docstring
from pathlib import Path
from os import path
import sys

DATA_DIR = Path(path.split(sys.executable)[0]) / "cascid_data" # Dir to install dataset, based on python install location
    