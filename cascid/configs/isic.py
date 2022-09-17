from cascid.configs import config
import os
import sys

BASE_API_URL = "https://api.isic-archive.com/api/v2"
SEARCH_URL = BASE_API_URL + "/images/search"
IMAGE_DIR = config.DATA_DIR / "ISIC"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
METADATA = config.DATA_DIR / "ISIC" / "metadata.csv"