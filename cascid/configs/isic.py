from cascid.configs import config
import os
import sys

BASE_API_URL = "https://api.isic-archive.com/api/v2"
SEARCH_URL = BASE_API_URL + "/images/search"
IMAGE_DIR = config.DATA_DIR / "ISIC"
METADATA = config.DATA_DIR / "ISIC" / "metadata.csv"

if not os.path.isdir(IMAGE_DIR):
    os.mkdir(IMAGE_DIR)