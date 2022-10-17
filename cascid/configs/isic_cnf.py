from cascid.configs import config

BASE_API_URL = "https://api.isic-archive.com/api/v2"
SEARCH_URL = BASE_API_URL + "/images/search"
IMAGES_DIR = config.DATA_DIR / "ISIC"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
METADATA = config.DATA_DIR / "ISIC" / "metadata.csv"
HAIRLESS_DIR = IMAGES_DIR / 'hairless'
HAIRLESS_DIR.mkdir(parents=True, exist_ok=True)
