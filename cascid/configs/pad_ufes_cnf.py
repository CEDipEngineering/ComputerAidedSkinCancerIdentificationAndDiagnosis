# pylint: disable=missing-docstring
from cascid.configs import config

# Configs for PAD-UFES dataset.
PAD_UFES_DIR = config.DATA_DIR / 'PAD-UFES'
METADATA = PAD_UFES_DIR / 'metadata.csv'
IMAGES_DIR = PAD_UFES_DIR / 'images'
PREPRO_DIR = PAD_UFES_DIR / 'preprocessed_hair'
IMAGES_DIR.mkdir(parents=True, exist_ok=True) # Make it real
HAIRLESS_QUANTIZED_DIR = IMAGES_DIR / 'hairless_quantized'
HAIRLESS_QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)
HAIRLESS_DIR = IMAGES_DIR / "hairless"
HAIRLESS_DIR.mkdir(parents=True, exist_ok=True) # Make it real