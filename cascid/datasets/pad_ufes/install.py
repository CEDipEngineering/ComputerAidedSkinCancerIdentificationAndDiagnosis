#!/usr/bin/env python3
import os
from pathlib import Path
import requests, zipfile
from io import BytesIO
import time
from shutil import rmtree
from cascid.configs import pad_ufes_cnf

def install_data_ufes(FORCE_INSTALL=False):
    '''
    Function to download and install the PAD UFES dataset @ https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip
    FORCE_INSTALL parameter forces deletion and reinstall of dataset.
    '''
    
    pad_ufes_dir = pad_ufes_cnf.PAD_UFES_DIR # Dir for PAD UFES    
    if FORCE_INSTALL: # Force install ensures download
        
        # rmtree(pad_ufes.PAD_UFES_DIR.parents[1]) # Erase everything
        pad_ufes_cnf.IMAGES_DIR.mkdir(parents=True, exist_ok=True) # Make it real

        # File url 
        url = "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"

        # Split URL to get the file name
        filename = url.split('/')[-1]

        # Downloading the file by sending the request to the URL
        print("Beginning dataset download (SLOW! Can take a few minutes, please be patient)")
        start = time.perf_counter()
        req = requests.get(url)
        print(f'Download completed in {time.perf_counter()-start:.03f}s')

        # Extracting the dataset zip
        zf = zipfile.ZipFile(BytesIO(req.content))
        zf.extractall(pad_ufes_dir)

        # Extract individual image zips
        img_folders = list(filter( lambda x: not x.endswith(".png"), os.listdir(Path(pad_ufes_dir) / "images"))) # Filter img folder zips
        img_zip_paths = list(map(lambda x: str(Path(pad_ufes_dir) / "images" / x), img_folders)) # Find zips
        for zip in img_zip_paths:
            with zipfile.ZipFile(zip, "r") as Zf:
                Zf.extractall(Path(pad_ufes_dir) / "images") # Extract all
            os.remove(zip) # Erase zip after extract

        img_folders = list(filter( lambda x: not x.endswith(".png"), os.listdir(Path(pad_ufes_dir) / "images"))) # Filter img folders
        print(img_folders)

        full_image_folder_paths = [Path(pad_ufes_dir) / "images" / f for f in img_folders] # Full folder paths
        for folder in full_image_folder_paths:
            img_names = os.listdir(folder)
            for img_name in img_names:
                curr_path = folder / img_name
                target_path = Path(pad_ufes_dir) / "images" / img_name
                os.rename(curr_path, target_path)
            rmtree(folder)
            print(f"{folder} done...")


    print(f"Files can be found at {pad_ufes_dir}")
    return True # Success

    
if __name__ == "__main__":
    install_data_ufes(True)