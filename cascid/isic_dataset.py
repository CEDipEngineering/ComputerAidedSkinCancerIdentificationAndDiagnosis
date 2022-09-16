import pandas as pd
from cascid.configs import isic
from cascid import isic_fetcher
import os

def get_db() -> pd.DataFrame:
    """
    Simple function to read metadata csv file for isic dataset.
    """
    df = pd.read_csv(isic.METADATA, index_col=0)
    return df

def update_all_files(df: pd.DataFrame) -> None:
    """
    Function to download files for isic dataset. Supply with dataframe containing columns 'isic_id' and 'image_url', 
    having strings containing the data that would be returned in the same name fields from the ISIC API.

    Will only download images if a file with the target name (<ISIC_ID>.jpg) cannot be found in the directory.
    Images are downloaded to a directory that can be found by checking cascid.configs.isic.IMAGE_DIR variable.

    Example:

    # Verify download for every image in dataset currently.
    df = get_db()
    update_all_files(df)

    """
    print("Downloading missing images:")
    count = len(df["isic_id"])
    i = 0
    for id, url in zip(df["isic_id"], df["image_url"]):
        print("Count: {}/{} ({:.02f}%)\r".format(i, count, (i/count)*100), end="")
        if not os.path.exists(isic.IMAGE_DIR / (id + ".jpg")):
            isic_fetcher.download_image(
                image_url=url,
                isic_id=id
            )
        i += 1
    print(" "*100, "\r", end="")
    print("Images Downloaded!")
    return