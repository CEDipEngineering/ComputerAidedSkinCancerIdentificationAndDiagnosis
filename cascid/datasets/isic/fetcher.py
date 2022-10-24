#!/usr/bin/env python3
from typing import List
from cascid.configs import config, isic_cnf
import requests
import json
import urllib
import pandas as pd

class LesionImage():
    def __init__(self, isic_id: str, sex: str, diagnostic: str, age_approx: int, image_url: str):
        self.isic_id = isic_id
        self.sex = sex
        self.diagnostic = diagnostic
        self.age_approx = age_approx
        self.image_url = image_url

    def to_dict(self):
        return self.__dict__

    def __str__(self) -> str:
        return "Image {}: {};\n".format(self.isic_id, self.diagnostic)

    def __repr__(self) -> str:
        return self.__str__()

def _fetch(url, params=None):
    output = {
        "next" : "",
        "lesion_image_list" : [] 
    }
    resp = requests.get(url, params=params) # Send get request to api using next field from previous request
    resp = json.loads(resp.content) # Convert bytes to json
    output["next"] = resp["next"] # Save next url
    # For each result in response, store required information
    for res in resp["results"]:
        try:
            isic_id=res["isic_id"],
            sex=res["metadata"]["clinical"]["sex"],
            diag=res["metadata"]["clinical"]["diagnosis"],
            # anatom_site_general=res["metadata"]["clinical"]["anatom_site_general"],
            # diagnosis_confirm_type=res["metadata"]["clinical"]["diagnosis_confirm_type"],
            age_approx=int(res["metadata"]["clinical"]["age_approx"]),
            image_url=res["files"]["full"]["url"]
        except KeyError as e:
            # print("Image {} is missing error {}".format(isic_id[0], e))
            continue
        # Extract useful fields from response JSON
        output["lesion_image_list"].append(
            LesionImage(
                isic_id=isic_id[0],
                sex=sex[0],
                diagnostic=diag[0],
                # anatom_site_general=anatom_site_general[0],
                # diagnosis_confirm_type=diagnosis_confirm_type[0],
                age_approx=age_approx[0],
                image_url=image_url
            )
        )
    return output

def fetch_from_isic(n_samples: int, diagnosis_list: List[str]) -> List[LesionImage]:
    # Output list created empty
    lesion_image_list = []
    print("Fetching {} images from ISIC dataset for each of {} diagnosis".format(n_samples, diagnosis_list))
    # If small or not round number of samples
    if n_samples < 100 or n_samples % 100 != 0:
        # For each requested diagnosis, collect n_samples
        for diagnosis in diagnosis_list:        
            limit = n_samples
            query_params = {"limit":limit, "query":"diagnosis:" + diagnosis}
            params = urllib.parse.urlencode(query_params, quote_via=urllib.parse.quote) # Encode special characters such as space and quotation marks
            out = _fetch(isic_cnf.SEARCH_URL, params=params)
            lesion_image_list += out["lesion_image_list"]
    else:
        next_urls = dict()
        n_samples -= 100
        # For each requested diagnosis, collect 100
        for diagnosis in diagnosis_list:        
            limit = 100
            query_params = {"limit":limit, "query":"diagnosis:" + diagnosis}
            params = urllib.parse.urlencode(query_params, quote_via=urllib.parse.quote) # Encode special characters such as space and quotation marks
            out = _fetch(isic_cnf.SEARCH_URL, params=params)
            next_urls[diagnosis] = out["next"]
            lesion_image_list += out["lesion_image_list"]
        print("{:04d} images left...\r".format(n_samples), end="\r")        
        while (n_samples//100) > 0:
            n_samples -= 100
            # For each requested diagnosis, collect 100 more
            for diagnosis in diagnosis_list:
                if next_urls[diagnosis] is None:
                    continue  
                out = _fetch(next_urls[diagnosis])
                next_urls[diagnosis] = out["next"]
                lesion_image_list += out["lesion_image_list"]
            print("{:04d} images left...\r".format(n_samples), end="")        
        print(" "*100)
        print("Done!")
    return lesion_image_list

def download_image(isic_id: str) -> None:
    resp = requests.get(isic_cnf.BASE_API_URL + '/images/' + isic_id) # Send get request to api to get new download link
    resp = json.loads(resp.content) # Convert to json
    image_url = resp['files']['full']['url']
    img_bytes = requests.get(image_url)
    img_path = isic_cnf.IMAGES_DIR / (isic_id+".jpg")
    with open(img_path , "wb") as imfile:
        imfile.write(img_bytes.content)

def save_metadata(image_list: List[LesionImage]):
    df = pd.DataFrame(list(map(lambda x: x.to_dict(), image_list))) # Build dataframe from list of dicts
    df["img_id"] = df["isic_id"].apply(lambda id: id + ".jpg") # Make column for image name
    try:
        old_df = pd.read_csv(isic_cnf.METADATA, index_col=0)
        df_merge = pd.concat([old_df,df]).drop_duplicates().reset_index(drop=True)
        df_merge.to_csv(isic_cnf.METADATA)
    except FileNotFoundError:
        df.to_csv(isic_cnf.METADATA)
        return
    
if __name__ == "__main__":
    images = fetch_from_isic(800, [
        "melanoma",
        "nevus",
        '"basal cell carcinoma"',
        '"seborrheic keratosis"',
        '"actinic keratosis"',
        '"squamous cell carcinoma"'
        ])
    save_metadata(image_list=images)
    exit(0)

  