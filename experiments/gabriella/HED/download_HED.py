from cascid.configs import hed_cnf
import os
import requests

HED_DIR = hed_cnf.HED_DIR
OUTPUT_DIR = hed_cnf.HED_RESULTS

if not os.path.exists(HED_DIR):
    os.makedirs(HED_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def download_file(URL, path):
    if not os.path.isfile(path):
        response = requests.get(URL)
        open(path, "wb").write(response.content)

print(HED_DIR)


path = str(HED_DIR / "hed_pretrained_bsds.caffemodel")
URL = "https://alinsperedu-my.sharepoint.com/:u:/g/personal/gabriellaec_al_insper_edu_br/EdKEUAiicqJCts5k6HyJFR4B9dnpMdxVzY6FQwq-1_pLtQ?e=7b08UE"
download_file(URL, path)

path = str(HED_DIR / "deploy.prototxt")
URL = "https://alinsperedu-my.sharepoint.com/:u:/g/personal/gabriellaec_al_insper_edu_br/EdQMvRXSD_RCtKYlkuoCKjEBMe3u-0XnsptpRN_2iYUYJQ?e=kVEcPB"
download_file(URL, path)