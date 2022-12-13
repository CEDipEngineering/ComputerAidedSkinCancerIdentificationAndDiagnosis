# Computer Aided Skin Cancer Identification And Diagnosis
CASCID or Computer Aided Skin Cancer Identification and Diagnosis.

This project was developed in 2022, and was part of the required capstone project for a bachelor's degree in Computer Engineering for the four group members (CEDipEngineering, fernandocfbf, SamuelNPorto, gabriellaec).

The goal for this project is to create a system for automatic diagnosis of skin cancer using both image and clinical data regarding a patient's history. This was accomplished through use of the [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1) dataset, which contains both the necessary types of data. This dataset was not very well balanced, however, and as a means of improving classification performance, and model generalization, the team agreed to use another dataset for images only, as a means of training the image classification models. This dataset is the [ISIC](https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main), an openly available international collaboration for skin cancer image collection.

## How to install

In order to run this project's code, you will need a few things.

Requirements:
- Python 3.8 (or greater) 
- [Expo Framework](https://docs.expo.dev/get-started/installation/)

The Expo framework is only necessary to run the final prototype application, and is not necessary to train, test or analyze the performance of any developed model.

For ease of installation, a pythonenv.yaml file is included for use with conda, simply importing this environment should work. Another option, is to install the requirements using

    $ pip install -r requirements.txt

## Datasets

**There is no need to download and place any dataset manually.** This project includes scripts that automatically download and organize these files for you, in a hidden folder in your home directory. If you wish to change where these files are installed, then simply alter the variable DATA_DIR at cascid/configs/config.py, and all the files will be downloaded there (WARNING: altering this path after downloading some data will cause all downloaded data to be left stranded in the previous directory, and become inacessible by the module. This means it will need to re-download everything, and some data can be left duplicate on your disk)

## Examples

The included cascid module has a directory named examples, inside of which there are some example jupyter notebooks on how to get started using the project.