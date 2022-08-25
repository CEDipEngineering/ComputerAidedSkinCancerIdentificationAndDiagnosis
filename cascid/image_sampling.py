import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

DATA_DIR = Path(__file__).parents[1]/'data'

IMAGE_DIR = DATA_DIR / "images"

def plot_age_distribution(df: pd.DataFrame) -> plt.figure:
    '''
    Function to show the age distribution of a particular dataframe.
    Dataframe must have the columns 'age', 'gender'
    'gender' must be MALE or FEMALE
    Returns a pyplot Figure with the graphed histogram of age

    Example:

    # Show ages of Basal Cell Carcinoma victims
    plot_age_distribution(df[df['diagnosis'] == 'BCC'])
    '''
    male_age = df[df['gender']=='MALE']['age']
    female_age = df[df['gender']=='FEMALE']['age']

    colors=['blue', 'red']
    labels=['Male', 'Female']

    fig = plt.figure(figsize=(18,10))
    ax = fig.add_subplot(1,1,1)
    ax.hist(x=[male_age, female_age], bins=max(male_age.unique().size, female_age.unique().size), color=colors, label=labels, density=True)
    ax.legend()
    ax.set_title('Age Distribution')
    # fig.show()
    return fig

def show_sample_pictures(df: pd.DataFrame, n_samples=9) -> plt.figure:
    '''
    Function to show <n_samples> images from the dataframe.
    Dataframe must have the columns 'img_id', 'diagnostic', 'patient_id'
    'img_id' must be name of picture file inside 'data/iamges/' folder
    Returns a pyplot Figure with the <n_samples> images.

    Examples:

    # Show 15 pictures of people in the dataset with ages between 18 and 60
    show_sample_pictures(df[ (df['age'] > 18) & (df['age'] < 60)], n_samples=15)
    
    # Show 3 pictures of people with Melanoma
    show_sample_pictures(df[ (df['diagnostic'] == "MEL")], n_samples=3)
    '''
    microdf = df.sample(n_samples).reset_index()
    imgs = microdf['img_id'].to_list()
    diganosis = microdf['diagnostic'].to_list()
    patients = microdf['patient_id'].to_list()
    fig = plt.figure(figsize=(18,3*((n_samples+6)//3)))
    axList = []
    for i in range(n_samples):
        axList.append(fig.add_subplot((2+n_samples)//3,3,i+1))
        filename = IMAGE_DIR.stem + "/" + imgs[i]
        print(filename)
        img = cv2.imread(filename)[:,:,::-1]
        axList[i].imshow(img)
        axList[i].set_title("Patient:{0} | Diagnosis:{1}".format(patients[i], diganosis[i]))
    # fig.show()
    return fig

def show_histograms_by_picture(df: pd.DataFrame, n_samples=3) -> plt.figure:
    '''
    Function to show <n_samples> images from dataframe beside their color histogram.
    Dataframe must have the columns 'img_id', 'diagnostic', 'patient_id'
    'img_id' must be name of picture file inside 'data/iamges/' folder
    Returns a pyplot Figure with the graphed images and histograms side by side

    Examples:

    # Show 5 pictures (with color histogram) of people in the dataset with ages between 18 and 60
    show_histograms_by_picture(df[ (df['age'] > 18) & (df['age'] < 60)], n_samples=15)
    
    # Show 3 pictures of people with Melanoma, and the color histogram of said pictures
    show_histograms_by_picture(df[ (df['diagnostic'] == "MEL")], n_samples=3)
    '''
    microdf = df.sample(n_samples).reset_index()
    imgs = microdf['img_id'].to_list()
    diganosis = microdf['diagnostic'].to_list()
    patients = microdf['patient_id'].to_list()
    fig = plt.figure(figsize=(12,6*n_samples))
    axList = [0]*n_samples*2
    for i in range(n_samples):
        axList[2*i] = fig.add_subplot(n_samples,2,2*i+1)
        axList[2*i+1] = fig.add_subplot(n_samples,2,2*i+2)
        filename = IMAGE_DIR.stem + "/" + imgs[i]
        img = cv2.imread(filename)[:,:,::-1]
        axList[2*i].imshow(img)
        axList[2*i].set_title("Patient:{0} | Diagnosis:{1}".format(patients[i], diganosis[i]))
        hist1 = cv2.calcHist([img],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([img],[1],None,[256],[0,256])
        hist3 = cv2.calcHist([img],[2],None,[256],[0,256])
        axList[2*i+1].plot(hist1, label='red', color='red')
        axList[2*i+1].plot(hist2, label='green', color='green')
        axList[2*i+1].plot(hist3, label='blue', color='blue')
        axList[2*i+1].legend()
    return fig