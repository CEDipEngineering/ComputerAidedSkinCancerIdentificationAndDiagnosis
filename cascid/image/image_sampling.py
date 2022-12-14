from typing import Callable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from cascid.configs import pad_ufes_cnf


DATA_DIR = pad_ufes_cnf.METADATA

IMAGE_DIR = pad_ufes_cnf.IMAGES_DIR

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

def show_sample_pictures(df: pd.DataFrame, n_samples: int = 9 , random_seed: int = None) -> plt.figure:
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
    microdf = df.sample(n_samples, random_state=random_seed).reset_index()
    imgs = microdf['img_id'].to_list()
    diganosis = microdf['diagnostic'].to_list()
    patients = microdf['patient_id'].to_list()
    fig = plt.figure(figsize=(18,3*((n_samples+6)//3)))
    axList = []
    for i in range(n_samples):
        axList.append(fig.add_subplot((2+n_samples)//3,3,i+1))
        filename = str(IMAGE_DIR) + "/" + imgs[i]
        # print(filename)
        img = cv2.imread(filename)[:,:,::-1]
        axList[i].imshow(img)
        axList[i].set_title("Patient:{0} | Diagnosis:{1}".format(patients[i], diganosis[i]))
    # fig.show()
    return fig
    
def apply_processing_show_sample_pictures(df: pd.DataFrame, func, n_samples=9):
    '''
    Function to process <n_samples> images from the dataframe with a specific
    function and show the results.
    Arguments:
        - df: dataframe containing image information
        - func: image preprocessing function
        - n_samples: number of images to process/show

    '''
    microdf = df.sample(n_samples, random_state=42).reset_index()
    imgs = microdf['img_id'].to_list()
    diganosis = microdf['diagnostic'].to_list()
    patients = microdf['patient_id'].to_list()
    results = []
    for i in range(n_samples):
        filename = str(pad_ufes_cnf.IMAGES_DIR / imgs[i])
        img = cv2.imread(filename)[:,:,::-1]
        processed = func(img)
        results.append(processed)
    return results
        
def show_histograms_by_picture(df: pd.DataFrame, n_samples: int = 3, random_seed: int = None) -> plt.figure:

    '''
    Function to show <n_samples> images from dataframe beside their color histogram.
    Dataframe must have the columns 'img_id', 'diagnostic', 'patient_id'
    'img_id' must be name of picture file inside 'data/images/' folder
    Returns a pyplot Figure with the graphed images and histograms side by side

    Examples:

    # Show 5 pictures (with color histogram) of people in the dataset with ages between 18 and 60
    show_histograms_by_picture(df[ (df['age'] > 18) & (df['age'] < 60)], n_samples=15)
    
    # Show 3 pictures of people with Melanoma, and the color histogram of said pictures
    show_histograms_by_picture(df[ (df['diagnostic'] == "MEL")], n_samples=3)
    '''
    microdf = df.sample(n_samples, random_state=random_seed).reset_index()
    imgs = microdf['img_id'].to_list()
    diganosis = microdf['diagnostic'].to_list()
    patients = microdf['patient_id'].to_list()
    fig = plt.figure(figsize=(12,6*n_samples))
    axList = [0]*n_samples*2
    for i in range(n_samples):
        axList[2*i] = fig.add_subplot(n_samples,2,2*i+1)
        axList[2*i+1] = fig.add_subplot(n_samples,2,2*i+2)
        filename = str(IMAGE_DIR) + "/" + imgs[i]
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

def image_grid(images, cols = 4):
    '''
    Function to plot multiple images from list
    Arguments:
        - images: list of images to be shown
        - cols: distribution parameter

    Example:
    image_grid(image_list, 2)
    '''
    n_images = len(images)
    fig = plt.figure()
    for n, (image) in enumerate(images):
        a = fig.add_subplot(int(cols), int(np.ceil(n_images/float(cols))), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def save_preprocessed_imgs(img_path, func, dest_dir):
    '''
    Function to save preprocessed images
    Arguments:
        - img_path: original image path
        - func: preprocessing function
        - dest_dir: directory in which images will be saved

    Example:
    # Preprocess all dataset images, removing hairs
    save_preprocessed_imgs(df, image_preprocessing.remove_hairs, DEST_DIR)
    '''
    try:
        # If preprocessed image already exists, do nothing
            img_name = Path(img_path).name
            img = cv2.imread(dest_dir+img_name)[:,:,::-1]
            print("already processed: "+dest_dir+img_name)
    except:
        # If preprocessed image does not exist, create it
            img = cv2.imread(img_path)[:,:,::-1]
            processed = func(img)
            cv2.imwrite(dest_dir, processed[:,:,::-1])
            print("saving image: "+dest_dir)

def apply_transform(df: pd.DataFrame, n_samples=3, f: Callable[[np.ndarray,], np.ndarray] = lambda x: x, random_seed: int = None) -> plt.figure:
    '''
    Function to transform <n_samples> images from dataframe. Display original image beside transformed result.
    Function f must receive and return a single image, which is a numpy ndarray, as read by opencv.
    Dataframe must have the columns 'img_id', 'diagnostic', 'patient_id'
    'img_id' must be name of picture file inside 'data/images/' folder
    Returns a pyplot Figure with the graphed images and histograms side by side

    Examples:

    # Apply histogram equalization to red channel and show result
    def clahe_transform(img: np.ndarray) -> np.ndarray:
        img_gray = img[:,:,0] # Red channel only
        img_blur = cv2.GaussianBlur(img_gray, (15,15), 0, 0) # Blur
        clahe = cv2.createCLAHE(clipLimit = 1.0, tileGridSize=(13,13)) 
        clahe_img = clahe.apply(img_blur) # Apply contrast enchance
        final_img = clahe_img
        return cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB) # return in RGB for matplotlib to show

    apply_transform(df, n_samples=9, f=clahe_transform);
    '''
    microdf = df.sample(n_samples, random_state=random_seed).reset_index()
    imgs = microdf['img_id'].to_list()
    diganosis = microdf['diagnostic'].to_list()
    patients = microdf['patient_id'].to_list()
    fig = plt.figure(figsize=(12,6*n_samples))
    axList = [0]*n_samples*2
    for i in range(n_samples):
        axList[2*i] = fig.add_subplot(n_samples,2,2*i+1)
        axList[2*i+1] = fig.add_subplot(n_samples,2,2*i+2)
        filename = str(IMAGE_DIR) + "/" + imgs[i]
        img = cv2.imread(filename)
        img_transformed = f(img)
        img = img[:,:,::-1]
        axList[2*i].imshow(img)
        axList[2*i].set_title("Patient:{0} | Diagnosis:{1}".format(patients[i], diganosis[i]))
        axList[2*i+1].imshow(img_transformed)
    return fig