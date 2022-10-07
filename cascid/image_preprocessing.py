import cv2
import numpy as np


def remove_hairs(img):
    '''
    Function to remove black hairs from image
    Argument: 
        - Original image
    Returns:
        - Processed image without hairs

    Example:
    dst = remove_hairs(img)
    '''
    kernel = cv2.getStructuringElement(1,(17,17))
    gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    edges = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    ret,thresh = cv2.threshold(edges,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(img,thresh,1,cv2.INPAINT_TELEA)
    
    return dst


def adaptive_histeq(img_gray):
    '''
    Function to apply adaptive histogram equalization (CLAHE)
    Argument: 
        - Original image (grayscale)
    Returns:
        - Processed image without hairs

    Example:
    adaptive_histeq = img_gray(img)
    '''
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
    final_img = clahe.apply(img_gray) 
    return final_img

def simple_processing_clahe(img):
    
    gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    clahe=adaptive_histeq(gray)
    return clahe



def lesion_segmentation(img):
    
    '''
    Function used for lesion segmentation
    Argument: 
        - Original image 
    Returns:
        - Threshold image indicating the region containing the skin lesion

    Example:
    processed_img = lesion_segmentation(img)
    '''
    
    R, G, B = cv2.split(img)
    norm_img = np.zeros((800,800))
    r_norm = cv2.normalize(R,  norm_img, 0, 255, cv2.NORM_MINMAX)

    blur = cv2.GaussianBlur(r_norm,(13,13),0)
    imgeq=adaptive_histeq(blur)
    
    T,imgt = cv2.threshold(imgeq,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    final_img = cv2.bitwise_not(imgt)
    
    kernel = np.ones((7,7),np.uint8)
    closing = cv2.morphologyEx(final_img, cv2.MORPH_CLOSE, kernel)
    
    
    return closing



def preprocessing_article(img):
    
    '''
    Function used for image preprocessing based on article:
        Melanoma Skin Cancer Detection Method Based on Adaptive
        Principal Curvature, Colour Normalisation and Feature Extraction
        with the ABCD Rule
    Available at:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7256173/pdf/10278_2019_Article_316.pdf
    Argument: 
        - Original image 
    Returns:
        - Preprocessed image

    Example:
    processed_img = preprocessing_article(img)
    '''
    R, G, B = cv2.split(img)
    norm_img = np.zeros((800,800))
    r_norm = cv2.normalize(R,  norm_img, 0, 255, cv2.NORM_MINMAX)

    return r_norm

def preprocessing_article_histeq(img):
  
    r_norm = preprocessing_article(img)
    blur = cv2.GaussianBlur(r_norm,(7,7),0)
    imgeq=adaptive_histeq(blur)
    
    return imgeq
    


def enhance_contrast_ab(img, alpha_value, beta_value):
    '''
    Function to enhance contrast and brightness
    Arguments: 
        - img: Original image 
        - alpha_value: contrast parameter
        - beta_value: brightness parameter
    Returns:
        - Preprocessed image

    Example:
    processed_img = enhance_contrast_ab(img, 1.4, 40)
    '''

    blur = cv2.GaussianBlur(img, (13, 13), 0)

    enhanced_img = cv2.convertScaleAbs(blur, alpha=alpha_value, beta=beta_value)
    return enhanced_img


def preprocessing_lab_histeq(img):
    '''
    Function to enhance contrast based on L channel manipulation from LAB image
    Arguments: 
        - img: Original image 
    Returns:
        - Preprocessed image (enhanced contrast)

    Example:
    processed_img = preprocessing_lab_histeq(img)
    '''
    blur = cv2.GaussianBlur(img, (7, 7), 0)

    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    cl = adaptive_histeq(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img

def preprocessing_lab_histeq_grey(img):
    processed = preprocessing_lab_histeq(img)
    gray = cv2.cvtColor( processed, cv2.COLOR_RGB2GRAY )

    return gray
