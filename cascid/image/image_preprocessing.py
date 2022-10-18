import cv2
import numpy as np
from scipy.ndimage.filters import median_filter
from sklearn.cluster import MiniBatchKMeans
from skimage import feature
from scipy.signal import convolve2d, wiener



# --------------------------------- #
# ---------- Hair Removal --------- #
# --------------------------------- #
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


def remove_hairs_lab_lchannel(img):
    '''
    Function to remove black hairs from image applied to the L channel (LAB colorspace)
    Argument: 
        - Original image
    Returns:
        - Processed image without hairs

    Example:
    dst = remove_hairs_lab_lchannel(img)
    '''
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    kernel = cv2.getStructuringElement(1,(17,17))
    edges = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel)
    ret,thresh = cv2.threshold(edges,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(img,thresh,1,cv2.INPAINT_TELEA)
    
    return dst


def remove_black_hairs_hessian(img):
    '''
    Function to remove black hairs from image based on hessian
    Argument: 
        - Original image
    Returns:
        - Processed image without hairs

    Example:
    dst = remove_black_hairs_hessian(img)
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H_elems = feature.hessian_matrix(gray, sigma=1, order='rc')
    result = feature.hessian_matrix_eigvals(H_elems)
    
    img_paper_black_hair_t= np.float32(result[0] > 0.02)

    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img_paper_black_hair_t.astype('uint8'),kernel,iterations = 1)
    
    dst_black = cv2.inpaint(img,dilation.astype('uint8'),2,cv2.INPAINT_TELEA)
    
    return dst_black

def remove_white_hairs_hessian(img):
    '''
    Function to remove white hairs from image based on hessian
    Argument: 
        - Original image
    Returns:
        - Processed image without hairs

    Example:
    dst = remove_white_hairs_hessian(img)
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H_elems = feature.hessian_matrix(gray, sigma=1, order='rc')
    result = feature.hessian_matrix_eigvals(H_elems)
    
    img_paper_white_hair_t = np.float32(result[1] < -0.02)

    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img_paper_white_hair_t.astype('uint8'),kernel,iterations = 1)
    
    dst_white = cv2.inpaint(img,dilation.astype('uint8'),2,cv2.INPAINT_TELEA)
    
    return dst_white


def remove_hairs_lab_gaussian(img):
    '''
    Function to remove white hairs from image based on hessian, applied to LAB colorspace
    Argument: 
        - Original image
    Returns:
        - Processed image without hairs

    Example:
    dst = remove_hairs_lab_gaussian(img)
    '''
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    H_elems = feature.hessian_matrix(l_channel, sigma=1, order='rc')
    result = feature.hessian_matrix_eigvals(H_elems)
    
    img_paper_black_hair_t= np.float32(result[0] > 0.02)
    img_paper_white_hair_t = np.float32(result[1] < -0.02)

    dst_black = cv2.inpaint(img,img_paper_black_hair_t.astype('uint8'),2,cv2.INPAINT_TELEA)
    dst_white = cv2.inpaint(dst_black,img_paper_white_hair_t.astype('uint8'),2,cv2.INPAINT_TELEA)
    
    return dst_white


def remove_hairs_dog(img):
        '''
        Function to remove hairs from image based on Derivative of Gaussian (DOG) method
        Argument: 
            - Original image
        Returns:
            - Processed image without hairs

        Example:
        dst = remove_hairs_dog(img)


        References:
        Hair removal methods: A comparative study for dermoscopy images: https://faculty.uca.edu/ecelebi/documents/BSPC_2011.pdf
        https://notebook.community/darshanbagul/ComputerVision/EdgeDetection-ZeroCrossings/EdgeDetectionByZeroCrossings
        '''
        DoG_kernel = [
                    [0,   0, -1, -1, -1, 0, 0],
                    [0,  -2, -3, -3, -3,-2, 0],
                    [-1, -3,  5,  5,  5,-3,-1],
                    [-1, -3,  5, 16,  5,-3,-1],
                    [-1, -3,  5,  5,  5,-3,-1],
                    [0,  -2, -3, -3, -3,-2, 0],
                    [0,   0, -1, -1, -1, 0, 0]
                ] 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dog_img = convolve2d(gray, DoG_kernel, mode="same")

        dog_img[dog_img >= 255] = 0

        bitwiseNot = cv2.bitwise_not(dog_img)

        mask = np.zeros(bitwiseNot.shape, np.uint8)
        mask[bitwiseNot > 250] = 255
        
        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(mask.astype('uint8'),kernel,iterations = 1)

        final_img = cv2.inpaint(img,dilation,1,cv2.INPAINT_NS)

        closing = cv2.morphologyEx(final_img, cv2.MORPH_CLOSE, kernel)

        return closing



# ----------------------------------------- #
# ---------- Image Classification --------- #
# ----------------------------------------- #

def calculate_hessian(gray):
    H_elems = feature.hessian_matrix(gray, sigma=1, order='rc')
    result = feature.hessian_matrix_eigvals(H_elems)

    img_paper_white_hair_t = np.float32(result[1] < -0.02)

    img_paper_black_hair_t= np.float32(result[0] > 0.02)

    return img_paper_white_hair_t,img_paper_black_hair_t


def adaptive_hair_removal(img):

    '''
    Function to classify the image type according to hair color and apply the corresponding hair removal methodology
    Argument: 
        - Original image
    Returns:
        - Processed image without hairs

    Example:
    dst = adaptive_hair_removal(img)
    '''
    
    COLOR_THRESH = 15
    
# Hessian Calculation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_paper_white_hair_t,img_paper_black_hair_t = calculate_hessian(gray)

## Black Hairs morph
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    kernel = cv2.getStructuringElement(1,(17,17))
    edges = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel)
    _,thresh = cv2.threshold(edges,10,255,cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img_paper_white_hair_t.astype('uint8'),kernel,iterations = 1)
    
## Metrics
    bp_img_original = np.sum(img[img <= COLOR_THRESH])
    bp_gray_original = np.sum(gray[gray <= COLOR_THRESH])

# Selecting first inpainting mask 
    dst = cv2.inpaint(img,dilation.astype(np.uint8),1,cv2.INPAINT_TELEA)
    
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    
## Metrics after inpainting
    bp_dst = np.sum(dst[dst <= COLOR_THRESH]) 
    bp_gray_dst = np.sum(dst_gray[dst_gray <= COLOR_THRESH])

## Decision
    ratio_gray = ratio_color = 0
    if bp_gray_original > 0:
        ratio_gray = bp_gray_dst/bp_gray_original
    if bp_img_original > 0:
        ratio_color = bp_dst/bp_img_original
        
    ## Apply black hair removal
    if ratio_gray >= 1.5 or ratio_color >= 1.5:
        dst = cv2.inpaint(img,thresh.astype(np.uint8),1,cv2.INPAINT_TELEA)

    return dst


def adaptive_hair_removal2(img):
    
    UPPER_COLOR_THRESH = 50
    LOWER_LIMIT = 1.375
    
    LOWER_COLOR_THRESH = 22
    UPPER_LIMIT = 1.5
    
    # Hessian Calculation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_paper_white_hair_t,img_paper_black_hair_t = calculate_hessian(gray)

## Black Hairs morph
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    kernel = cv2.getStructuringElement(1,(17,17))
    edges = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel)
    _,thresh = cv2.threshold(edges,10,255,cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img_paper_white_hair_t.astype('uint8'),kernel,iterations = 1)
    
## Metrics
    bp_img_original = np.sum(img[img <= LOWER_COLOR_THRESH])
    bp_gray_original = np.sum(gray[gray <= LOWER_COLOR_THRESH])
    
    bp_img_original_upper = np.sum(img[img <= UPPER_COLOR_THRESH])
    bp_gray_original_upper = np.sum(gray[gray <= UPPER_COLOR_THRESH])

# First inpainting  
    dst = cv2.inpaint(img,dilation.astype(np.uint8),1,cv2.INPAINT_TELEA)    
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    
## Metrics
    bp_dst = np.sum(dst[dst <= LOWER_COLOR_THRESH]) 
    bp_gray_dst = np.sum(dst_gray[dst_gray <= LOWER_COLOR_THRESH])
    
    bp_dst_upper = np.sum(dst[dst <= UPPER_COLOR_THRESH]) 
    bp_gray_dst_upper = np.sum(dst_gray[dst_gray <= UPPER_COLOR_THRESH])

## Decision
    ratio_gray = ratio_color = ratio_color_upper = ratio_gray_upper = 0
    if bp_gray_original > 0:
        ratio_gray = bp_gray_dst/bp_gray_original
    if bp_img_original > 0:
        ratio_color = bp_dst/bp_img_original
        
    if bp_img_original_upper > 0:
        ratio_gray_upper = bp_dst_upper/bp_img_original_upper
    if bp_gray_original_upper > 0:
        ratio_color_upper = bp_gray_dst_upper/bp_gray_original_upper
        
    ## Apply black hair removal
    if (ratio_gray >= UPPER_LIMIT or ratio_color >= UPPER_LIMIT) or (ratio_gray_upper >= LOWER_LIMIT or ratio_color_upper >= LOWER_LIMIT):
        dst = cv2.inpaint(img,thresh.astype(np.uint8),1,cv2.INPAINT_TELEA)

    return dst



# ----------------------------------------- #
# ---------- Contrast Enhancement --------- #
# ----------------------------------------- #
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
    

def enhance_contrast_ab(img, alpha_value=1.4, beta_value=5):
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


def unsharp_masking(img):
    '''
    Function to enhance contrast based on Laplacian filter
    Arguments: 
        - img: Original image 
    Returns:
        - Preprocessed image (sharpened)
    
    Reference: https://www.idtools.com.au/unsharp-masking-with-python-and-opencv/
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mf = median_filter(gray, 1)
    lap = cv2.Laplacian(gray_mf,cv2.CV_64F)
    sharp = gray - 0.7*lap
    return sharp

def red_band_unsharp(img):
    '''
    Function to enhance contrast based on Laplacian filter 
    on the normalized red band
    Arguments: 
        - img: Original image 
    Returns:
        - Preprocessed image
    '''
    r_norm = preprocessing_article(img)
    gray_mf = median_filter(r_norm, 1)
    lap = cv2.Laplacian(gray_mf,cv2.CV_64F)
    sharp = r_norm - 0.7*lap
    return sharp


def color_quantization(img, k= 20):
        '''
        Function to enhance contrast based on a color quantization tecnique
        on the normalized red band
        Arguments: 
            - img: Original image 
        Returns:
            - Preprocessed image

        Reference: https://pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
        '''
        (h, w) = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        clt = MiniBatchKMeans(n_clusters = k)
        labels = clt.fit_predict(img)
        quant = clt.cluster_centers_.astype("uint8")[labels]
        quant = quant.reshape((h, w, 3))
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        return quant


# ---------------------------------------- #
# ---------- Lesion Segmentation --------- #
# ---------------------------------------- #
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


def auto_canny(image, sigma=0.33):
    '''
    Function to detect edges
    Argument: 
        - Original image 
        - sigma parameter
    Returns:
        - Threshold image indicating the region containing the skin lesion

    Example:
    processed_img = auto_canny(img)
    '''
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


# ---------------------------------- #
# ---------- Noise Removal --------- #
# ---------------------------------- #
def remove_noise(img):
    '''
    Function to remove noise from image
    Argument: 
        - Original image 
    Returns:
        - Noiseless image

    Example:
    processed_img = remove_noise(img)
    '''
    return cv2.fastNlMeansDenoising(img, None, 5, 7, 21) 

def remove_noise_colored(dst):
    return cv2.fastNlMeansDenoisingColored(dst, None, 5, 5, 7, 21) 

def wiener_filter(gray,k=3):
    '''
    Function which uses the Wiener Filter to remove noise from image
    Argument: 
        - Original image 
        - kernel size
    Returns:
        - Noiseless image

    Example:
    processed_img = wiener_filter(img)
    '''
    kernel = (k,k)
    filtered_img = wiener(gray, kernel) 
    return filtered_img