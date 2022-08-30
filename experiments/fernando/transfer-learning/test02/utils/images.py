from os import listdir
import cv2 as cv

def get_file_names_in_path(path):
    return [file for file in listdir(path)]

def read_image_file(file_path):
    image = cv.imread(file_path)
    image_resized = cv.resize(image, (64, 64))
    image_normalized = image_resized / 255
    return image_normalized

def read_all_images_in_path(path):
    files_to_read = get_file_names_in_path(path)
    read_files = list()
    for file in files_to_read:
        dir_to_file = "{0}/{1}".format(path, file) 
        image_array = read_image_file(dir_to_file)
        read_files.append(image_array)
    return read_files, files_to_read

def load_images(path):
    images_01, file_names = read_all_images_in_path(path+"imgs_part_1")
    images_02, file_names = read_all_images_in_path(path+"imgs_part_2")
    images_03, file_names = read_all_images_in_path(path+"imgs_part_3")
    all_images = images_01 + images_02 + images_03
    all_file_names = file_names + file_names + file_names
    return all_images, all_file_names