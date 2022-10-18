
import os
import pickle
import cv2
import argparse

parser = argparse.ArgumentParser(description='Save images to pickle')

def load_images(pickle_file):
        with open(pickle_file,"rb") as f:
            recovered_list = pickle.load(f)        
        return recovered_list
        
def create_img_folder(recovered_list, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for img, name in recovered_list:
        cv2.imwrite(dest_path+name, img)



parser.add_argument('pickle',
                    help='Pickle path - pkl file')

parser.add_argument('dst', 
                    help='Destine path - images folder')

args = parser.parse_args()


create_img_folder(load_images(args.pickle), args.dst)