
import os
import pickle
import cv2
import argparse

parser = argparse.ArgumentParser(description='Save images to pickle')

def save_imgs(src_dir, dest_dir):
    imgs_list = []
    with open(dest_dir,"wb") as dest:
        for f in os.listdir(src_dir):
            if f.endswith(".png") or f.endswith(".jpg"):
                img = cv2.imread(src_dir+f)[:,:,::-1]
                imgs_list.append((img,f))
                
        pickle.dump(imgs_list,dest)


parser.add_argument('src',
                    help='Source path - images folder')

parser.add_argument('dst', 
                    help='Destine path - pickle file')
args = parser.parse_args()
save_imgs(args.src, args.dst)