#!/usr/bin/env python3

import sys
import os
import cv2
import datetime
import argparse

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--save_folder", type=str, default="data/samples", help="path to dataset")
    
    opt = parser.parse_args()
    allowed_width = 256
    allowed_height = 192

    image_folder = opt.image_folder
    save_folder = opt.save_folder

    
    for img_file in os.listdir(image_folder):
        
        filename, ext = os.path.splitext(img_file)
        if ext not in [".jpg"]:
            continue


        img = cv2.imread(os.path.join(image_folder,img_file))
        h, w, c = img.shape

        resized = cv2.resize(img, (allowed_width,allowed_height), interpolation = cv2.INTER_AREA)
        print(f"Resized:{img_file} from {h,w,c} to {resized.shape}")
        cv2.imwrite(os.path.join(save_folder,img_file),resized)
        
    print("Completed")



