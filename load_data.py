# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:37:56 2020

@author: Thijs Werrij
"""

import matplotlib.pyplot as plt
from pydicom import dcmread
import os
#import numpy as np
import cv2
import PIL

from tqdm import tqdm

#fpath = get_testdata_file()
#ds = dcmread("1-1.dcm")

def list_paths(dir, file_type=None, limit=None):
    r = []
    file_gen = os.walk(dir)
    length = len(list(file_gen))+1
    i = 1
    
    for root, dirs, files in os.walk(dir):
        if (i%1000==0):
            print(str(i) + "/" + str(length) + " files passed")
        for name in files:
            if (not file_type or name.endswith(file_type)):
                r.append(os.path.join(root, name))
                i+=1
        if limit and i>limit:
            break
    #print(str(len(r)) + " files read")
    return r

def display_data(ds, plot=True):
    # https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_read_dicom.html
    
    # Normal mode:
    print()
    print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
    print()
    
    pat_name = ds.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print(f"Patient's Name...: {display_name}")
    print(f"Patient ID.......: {ds.PatientID}")
    print(f"Modality.........: {ds.Modality}")
    print(f"Study Date.......: {ds.StudyDate}")
    print(f"Image size.......: {ds.Rows} x {ds.Columns}")
    
    # use .get() if not sure the item exists, and want a default value if missing
    print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")
    
    if (plot):
        plot(ds.pixel_array)
        
def plot(img):
    # plot the image using matplotlib
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

def get_dcm(dir, limit=None):
    return list_paths(dir, "dcm", limit)

def to_image(img_array):
    return PIL.Image.fromarray(img_array)

def resize_image(img, width, height, scale=1):
    if scale != 1:
        return cv2.resize(img, (0, 0), fx=scale, fy=scale, 
                          interpolation=cv2.INTER_CUBIC)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

def get_images_and_resize(path_list, width, height):
    image_list = []
    
    for path in tqdm(path_list):
        ds = dcmread(path)
        #img = cv2.GaussianBlur(ds.pixel_array, (0, 0), 1, 1)
        img = resize_image(ds.pixel_array, width, height)
        image_list.append(img)
    return image_list

folder_path = "CBIS-DDSM"
folder_path = "D:\CBIS-DDSM"

path_list = get_dcm(folder_path, limit=200)
images = get_images_and_resize(path_list, 180, 180)