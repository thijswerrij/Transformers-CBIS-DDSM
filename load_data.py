# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:37:56 2020

@author: Thijs Werrij
"""

import matplotlib.pyplot as plt
from pydicom import dcmread
import os
import numpy as np
import cv2
import PIL
import pandas as pd
import csv
import h5py

from tqdm import tqdm

import sys
sys.path.append('../breast_cancer_classifier')
from src.cropping.crop_mammogram import crop_img_from_largest_connected, image_orientation

#fpath = get_testdata_file()
#ds = dcmread("1-1.dcm")

csv_path = "data/"
h5_path = "data/"
images_path = "D:/CBIS-DDSM/"

#%%

def read_csv(path, sample=None):
    data = pd.read_csv(csv_path + path)
    
    img_paths = data[['image file path','cropped image file path','ROI mask file path']].values
    
    labels = data['pathology'].values
    
    if sample and sample>0:
        img_paths, labels = img_paths[:sample], labels[:sample]
                    
    return img_paths, labels

def get_image(path):
    return dcmread(path)

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
    
def plot_multiple(images, size=None):
    size = min(len(images),10,size)
    fig, axs = plt.subplots(size, 3)
    for i in range(size):
        for j in range(3):
            axs[i,j].imshow(images[i][j], cmap=plt.cm.gray)
            

def to_image(img_array):
    return PIL.Image.fromarray(img_array)

def resize_image(img, width, height, scale=1):
    if scale != 1:
        return cv2.resize(img, (0, 0), fx=scale, fy=scale, 
                          interpolation=cv2.INTER_CUBIC)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

def get_direction(ds):
    try:
        if ds.Laterality in ["L","R"]:
            return ds.Laterality
    except:
        if ds.BodyPartExamined[0] in ["L","R"]:
            return ds.BodyPartExamined[0]
    return None

def get_images_and_resize(path_list, width, height, crop=False, show_plot=False, include_files=[True,True,True]):
    image_list = []
    
    for paths in tqdm(path_list):
        row = []
        bp = None
        
        for i in range(3):
            if include_files[i]:
                path = paths[i]
                ds = get_image(os.path.join(images_path, path.strip()))
                
                pixel_array = ds.pixel_array
                
                if (i == 0):
                    bp = get_direction(ds)
                
                if crop:
                    cropped = crop_img_from_largest_connected(pixel_array, image_orientation('NO', bp))
                    y_min, y_max, x_min, x_max = cropped[0]
                    pixel_array = pixel_array[y_min:y_max,x_min:x_max]
                
                #img = cv2.GaussianBlur(pixel_array, (0, 0), 1, 1)
                img = resize_image(pixel_array, width, height)
                
                row.append(img)
        image_list.append(row)
        
    image_list = np.asarray(image_list)
    
    return image_list


if __name__ == "__main__":
    to_crop = True
    file_name = "calc_case_description_test_set"
    img_size = 180
    
    #path_list, labels = read_csv(f"{file_name}.csv", sample=100)
    path_list, labels = read_csv(f"{file_name}.csv")
    images = get_images_and_resize(path_list, img_size, img_size, crop=True, show_plot=True, include_files=[True,False,False])
    
    #plot_multiple(images[:], size=3)
    
    
    # Save images
    if to_crop:
        h5out = h5py.File(f"{h5_path}{file_name}_{img_size}_cropped.h5", 'w')
    else:
        h5out = h5py.File(f"{h5_path}{file_name}_{img_size}.h5", 'w')
    
    saved_images = images[:,0]
    
    dataset = h5out.create_dataset(
        "images", np.shape(saved_images), data=saved_images
    )
    meta_set = h5out.create_dataset(
        "meta", (labels.shape[0],1), data=labels.reshape(labels.shape[0],1)
    )
    h5out.close()
