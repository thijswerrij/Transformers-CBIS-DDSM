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

#fpath = get_testdata_file()
#ds = dcmread("1-1.dcm")

csv_path = "data/"
h5_path = "data/"
images_path = "D:/CBIS-DDSM/"

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

def get_images_and_resize(path_list, width, height, plot=False):
    image_list = []
    
    for paths in tqdm(path_list):
        row = []
        for path in paths:
            #ds = get_image(images_path + path)
            ds = get_image(os.path.join(images_path, path.strip()))
            #img = cv2.GaussianBlur(ds.pixel_array, (0, 0), 1, 1)
            img = resize_image(ds.pixel_array, width, height)
            row.append(img)
        image_list.append(row)
        
    image_list = np.asarray(image_list)
    
    return image_list

file_name = "calc_case_description_test_set"

path_list, labels = read_csv(f"{file_name}.csv")#, sample=50)
images = get_images_and_resize(path_list, 180, 180, True)

plot_multiple(images[:], size=3)


# Save images
h5out = h5py.File(f"{h5_path}{file_name}.h5", 'w')
saved_images = images[:,0]

dataset = h5out.create_dataset(
    "images", np.shape(saved_images), h5py.h5t.STD_U8BE, data=saved_images
)
meta_set = h5out.create_dataset(
    "meta", (labels.shape[0],1), data=labels.reshape(labels.shape[0],1)
)
h5out.close()
