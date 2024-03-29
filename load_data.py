# -*- coding: utf-8 -*-
"""
Cropping, rescaling and saving of CBIS-DDSM images
"""

import matplotlib.pyplot as plt
from pydicom import dcmread
import os
import numpy as np
import cv2
import PIL
import pandas as pd
import h5py

from tqdm import tqdm, trange
from time import sleep

from crop_mammogram import crop_img_from_largest_connected, image_orientation

#%%

def read_csv(csv_path, sample=None):
    data = pd.read_csv(csv_path)
    
    img_paths = data[['image file path','cropped image file path','ROI mask file path']].values
    
    labels = data['pathology'].values
    
    direction = data[['left or right breast', 'image view']].values
    
    patient_ids = data['patient_id']
    
    if sample and sample>0:
        img_paths, labels, direction, patient_ids = img_paths[:sample], labels[:sample], direction[:sample], patient_ids[:sample]
                    
    return img_paths, labels, direction, patient_ids

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
        
def plot(img, save=False):
    # plot the image using matplotlib
    plt.imshow(img, cmap=plt.cm.gray)
    if save:
        plt.imsave(f"tmp/{save}.png", img, format="png", cmap=plt.cm.gray)
    plt.show()
    
def plot_multiple(images, size=None):
    size = min(len(images),10,size)
    fig, axs = plt.subplots(size, 3)
    for i in range(size):
        for j in range(3):
            axs[i,j].imshow(images[i][j], cmap=plt.cm.gray)
            

def to_image(img_array):
    return PIL.Image.fromarray(img_array)

def resize_image(img, max_size=0, height=0, width=0, scale=1, keep_ratio=True):
    h, w = img.shape
    
    if scale != 1:
        return cv2.resize(img, (0, 0), fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
                          #interpolation=cv2.INTER_CUBIC)
    elif height>0 and width>0:
        return cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
    elif max_size>0 and (h>max_size or w>max_size):
        if keep_ratio:
            if h>w:
                w = int(w/(h/max_size))
                h = max_size
            elif h==w:
                h, w = max_size, max_size
            else:
                h = int(h/(w/max_size))
                w = max_size
        else:
            h, w = max_size, max_size
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    print("Image not resized")
    return img

# Credits to Gijs van Tulder
def compute_patch_offset(centroid, patch_size, image_size):
    centroid = int(np.floor(centroid))
    offset = max(0, centroid - patch_size // 2)
    offset = min(offset, image_size - patch_size)
    return offset

def get_crops(path_list, direction):
    crops = []
    bp_list = []
    
    print("Loading images and computing crops...\n")
    sleep(0.2)
    for i in trange(len(path_list)):
        path = path_list[i][0]
        ds = get_image(os.path.join(images_path, path.strip()))
        
        pixel_array = ds.pixel_array
        
        # check which side of the image has highest values for direction
        if np.sum(pixel_array[:,0:5]) > np.sum(pixel_array[:,-5:-1]):
            bp = "L"
        else:
            bp = "R"
        bp_list.append([bp, direction[i][0], direction[i][1]])
        
        cropped = crop_img_from_largest_connected(pixel_array, image_orientation('NO', bp))
        y_min, y_max, x_min, x_max = cropped[0]
        crops.append([y_min, y_max, x_min, x_max])
    
    return np.array(crops), bp_list

def get_equal_crops(img, crop, patch_size):
    y_min, y_max, x_min, x_max = crop
    min_img_value = np.min(img)
    
    # set all values outside of the crop to minimum, so that if the crop is smaller
    # than the actual patch, outside noise is still reduced
    img[0:y_min], img[y_max:img.shape[0]], img[:,0:x_min], img[:,x_max:img.shape[1]] = (min_img_value,)*4
    
    y_centroid, x_centroid = y_min+int(np.floor(patch_size[0]/2)), x_min+int(np.floor(patch_size[1]/2))
    
    offsets = [compute_patch_offset(y_centroid, patch_size[0], img.shape[0]),
               compute_patch_offset(x_centroid, patch_size[1], img.shape[1])]
    
    if patch_size[0] > img.shape[0] or patch_size[1] > img.shape[1]:
        pad_width = ((max(0, patch_size[0] - img.shape[0]), 0),
                         (max(0, patch_size[1] - img.shape[1]), 0))
        offsets = [max(0, offsets[0]), max(0, offsets[1])]
        img = np.pad(img, pad_width, constant_values=min_img_value)
        
    patch = img[offsets[0]:offsets[0] + patch_size[0], offsets[1]:offsets[1] + patch_size[1]]
    
    return patch

def get_images_and_resize(path_list, img_scale=1, img_size=None, crops=False, percentile=90, include_files=[True,True,True], normalize=None):
    image_list = [[],[],[]]
    
    size_is_tuple, size_is_int = type(img_size) is tuple, type(img_size) is int
    
    if crops is not None:
        y_max_val = int(np.percentile([y_max-y_min for [y_min,y_max,_,_] in crops], percentile))
        x_max_val = int(np.percentile([x_max-x_min for [_,_,x_min,x_max] in crops], percentile))
    
    print("Preprocessing and saving images...\n")
    sleep(0.3)
    
    for p in tqdm(range(len(path_list))):
        for i in range(3):
            if include_files[i]:
                path = path_list[p][i]
                ds = get_image(os.path.join(images_path, path.strip()))
                
                pixel_array = ds.pixel_array.astype('float32')
                h, w = pixel_array.shape
                
                if crops is not None and (i==0 or i==1):
                    if not img_size:
                        cropped = get_equal_crops(pixel_array, crops[p], (y_max_val, x_max_val))
                    else:
                        y_min, y_max, x_min, x_max = crops[p][0]
                        cropped = pixel_array[y_min:y_max,x_min:x_max]
                    
                    #y_min, y_max, x_min, x_max = cropped
                    
                    #pixel_array = pixel_array[:,x_min:x_max]
                    pixel_array = cropped
                    #plot(pixel_array)
                
                #img = cv2.GaussianBlur(pixel_array, (0, 0), 1, 1)
                #print(pixel_array.shape)
                if img_scale != 1:
                    img = resize_image(pixel_array, scale=img_scale)
                elif size_is_tuple:
                    img = resize_image(pixel_array, width=img_size[0], height=img_size[1])
                elif size_is_int:
                    img = resize_image(pixel_array, img_size, keep_ratio=True)
                
                #plot(img)

                if normalize:
                    img -= normalize[0]
                    img /= normalize[1]
                
                image_list[i].append(img)
    
    return image_list

#%%

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', metavar='CSV', required=True,
                        #default="data/mass_case_description_train_set.csv",
                        help='CSV file you want to load (e.g. "data/mass_case_description_train_set.csv")')
    parser.add_argument('--images-path', metavar='DIR', required=True,
                        #default="D:/CBIS-DDSM",
                        help='Folder where your CBIS-DDSM files are stored (e.g. "D:/CBIS-DDSM")')
    parser.add_argument('--sample', metavar='N', type=int, default=0,
                        help='sample size, is ignored when <= 0')
    parser.add_argument('--scale', metavar='N', type=float, default=0.1,
                        help='scale used for rescaling images (default 0.1)')
    parser.add_argument('--percentile', metavar='N', type=int, default=90,
                        help='percentile used when selecting the general crop (default 90%)')
    parser.add_argument('--no-cropping', action='store_true',
                        help='do not use cropping (cropping is used by default)')
    parser.add_argument('--overwrite-crops', action='store_true',
                        help='create new crops, even if crop file already exists')
    
    args = parser.parse_args()
    vargs = vars(args)
    print(vargs)
    print()
    
    csv_path    = args.file
    images_path = args.images_path
    
    to_crop             = not args.no_cropping
    use_existing_file   = not args.overwrite_crops
    sample              = args.sample if args.sample > 0 else None
    
    simple_file_name = csv_path.split("/")[-1][:-4] # e.g. "mass_case_description_test_set"
    
    path_list, labels, direction, patient_ids = read_csv(csv_path, sample=sample)
    
    sample_name = '' if sample is None else f"{sample}_sample_"
    
    if to_crop:
        crops_folder_name = os.path.join("data/crops/")
        
        if not os.path.exists(crops_folder_name):
            os.makedirs(crops_folder_name)
        
        crop_filename = f"{crops_folder_name}{sample_name}{simple_file_name}.h5"
        
        if use_existing_file:
            try:
                with h5py.File(crop_filename, "r+") as f:
                    crops = np.array(f["/crops"])
                    bp_list = list(f["/bp"])
            except:
                use_existing_file = False
            
        if not use_existing_file:
            crops, bp_list = get_crops(path_list, direction)
            
            with h5py.File(crop_filename,'w') as f:
                crop_set = f.create_dataset(
                    "crops", crops.shape, data=crops
                )
                bp_set = f.create_dataset(
                    "bp", (len(bp_list),3), data=bp_list
                )
                patient_id_set = f.create_dataset(
                    "patient_id", (len(patient_ids),1), data=list(patient_ids)
                )
            
        
    #%%
    
    percentile  = args.percentile
    img_scale   = args.scale
    
    include_files=[True,False,False]
    normalize = (12513.3505859375, 16529.138671875) # values used to normalize images
    
    if to_crop:
        images = get_images_and_resize(path_list, img_scale=img_scale, crops=crops, percentile=percentile, include_files=include_files, normalize=normalize)
    else:
        images = get_images_and_resize(path_list, img_scale=img_scale, include_files=include_files, normalize=normalize)
    
    #plot_multiple(images[:], size=3)
    
    saved_images = np.array(images[0])
    
    #%%
    
    str_img_size = "_"
    
    if img_scale != 1:
        str_img_size += f"scaled_{img_scale}"
    if normalize:
        str_img_size += '_normalized'
    
    # Save images
    h5_folder_name = "data/"
    if not os.path.exists(h5_folder_name):
        os.makedirs(h5_folder_name)
    if to_crop:
        h5_filename = os.path.join(h5_folder_name, f"{simple_file_name}{str_img_size}_cropped.h5")
    else:
        h5_filename = os.path.join(h5_folder_name, f"{simple_file_name}{str_img_size}.h5")
    
    if not sample:
        with h5py.File(h5_filename,'w') as f:
            dataset = f.create_dataset(
                #"images", saved_images.shape, data=saved_images.astype('float16'), compression='gzip'
                "images", saved_images.shape, data=saved_images, compression='gzip'
            )
            meta_set = f.create_dataset(
                "meta", (labels.shape[0],1), data=labels.reshape(labels.shape[0],1)
            )
            bp_set = f.create_dataset(
                "bp", (len(bp_list),3), data=bp_list
            )
            patient_id_set = f.create_dataset(
                "patient_id", (len(patient_ids),1), data=list(patient_ids)
            )
