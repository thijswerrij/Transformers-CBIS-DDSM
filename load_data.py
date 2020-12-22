# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:37:56 2020

@author: Thijs Werrij
"""

import matplotlib.pyplot as plt
from pydicom import dcmread
import os

#fpath = get_testdata_file()
#ds = dcmread("1-1.dcm")

def list_paths(dir, file_type=None):
    r = []
    file_gen = os.walk(dir)
    length = len(list(file_gen))+1
    i = 0
    
    for root, dirs, files in os.walk(dir):
        i+=1
        if (i%1000==0):
            print(str(i) + "/" + str(length) + " files passed")
        for name in files:
            if (not file_type or name[-3:] == file_type):
                r.append(os.path.join(root, name))
    return r

def display_data(ds):
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
    
    # plot the image using matplotlib
    plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
    plt.show()

def get_dcm(dir):
    return list_paths(dir, "dcm")

folder_path = "CBIS-DDSM"
#folder_path = "D:\CBIS-DDSM"
path_list = get_dcm(folder_path)
image_list = []

i = 0
length = len(path_list)
for path in path_list:
    i+=1
    ds = dcmread(path)
    #display_data(ds)
    image_list.append(ds)
    if (i%100==0):
        print(str(i) + "/" + str(length) + " files added")