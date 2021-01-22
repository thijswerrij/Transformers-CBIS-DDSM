# -*- coding: utf-8 -*-
"""
Custom version of ResViT to train on CBIS-DDSM
"""
import PIL
import time
import torch
import torchvision

from load_data import plot

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
sys.path.append('../VisualTransformers')
from ResViT import ViTResNet, BasicBlock, train, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

#%% Load hdf5

import h5py
import numpy as np

h5_path = "data/"

def read_hdf5(file_name):
    file = h5py.File(f"{h5_path}{file_name}.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("str")

    return images, labels

#%% Custom dataset (CBIS-DDSM)

from torch.utils.data import Dataset, DataLoader

label_to_int = { 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 1, 'MALIGNANT' : 2 }
label_to_bin = { 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT' : 1 }

class CBISDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_name, transform=None, batch_size=None, binary=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images, labels = read_hdf5(file_name)
        
        if batch_size:
            data_size = int(len(labels)/batch_size)*batch_size
            self.images, labels = self.images[:data_size], labels[:data_size]
        
        #(self.images.shape, labels.shape)
        if binary:
            self.labels = np.array([label_to_bin[i[0]] for i in labels]).astype('float')
        else:
            self.labels = np.array([label_to_int[i[0]] for i in labels]).astype('float')
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i].astype('float')
        label = self.labels[i].astype('float')#np.array([self.labels[i]])
        
        if self.transform:
            image = self.transform(PIL.Image.fromarray(image))
            #image = PIL.Image.totensor(image)
        
        sample = (image, label)

        return sample


transform = torchvision.transforms.Compose(
     [torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
     torchvision.transforms.RandomAffine(8, translate=(.15,.15)),
     torchvision.transforms.ToTensor(),
     #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     torchvision.transforms.Normalize((0.5), (0.5))
     ])

#%%

BATCH_SIZE_TRAIN = 50
BATCH_SIZE_TEST = 50
batch_size = (BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

train_dataset = CBISDataset("calc_case_description_train_set_180", transform, BATCH_SIZE_TRAIN)
test_dataset = CBISDataset("calc_case_description_test_set_180", transform, BATCH_SIZE_TEST)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

N_EPOCHS = 10#50

model = ViTResNet(BasicBlock, [3, 3, 3], in_channels=1, num_classes=3, batch_size=batch_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9,weight_decay=1e-4)
#lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[35,48],gamma = 0.1)

init_time = time.time()
train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    start_time = time.time()
    train(model, optimizer, train_loader, train_loss_history)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    evaluate(model, test_loader, test_loss_history)
    
minutes, seconds = divmod(time.time() - init_time, 60)
print('Total execution time:', '{:.0f}m {:.2f}s'.format(minutes, seconds))

PATH = ".\ViTRes.pt" # Use your own path
torch.save(model.state_dict(), PATH)


# =============================================================================
# model = ViT()
# model.load_state_dict(torch.load(PATH))
# model.eval()            
# =============================================================================
