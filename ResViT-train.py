# -*- coding: utf-8 -*-
"""
Custom version of ResViT to train on CBIS-DDSM
"""
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F

from load_data import plot

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
sys.path.append('../VisualTransformers')
from ResViT import ViTResNet, BasicBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

#%% Load hdf5

import h5py
import numpy as np

h5_path = "data/"

def read_hdf5(file_name):
    file = h5py.File(f"{h5_path}{file_name}.h5", "r+")

    images = np.array(file["/images"]).astype("uint16")
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
        
        if binary:
            self.labels = np.array([label_to_bin[i[0]] for i in labels])
        else:
            self.labels = np.array([label_to_int[i[0]] for i in labels])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i].astype('float')
        label = np.array(self.labels[i])
        
        if self.transform:
            image = self.transform(PIL.Image.fromarray(image))
        else:
            image = torchvision.transforms.ToTensor()(PIL.Image.fromarray(image))
        
        sample = (image, label)

        return sample


transform = torchvision.transforms.Compose([
     #torchvision.transforms.RandomHorizontalFlip(),
     #torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
     #torchvision.transforms.RandomAffine(8, translate=(.15,.15)),
     torchvision.transforms.ToTensor(),
     #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     torchvision.transforms.Normalize((0.5), (0.5))
     ])

#%%

def train(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device, dtype=torch.int64)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
            
def evaluate(model, data_loader, loss_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

#%%

BATCH_SIZE_TRAIN = 50
BATCH_SIZE_TEST = 50
batch_size = (BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)
is_binary = True

file_params = "180_cropped"
train_dataset = CBISDataset(f"calc_case_description_train_set_{file_params}", transform, BATCH_SIZE_TRAIN, is_binary)
test_dataset = CBISDataset(f"calc_case_description_test_set_{file_params}", transform, BATCH_SIZE_TEST, is_binary)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

N_EPOCHS = 5

model = ViTResNet(BasicBlock, [3, 3, 3], in_channels=1, num_classes=2 if is_binary else 3, batch_size=batch_size).to(device)
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

PATH = "./ViTRes.pt" # Use your own path
torch.save(model.state_dict(), PATH)


# =============================================================================
# model = ViT()
# model.load_state_dict(torch.load(PATH))
# model.eval()            
# =============================================================================
