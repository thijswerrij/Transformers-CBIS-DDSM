# -*- coding: utf-8 -*-
"""
Custom version of ResViT to train on CBIS-DDSM
"""
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

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

    def __init__(self, file_name, transform=None, batch_size=None, binary=False, sample=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images, labels = read_hdf5(file_name)
        
        if sample:
            self.images, labels = self.images[:sample], labels[:sample]
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

def train(model, optimizer, data_loader, loss_history, acc_history):
    total_samples = len(data_loader.dataset)
    model.train()
    correct_samples = 0
    total_loss = 0

    for i, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device, dtype=torch.int64)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = torch.max(output, dim=1)
        correct_samples += pred.eq(target).sum().item()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
        
    avg_loss = total_loss / total_samples
    accuracy = correct_samples / total_samples
    loss_history.append(avg_loss)
    acc_history.append(accuracy)
    return output
            
def evaluate(model, data_loader, loss_history, acc_history, conf_matrices, binary=False):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0
    conf_mat = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum().item()
            
            if not binary:
                conf_mat = np.add(conf_mat, confusion_matrix(target.cpu(),pred.cpu(), labels=[0,1,2]))
            else:
                conf_mat = np.add(conf_mat, confusion_matrix(target.cpu(),pred.cpu(), labels=[0,1]))

    avg_loss = total_loss / total_samples
    accuracy = correct_samples / total_samples
    loss_history.append(avg_loss)
    acc_history.append(accuracy)
    conf_matrices.append(conf_mat)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * accuracy) + '%)\n')
    return output

#%%

if __name__ == "__main__":

    batch_size = (10, 10)
    is_binary = False
    
    #file_params = "180_cropped"
    #file_params = "400x1000_cropped"
    file_params = "scaled_0.1_cropped"
    train_dataset = CBISDataset(f"calc_case_description_train_set_{file_params}", transform, batch_size[0], is_binary)
    test_dataset = CBISDataset(f"calc_case_description_test_set_{file_params}", transform, batch_size[1], is_binary)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size[1], shuffle=False)
    
    N_EPOCHS = 5
    
    model = ViTResNet(BasicBlock, [3, 3, 3], in_channels=1, num_classes=2 if is_binary else 3, batch_size=batch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9,weight_decay=1e-4)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[35,48],gamma = 0.1)
    
    init_time = time.time()
    train_loss_history, test_loss_history = [], []
    train_acc_history, test_acc_history = [], []
    conf_matrices = []
    predictions = []
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        start_time = time.time()
        train_predict = train(model, optimizer, train_loader, train_loss_history, train_acc_history)
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        eval_predict = evaluate(model, test_loader, test_loss_history, test_acc_history, conf_matrices, is_binary)
        predictions.append([train_predict, eval_predict])
        
    minutes, seconds = divmod(time.time() - init_time, 60)
    print('Total execution time:', '{:.0f}m {:.2f}s'.format(minutes, seconds))
    
    PATH = "./ViTRes.pt" # Use your own path
    torch.save(model.state_dict(), PATH)
    
#%%

import matplotlib.pyplot as plt

#avg_train_loss_history = np.mean(np.array(train_loss_history).reshape(N_EPOCHS,-1), axis=1)

#plt.gca().set_ylim([0,None])

plt.figure()
plt.plot(train_loss_history, label="train")
plt.plot(test_loss_history, 'r', label="eval")
plt.legend(loc="upper right")
plt.suptitle('Tranformer loss')
plt.show()

plt.plot(train_acc_history, label="train")
plt.plot(test_acc_history, 'r', label="eval")
plt.legend(loc="upper left")
plt.suptitle('Transformer accuracy')
plt.show()


# =============================================================================
# model = ViT()
# model.load_state_dict(torch.load(PATH))
# model.eval()            
# =============================================================================
