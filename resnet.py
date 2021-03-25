# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:44:29 2021

@author: thijs
"""
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import uuid

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

#%% Load hdf5

import h5py
import numpy as np

h5_path = "data/"

#%% Custom dataset (CBIS-DDSM)

from torch.utils.data import DataLoader

label_to_int = { 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 1, 'MALIGNANT' : 2 }
label_to_bin = { 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT' : 1 }

from ResViT_train import CBISDataset, train, evaluate, plot

transform = {
    'train': torchvision.transforms.Compose([
        #torchvision.transforms.RandomHorizontalFlip(),
        #torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        #torchvision.transforms.RandomAffine(8, translate=(.15,.15)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'val': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
}

#%%

from torchvision.models import resnet18, resnet152

if __name__ == "__main__":

    batch_size = (10, 10)
    is_binary = False
    oversample = False
    reorient = (False, False)
    
    #file_name = "calc_case_description"
    file_name = "mass_case_description"
    
    #file_params = "180x180_cropped"
    file_params = "scaled_0.1_cropped"
    
    #dataset = CBISDataset(f"{file_name}_train_set_{file_params}", batch_size[0], transform['train'], binary=is_binary, oversample=False, reorient=False)
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.7)+1, int(len(dataset)*0.3)])
    train_dataset = CBISDataset(f"{file_name}_train_set_{file_params}", batch_size[0], transform['train'], binary=is_binary, oversample=oversample, reorient=reorient[0])
    test_dataset = CBISDataset(f"{file_name}_test_set_{file_params}", batch_size[1], transform['val'], binary=is_binary, oversample=False, reorient=reorient[1])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size[1], shuffle=False)
    
    N_EPOCHS = 300
    categories = 2 if is_binary else 3
    
    # List of arguments
    learning_rate = 0.0003
    
    model = resnet18(pretrained=True, progress=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
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
    print('Total execution time:', '{:.0f}m {:.1f}s'.format(minutes, seconds))
    
#%%
    
    results_folder_name = f"results/resnet_{str(uuid.uuid4())[:8]}"
    
    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)
    
    PATH = f"{results_folder_name}/resnet.pt"
    torch.save(model.state_dict(), PATH)
        
    loss_history = np.stack([train_loss_history, test_loss_history])
    acc_history = np.stack([train_acc_history, test_acc_history])
    
    with h5py.File(f"{results_folder_name}/stats.h5",'w') as f:
        loss_set = f.create_dataset(
            "loss_history", loss_history.shape, data=loss_history
        )
        bp_set = f.create_dataset(
            "acc_history", acc_history.shape, data=acc_history
        )
    
#%% Loss & accuracy plots
    
    save_plots = True
    
    #avg_train_loss_history = np.mean(np.array(train_loss_history).reshape(N_EPOCHS,-1), axis=1)
    
    #plt.gca().set_ylim([0,None])
    
    plt.figure()
    plt.plot(test_loss_history, 'r', label="eval")
    plt.plot(train_loss_history, label="train")
    plt.legend(loc="upper right")
    plt.suptitle('Transformer loss')
    if save_plots:
        plt.savefig(f"{results_folder_name}/loss.png")
    plt.show()
    plt.clf()
    
    plt.plot(test_acc_history, 'r', label="eval")
    plt.plot(train_acc_history, label="train")
    plt.legend(loc="upper left")
    plt.suptitle('Transformer accuracy')
    if save_plots:
        plt.savefig(f"{results_folder_name}/accuracy.png")
    plt.show()
    plt.clf()