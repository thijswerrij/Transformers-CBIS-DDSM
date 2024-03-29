# -*- coding: utf-8 -*-
"""
Code used to train and test on CBIS-DDSM data
"""

import time
import torch
import torchvision
import torch.utils.tensorboard
import sklearn.metrics
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Load hdf5

import h5py
import numpy as np

h5_path = "data/"

def read_hdf5(file_name):
    with h5py.File(file_name, "r") as file:
        images = np.array(file["/images"]).astype('float32')
        labels = np.array(file["/meta"]).astype("str")
        bp = np.array(file["/bp"]).astype("str")
        patient_ids = np.array(file["/patient_id"]).astype("str")
    
    return images, labels, bp, patient_ids

def plot(img):
    # plot the image using matplotlib
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

#%% Custom dataset (CBIS-DDSM)

from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import RandomOverSampler 

# if is_binary = True, BENIGN and BENIGN_WITHOUT_CALLBACK are both assigned label 0
label_to_int = { 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 1, 'MALIGNANT' : 2 }
label_to_bin = { 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT' : 1 }

class CBISDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_name, batch_size=None, transform=None, binary=False, reorient=False, sample=None, oversample=False, bp_filter=""):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.images, labels, bp_list, patient_ids = read_hdf5(file_name)
        bp_list = np.array(bp_list) # bp_list[i] e.g. ['L', 'LEFT', 'MLO']
        patient_ids = np.array([i[0] for i in patient_ids])
        
        filtered = None
        if bp_filter == "L" or bp_filter == "R":
            filtered = bp_list[:,0] == bp_filter
        elif bp_filter == "LEFT" or bp_filter == "RIGHT":
            filtered = bp_list[:,1] == bp_filter
        elif bp_filter == "CC" or bp_filter == "MLO":
            filtered = bp_list[:,2] == bp_filter
        
        if filtered is not None:
            self.images, labels, bp_list, patient_ids = self.images[filtered], labels[filtered], bp_list[filtered], patient_ids[filtered]
            print(f"Filtered on {bp_filter}")
        
        if sample:
            self.images, labels, bp_list, patient_ids = self.images[:sample], labels[:sample], bp_list[:sample], patient_ids[:sample]
        
        #mean_val, std_val = self.images.mean(), self.images.std()
        #print(f"{file_name}\nmean: {mean_val}\nstd: {std_val}\n")
            
        if reorient:
            for i in range(len(bp_list)):
                if bp_list[i][0] == 'R':
                    self.images[i] = np.flip(self.images[i], axis=1)
        
        if binary:
            self.labels = np.array([label_to_bin[i[0]] for i in labels])
        else:
            self.labels = np.array([label_to_int[i[0]] for i in labels])
        
        if not sample:
            if oversample:
                print(f"Labels before oversampling - 0: {sum(self.labels==0)}, 1: {sum(self.labels==1)}, 2: {sum(self.labels==2)}")
                #img_shape = self.images.shape
                #self.images = self.images.reshape(img_shape[0], img_shape[1]*img_shape[2])
                ros = RandomOverSampler(random_state=42)
                #self.images, self.labels = ros.fit_resample(self.images, self.labels)
                
                # very hacky way to do this, but I didn't see an alternative because it needs to include patient ids
                resampled_ids, self.labels = ros.fit_resample([[i] for i in range(len(self.images))], self.labels)
                
                self.images = np.array([self.images[i[0]] for i in resampled_ids])
                patient_ids = np.array([patient_ids[i[0]] for i in resampled_ids])
                
                #self.images = self.images.reshape(self.images.shape[0], img_shape[1], img_shape[2])
                
            if batch_size:
                data_size = int(len(self.labels)/batch_size)*batch_size
                self.images, self.labels, patient_ids = self.images[:data_size], self.labels[:data_size], patient_ids[:data_size]
            
            if oversample:
                print(f"Final label distribution - 0: {sum(self.labels==0)}, 1: {sum(self.labels==1)}, 2: {sum(self.labels==2)}\n")
            
        self.patient_ids = patient_ids
        
        self.transform = transform
        self.batch_size = batch_size
        self.binary = binary

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i]
        label = np.array(self.labels[i])
        
        #print(image.dtype)
        
        if self.transform:
            image = self.transform(np.copy(image))
        else:
            image = torchvision.transforms.ToTensor()(np.copy(image))
        
        #print(image.dtype)
        
        sample = (image, label)

        return sample
        
class CBISSubset(CBISDataset):
    def __init__(self, dataset, ids, transform=None):
        self.images, self.labels = dataset.images[ids], dataset.labels[ids]
        
        self.batch_size = dataset.batch_size
        
        if self.batch_size:
            data_size = int(len(self.labels)/self.batch_size)*self.batch_size
            self.images, self.labels = self.images[:data_size], self.labels[:data_size]
            
        self.transform = transform
    
def get_mean_std(loader, non_zero=False):
    total_sum, total_squared_sum, num_batches = 0, 0, 0
    
    for data, _ in loader:
        if non_zero:
            minimum = data.min()
            data[data == minimum] = 0
            mask = data != minimum
            total_sum += data.sum()/mask.sum()
            total_squared_sum += (data**2).sum()/mask.sum()
        else:
            total_sum += torch.mean(data)
            total_squared_sum += torch.mean(data**2)
        num_batches += 1
    
    mean = total_sum / num_batches
    std = (total_squared_sum / num_batches - mean**2)**0.5
    
    return mean, std

#%%

def train(model, optimizer, data_loader, epoch, loss_history=[], acc_history=[], conf_matrices=[], auc_scores=[], binary=False, tensorboard_writer=None, tb_name="train"):
    total_samples = len(data_loader.dataset)
    model.train()
    minibatches = 0
    correct_samples = 0
    total_loss = 0
    conf_mat = 0

    outputs, targets = [], []

    predicted_labels = [0,1] if binary else [0,1,2]

    for i, (data, target) in enumerate(data_loader):
        minibatches += 1
        data, target = data.to(device), target.to(device, dtype=torch.int64)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        outputs.append(output.cpu().detach().numpy())
        targets.append(target.cpu().detach().numpy())
        
        total_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        correct_samples += pred.eq(target).sum().item()
        conf_mat = np.add(conf_mat, confusion_matrix(target.cpu(),pred.cpu(), labels=predicted_labels))

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
        
    avg_loss = total_loss / minibatches
    accuracy = correct_samples / total_samples
    loss_history.append(avg_loss)
    acc_history.append(accuracy)
    conf_matrices.append(conf_mat)
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    if binary:
        roc_auc_score = sklearn.metrics.roc_auc_score(targets == 1, outputs[:, 1])
        auc_scores.append(roc_auc_score)
    
    if tensorboard_writer:
        tensorboard_writer.add_scalar(f'loss/{tb_name}', avg_loss, epoch)
        tensorboard_writer.add_scalar(f'accuracy/{tb_name}', accuracy, epoch)
        tensorboard_writer.add_figure(f'confmat/{tb_name}', util.plot_confmat(conf_mat), epoch)
    
        if binary:
            tensorboard_writer.add_scalar(f'auc_roc/{tb_name}', roc_auc_score, epoch)
            tensorboard_writer.add_figure(f'roc/{tb_name}', util.plot_roc_curve(targets == 1, outputs[:, 1]), epoch)
    
    return outputs, targets
            
def evaluate(model, data_loader, epoch, loss_history=[], acc_history=[], conf_matrices=[], auc_scores=[], binary=False, tensorboard_writer=None, tb_name="test"):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    minibatches = 0
    correct_samples = 0
    total_loss = 0
    conf_mat = 0

    outputs, targets = [], []
    
    predicted_labels = [0,1] if binary else [0,1,2]

    with torch.no_grad():
        for data, target in data_loader:
            minibatches += 1
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = torch.argmax(output, dim=1)

            outputs.append(output.cpu().detach().numpy())
            targets.append(target.cpu().detach().numpy())
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum().item()
            conf_mat = np.add(conf_mat, confusion_matrix(target.cpu(),pred.cpu(), labels=predicted_labels))

    avg_loss = total_loss / minibatches
    accuracy = correct_samples / total_samples
    loss_history.append(avg_loss)
    acc_history.append(accuracy)
    conf_matrices.append(conf_mat)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * accuracy) + '%)\n')
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    if binary:
        roc_auc_score = sklearn.metrics.roc_auc_score(targets == 1, outputs[:, 1])
        auc_scores.append(roc_auc_score)
    
    if tensorboard_writer:
        tensorboard_writer.add_scalar(f'loss/{tb_name}', avg_loss, epoch)
        tensorboard_writer.add_scalar(f'accuracy/{tb_name}', accuracy, epoch)
        tensorboard_writer.add_figure(f'confmat/{tb_name}', util.plot_confmat(conf_mat), epoch)
    
        if binary:
            tensorboard_writer.add_scalar(f'auc_roc/{tb_name}', roc_auc_score, epoch)
            tensorboard_writer.add_figure(f'roc/{tb_name}', util.plot_roc_curve(targets == 1, outputs[:, 1]), epoch)
    
    return outputs, targets

#%%

from sklearn.model_selection import GroupKFold

def run(model, optimizer, train_loader, test_loader, epochs, binary, tensorboard_writer=None):
    train_loss_history, test_loss_history = [], []
    train_acc_history, test_acc_history = [], []
    train_conf_matrices, test_conf_matrices = [], []
    train_auc_scores, test_auc_scores = [], []
    
    for epoch in range(1, epochs + 1):
        print('Epoch:', epoch)
        start_time = time.time()
        train_predict, train_target = train(model, optimizer, train_loader, epoch, train_loss_history, train_acc_history, train_conf_matrices, train_auc_scores, binary, tensorboard_writer)
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        eval_predict, eval_target = evaluate(model, test_loader, epoch, test_loss_history, test_acc_history, test_conf_matrices, test_auc_scores, binary, tensorboard_writer)

        if tensorboard_writer:
            tensorboard_writer.add_scalar('time per epoch', time.time() - start_time, epoch)
            
    if tensorboard_writer:
        tensorboard_writer.close()

def cross_validate(model, optimizer, train_dataset, test_loader, k_folds, epochs, transform, binary, tensorboard_writer=None):
    #kfold = KFold(n_splits=k_folds, shuffle=True)
    kfold = GroupKFold(n_splits=k_folds) # added to ensure that patients do not show up in both training and validation sets
    
    print(f"{k_folds}-folds cross-validation")
    
    train_loss_history, eval_loss_history, test_loss_history = [], [], []
    train_acc_history, eval_acc_history, test_acc_history = [], [], []
    train_conf_matrices, eval_conf_matrices, test_conf_matrices = [], [], []
    train_auc_scores, eval_auc_scores, test_auc_scores = [], [], []
    best_auc_score_ids = []
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset,groups=train_dataset.patient_ids)):
        print('Fold', fold)
        
        train_fold = CBISSubset(train_dataset, train_ids, transform['train'])
        val_fold = CBISSubset(train_dataset, test_ids, transform['val'])
        
        train_loader = DataLoader(train_fold, batch_size=train_fold.batch_size, shuffle=True)
        eval_loader = DataLoader(val_fold, batch_size=val_fold.batch_size, shuffle=False)
        
        for epoch in range(1, epochs + 1):
            print('Epoch:', epoch)
            start_time = time.time()
            train_predict, train_target = train(model, optimizer, train_loader, epoch, train_loss_history, train_acc_history, train_conf_matrices, train_auc_scores, binary, tensorboard_writer, f"fold {fold} train")
            print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
            eval_predict, eval_target = evaluate(model, eval_loader, epoch, eval_loss_history, eval_acc_history, eval_conf_matrices, eval_auc_scores, binary, tensorboard_writer, f"fold {fold} eval")
            
            print("Testing:")
            test_predict, test_target = evaluate(model, test_loader, epoch, test_loss_history, test_acc_history, test_conf_matrices, test_auc_scores, binary, tensorboard_writer, f"fold {fold} test")
            if tensorboard_writer:
                tensorboard_writer.add_scalar(f"time per epoch/fold {fold}", time.time() - start_time, epoch)
            
        if binary:
            best_score_this_fold = np.argmax(eval_auc_scores[-epochs:]) + epochs*fold
            best_auc_score_ids.append(best_score_this_fold)
            best_score_str = (f"Fold {fold}:  \nAUC: {eval_auc_scores[best_score_this_fold]}  \nEpoch {best_score_this_fold+1}  \n"
            f"Test accuracy: {test_acc_history[best_score_this_fold]}  \nTest loss: {test_loss_history[best_score_this_fold]}  \n  \n")
            print(best_score_str)
            if tensorboard_writer:
                tensorboard_writer.add_scalar('loss/best test per fold', test_loss_history[best_score_this_fold], fold)
                tensorboard_writer.add_scalar('accuracy/best test per fold', test_acc_history[best_score_this_fold], fold)
                tensorboard_writer.add_text('best eval per fold', best_score_str, fold)
                
    if tensorboard_writer:
        tensorboard_writer.close()
