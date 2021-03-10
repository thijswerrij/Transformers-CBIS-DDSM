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
import matplotlib.pyplot as plt
import uuid

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
    with h5py.File(f"{h5_path}{file_name}.h5", "r+") as file:
        images = np.array(file["/images"]).astype("uint16")
        labels = np.array(file["/meta"]).astype("str")
        bp = np.array(file["/bp"]).astype("str")
    
    return images, labels, bp

def plot(img):
    # plot the image using matplotlib
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

#%% Custom dataset (CBIS-DDSM)

from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import RandomOverSampler 

label_to_int = { 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 1, 'MALIGNANT' : 2 }
label_to_bin = { 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT' : 1 }

class CBISDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_name, batch_size=None, transform=None, binary=False, reorient=False, sample=None, oversample=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # If loading the hdf5 file initially doesn't succeed, try up to four times
        # (added as I was running multiple experiments on same data)
        
        for x in range(0, 4):
            failed = False
            try:
                self.images, labels, self.bp_list = read_hdf5(file_name)
                break
            except OSError:
                print(f"Attempt {x+1} of loading {file_name} failed.")
                failed = True
            
            if failed and x != 3:
                time.sleep(5*(x+1)) # wait for a few seconds before trying again
        if failed:
            raise OSError(f"Loading {file_name} failed.")
        
        if sample:
            self.images, labels, self.bp_list = self.images[:sample], labels[:sample], self.bp_list[:sample]
            
        if reorient:
            for i in range(len(self.bp_list)):
                if self.bp_list[i] == 'R':
                    self.images[i] = np.flip(self.images[i], axis=1)
        
        if binary:
            self.labels = np.array([label_to_bin[i[0]] for i in labels])
        else:
            self.labels = np.array([label_to_int[i[0]] for i in labels])
        
        if not sample:
            if oversample:
                print(f"{file_name}\nLabels before oversampling - 0: {sum(self.labels==0)}, 1: {sum(self.labels==1)}, 2: {sum(self.labels==2)}")
                img_shape = self.images.shape
                self.images = self.images.reshape(img_shape[0], img_shape[1]*img_shape[2])
                ros = RandomOverSampler(random_state=42)
                self.images, self.labels = ros.fit_resample(self.images, self.labels)
                self.images = self.images.reshape(self.images.shape[0], img_shape[1], img_shape[2])
                
            if batch_size:
                data_size = int(len(self.labels)/batch_size)*batch_size
                self.images, self.labels = self.images[:data_size], self.labels[:data_size]
            
            if oversample:
                print(f"Final label distribution - 0: {sum(self.labels==0)}, 1: {sum(self.labels==1)}, 2: {sum(self.labels==2)}\n")
            
        
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i].astype('float')
        label = np.array(self.labels[i])
        
        
        if self.transform:
            image = self.transform(PIL.Image.fromarray(np.copy(image)))
        else:
            image = torchvision.transforms.ToTensor()(PIL.Image.fromarray(np.copy(image)))
        
        sample = (image, label)

        return sample


transform = {
    'train': torchvision.transforms.Compose([
         #torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
         #torchvision.transforms.RandomAffine(8, translate=(.15,.15)),
         torchvision.transforms.ToTensor(),
         #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         torchvision.transforms.Normalize((0.5), (0.5))
     ]),
    'val': torchvision.transforms.Compose([
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5), (0.5))
     ])
}

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
    
    predicted_labels = [0,1] if binary else [0,1,2]

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum().item()
            conf_mat = np.add(conf_mat, confusion_matrix(target.cpu(),pred.cpu(), labels=predicted_labels))

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
    oversample = True
    reorient = (True, True)
    
    #file_name = "calc_case_description"
    file_name = "mass_case_description"
    
    #file_params = "180x180_cropped"
    file_params = "scaled_0.1_cropped"
    train_dataset = CBISDataset(f"{file_name}_train_set_{file_params}", batch_size[0], transform['train'], binary=is_binary, oversample=oversample, reorient=reorient[0])
    test_dataset = CBISDataset(f"{file_name}_test_set_{file_params}", batch_size[1], transform['val'], binary=is_binary, oversample=False, reorient=reorient[1])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size[1], shuffle=False)
    
    N_EPOCHS = 300
    categories = 2 if is_binary else 3
    
    # List of arguments
    num_tokens = 8
    depth = 6
    learning_rate = 0.0003
    
    model = ViTResNet(BasicBlock, [3, 3, 3], in_channels=1, num_classes=categories, num_tokens=num_tokens, depth=depth, batch_size=batch_size).to(device)
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
    
    if is_binary:
        file_params += "_binary"
    if oversample:
        file_params += "_oversampled"
    file_params += f"_epochs={N_EPOCHS}"
    
    results_folder_name = f"results/{file_name}_{file_params}_{str(uuid.uuid4())[:8]}"
    
#%%
    
    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)
    
    if False:
        PATH = f"{results_folder_name}/ViTRes.pt"
        torch.save(model.state_dict(), PATH)
        
    params = open(f"{results_folder_name}/params.txt", 'w')
    
    params.write(
        f"number of tokens: {num_tokens}\n"
        f"depth: {depth}\n"
        f"learning rate: {format(learning_rate, 'f')}\n"
        f"reoriented: {reorient}\n"
        f"\ntransformations: \n {transform}\n"
        f"\nExecution time: {int(minutes)}m {seconds:.1f}s")
    params.close()
    
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

#%% ROC curve
    
    from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
    
    predicted_labels = [0,1] if is_binary else [0,1,2]
    
    model.eval()
    probabilities = np.array([]).reshape(0,categories)
    labels = np.array([])
    
    conf_mat = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            output = model(data)
            _, pred = torch.max(F.log_softmax(output, dim=1), dim=1)
            probs = output.cpu().numpy()
            probabilities = np.concatenate((probabilities,probs))
            labels = np.concatenate((labels,target.cpu().numpy()))
            
            conf_mat = np.add(conf_mat, confusion_matrix(target.cpu(),pred.cpu(), labels=predicted_labels))
    
#%%
    auc = []
    
    #loop = range(1,2) if categories==2 else range(categories)
    loop = range(categories)
    for i in loop:
        i_probs = probabilities[:,i]
        i_labels = (labels == i)
        fpr, tpr, _ = roc_curve(i_labels,i_probs)
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.suptitle('ROC curve for category = ' +str(i))
        if save_plots:
            plt.savefig(f"{results_folder_name}/roc{i}.png")
        plt.show()
        auc.append(roc_auc_score(i_labels,i_probs))
        
#%% Plot confusion matrices
    
    plt.matshow(conf_mat, cmap=plt.cm.Blues)
    
    conf_N = conf_mat.shape[0]
    for i in range(conf_N):
        for j in range(conf_N):
            plt.text(j, i, str(conf_mat[i,j]), va='center', ha='center')
    if save_plots:
        plt.savefig(f"{results_folder_name}/conf.png")
    plt.show()

# =============================================================================
# model = ViT()
# model.load_state_dict(torch.load(PATH))
# model.eval()            
# =============================================================================
