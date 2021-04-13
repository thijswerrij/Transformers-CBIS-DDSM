# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:44:29 2021

@author: thijs
"""
import argparse
import json
import time
import PIL
import torch
import torchvision
import torch.utils.tensorboard
import sklearn.metrics
import matplotlib.pyplot as plt
import uuid
from torchvision.models import resnet18
import util

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Load hdf5

import h5py
import numpy as np

h5_path = "data/"

#%% Custom dataset (CBIS-DDSM)

from torch.utils.data import DataLoader
from ResViT_train import CBISDataset, train, evaluate, plot

# if is_binary = True, BENIGN and BENIGN_WITHOUT_CALLBACK are both assigned label 0
label_to_int = { 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 1, 'MALIGNANT' : 2 }
label_to_bin = { 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT' : 1 }

#%% Transform

transform = {
    'train': torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        #torchvision.transforms.RandomAffine(8, translate=(.15,.15)),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((12649.69140625, 16783.240234375)),
        torchvision.transforms.Lambda(lambda x: x.expand(3, -1, -1)), # go from BW images to color images
     ]),
    'val': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((12649.69140625, 16783.240234375)),
        torchvision.transforms.Lambda(lambda x: x.expand(3, -1, -1)),
     ])
}

#%% Training and evaluation

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', metavar='HDF5', #required=True, # currently using default values for testing
                        default="data/mass_case_description_train_set_scaled_0.1_cropped.h5",
                        help='training samples (HDF5)')
    parser.add_argument('--val-data', metavar='HDF5', #required=True,
                        default="data/mass_case_description_test_set_scaled_0.1_cropped.h5",
                        help='validation samples (HDF5)')
    parser.add_argument('--epochs', metavar='EPOCHS', type=int, default=300)
    parser.add_argument('--learning-rate', metavar='LR', type=float, default=0.0003)
    parser.add_argument('--batch-size-train', metavar='N', type=int, default=10,
                        help='batch size for training')
    parser.add_argument('--batch-size-val', metavar='N', type=int, default=10,
                        help='batch size for validation and test')
    parser.add_argument('--binary-classification', action='store_true',
                        help='use binary classification instead of 3-class classification')
    parser.add_argument('--oversample', action='store_true',
                        help='use oversampling to balance classes')
    parser.add_argument('--tensorboard-dir', metavar='DIR',
                        help='log statistics to tensorboard')
    args = parser.parse_args()
    vargs = vars(args)
    print(vargs)
    print()

    train_dataset = CBISDataset(args.train_data, args.batch_size_train, transform['train'], binary=args.binary_classification, oversample=args.oversample)
    test_dataset = CBISDataset(args.val_data, args.batch_size_val, transform['val'], binary=args.binary_classification, oversample=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)
    
    categories = 2 if args.binary_classification else 3
    
    # List of arguments
    model = resnet18(pretrained=True, progress=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9,weight_decay=1e-4)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[35,48],gamma = 0.1)

    if args.tensorboard_dir:
        tensorboard_writer = torch.utils.tensorboard.SummaryWriter(args.tensorboard_dir)
        tensorboard_writer.add_text('args', json.dumps(vars(args)))
        tensorboard_writer.add_text('transform', str(transform))
    else:
        tensorboard_writer = None
    
    init_time = time.time()
    train_loss_history, test_loss_history = [], []
    train_acc_history, test_acc_history = [], []
    train_conf_matrices, test_conf_matrices = [], []
    for epoch in range(1, args.epochs + 1):
        print('Epoch:', epoch)
        start_time = time.time()
        train_predict, train_target = train(model, optimizer, train_loader, train_loss_history, train_acc_history, train_conf_matrices, args.binary_classification)
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        eval_predict, eval_target = evaluate(model, test_loader, test_loss_history, test_acc_history, test_conf_matrices, args.binary_classification)

        if tensorboard_writer:
            # a bit hacky, it would be nicer if train and evaluate would return this
            tensorboard_writer.add_scalar('loss/train', train_loss_history[-1], epoch)
            tensorboard_writer.add_scalar('accuracy/train', train_acc_history[-1], epoch)
            tensorboard_writer.add_scalar('loss/test', test_loss_history[-1], epoch)
            tensorboard_writer.add_scalar('accuracy/test', test_acc_history[-1], epoch)
            tensorboard_writer.add_scalar('time per epoch', time.time() - start_time, epoch)
            tensorboard_writer.add_figure('confmat/train', util.plot_confmat(train_conf_matrices[-1]), epoch)
            tensorboard_writer.add_figure('confmat/test', util.plot_confmat(test_conf_matrices[-1]), epoch)

            if args.binary_classification:
                tensorboard_writer.add_scalar('auc_roc/train', sklearn.metrics.roc_auc_score(train_target == 1, train_predict[:, 1]), epoch)
                tensorboard_writer.add_figure('roc/train', util.plot_roc_curve(train_target == 1, train_predict[:, 1]), epoch)
                tensorboard_writer.add_scalar('auc_roc/test', sklearn.metrics.roc_auc_score(eval_target == 1, eval_predict[:, 1]), epoch)
                tensorboard_writer.add_figure('roc/test', util.plot_roc_curve(eval_target == 1, eval_predict[:, 1]), epoch)
        
    minutes, seconds = divmod(time.time() - init_time, 60)
    print('Total execution time:', '{:.0f}m {:.1f}s'.format(minutes, seconds))
    
#%% Storing of results
    
    file_name = args.train_data.split("/")[-1][0:9] # file name, e.g. mass_case
    results_folder_name = f"results/resnet_{file_name}_{str(uuid.uuid4())[:8]}"
    
    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)
    
    PATH = f"{results_folder_name}/resnet.pt"
    #torch.save(model.state_dict(), PATH)
        
    loss_history = np.stack([train_loss_history, test_loss_history])
    acc_history = np.stack([train_acc_history, test_acc_history])
    
    with h5py.File(f"{results_folder_name}/stats.h5",'w') as f:
        loss_set = f.create_dataset(
            "loss_history", loss_history.shape, data=loss_history
        )
        bp_set = f.create_dataset(
            "acc_history", acc_history.shape, data=acc_history
        )
        
    params = open(f"{results_folder_name}/params.txt", 'w')
    
    params.write(
        f"arguments:\n{vargs}\n"
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