# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:11:09 2021

@author: thijs
"""

import json
import time
import torch
import torchvision
from torch import nn
import PIL
from einops import rearrange

from ResViT import Transformer
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Set seed for experimenting; remove in final code

import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

#%%

import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class PretrainedViTResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, dim = 128, num_tokens = 8, mlp_dim = 256, heads = 8, depth = 6, emb_dropout = 0.1, dropout= 0.1, batch_size=(100,100), pretrained=True):
        #super().__init__(BasicBlock, [3, 3, 3], *args, **kwargs)
        super(PretrainedViTResNet, self).__init__()
        
        self.in_planes = 16
        self.L = num_tokens
        self.cT = dim
        
        resnet = resnet18(pretrained=pretrained, progress=False).to(device)
        modules = list(resnet.children())
        self.resnet = nn.Sequential(*modules[:-2])
        #self.final = modules[-2:]
        #self.apply(_weights_init)
        outsize = 512
        
        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(batch_size[0],self.L, outsize),requires_grad = True) #Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(batch_size[1],outsize,self.cT),requires_grad = True) #Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wV)        
             
        
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std = .02) # initialized based on the paper

        #self.patch_conv= nn.Conv2d(64,dim, self.patch_size, stride = self.patch_size) 

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) #initialized based on the paper
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        
    def forward(self, img, mask = None):
        
        x = self.resnet(img)
        
        x = rearrange(x, 'b c h w -> b (h w) c') # 64 vectors each with 64 points. These are the sequences or word vecotrs like in NLP

        #Tokenization 
        wa = rearrange(self.token_wA, 'b h w -> b w h') #Transpose
        A= torch.einsum('bij,bjk->bik', x, wa) 
        A = rearrange(A, 'b h w -> b w h') #Transpose
        A = A.softmax(dim=-1)

        VV= torch.einsum('bij,bjk->bik', x, self.token_wV)       
        T = torch.einsum('bij,bjk->bik', A, VV)  
        #print(T.size())

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask) #main game
        x = self.to_cls_token(x[:, 0])       
        x = self.nn1(x)
        
        return x

#%% Custom dataset (CBIS-DDSM)

from torch.utils.data import DataLoader
from ResViT_train import CBISDataset

#%% Transform

transform = {
    'train': torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        #torchvision.transforms.RandomAffine(8, translate=(.15,.15)),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((12513.3505859375), (16529.138671875)), # not necessary with pre-normalized dataset
        torchvision.transforms.Lambda(lambda x: x.expand(3, -1, -1)), # go from BW images to color images
     ]),
    'val': torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((12513.3505859375), (16529.138671875)),
        torchvision.transforms.Lambda(lambda x: x.expand(3, -1, -1)),
     ])
}

#%% Training and evaluation

from args import parser
from ResViT_train import run, cross_validate, plot

if __name__ == "__main__":

    parser.add_argument('--no-pretrain', action='store_true',
                        help='do not use pretraining (pretrained ResNet is used by default)')
    args = parser.parse_args()
    vargs = vars(args)
    print(vargs)
    print()

    train_dataset = CBISDataset(args.train_data, args.batch_size_train, transform['train'], binary=args.binary_classification, oversample=args.oversample, sample)
    test_dataset = CBISDataset(args.val_data, args.batch_size_val, transform['val'], binary=args.binary_classification, oversample=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)
    
    #%%
    
    categories = 2 if args.binary_classification else 3
    pretrained = not args.no_pretrain
    batch_size = (args.batch_size_train, args.batch_size_val)
    
    # List of arguments
    num_tokens = args.num_tokens    # number of tokens used in transformer step
    depth = args.transform_depth    # number of transformer layers
    
    model = PretrainedViTResNet(pretrained=pretrained, num_classes=categories, dim=args.dim, mlp_dim=args.mlp_dim, num_tokens=num_tokens, depth=depth, batch_size=batch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9,weight_decay=1e-4)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[35,48],gamma = 0.1)

    if args.tensorboard_dir:
        tensorboard_writer = torch.utils.tensorboard.SummaryWriter(args.tensorboard_dir)
        tensorboard_writer.add_text('args', json.dumps(vars(args)))
        tensorboard_writer.add_text('transform', str(transform))
        #tensorboard_writer.add_text('model', str(model))
    else:
        tensorboard_writer = None
    
    init_time = time.time()
    if args.cross_val < 1:
        run(model, optimizer, train_loader, test_loader, args.epochs, args.binary_classification, tensorboard_writer)
    else:
        cross_validate(model, optimizer, train_dataset, test_loader, args.cross_val, args.epochs, transform, args.binary_classification, tensorboard_writer)
        
    minutes, seconds = divmod(time.time() - init_time, 60)
    print('Total execution time:', '{:.0f}m {:.1f}s'.format(minutes, seconds))
