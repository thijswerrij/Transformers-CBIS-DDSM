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

from ResViT import ViTResNet, BasicBlock
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

class PretrainedViTResNet(ViTResNet):
    def __init__(self, pretrained=True, *args, **kwargs):
        super().__init__(BasicBlock, [3, 3, 3], *args, **kwargs)
        
        resnet = resnet18(pretrained=pretrained, progress=False).to(device)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(batch_size[0],self.L, 512),requires_grad = True) #Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(batch_size[1],512,self.cT),requires_grad = True) #Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wV) 
        
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
from ResViT_train import CBISDataset, train, evaluate, plot

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

if __name__ == "__main__":

    parser.add_argument('--no-pretrain', action='store_true',
                        help='do not use pretraining (pretrained ResNet is used by default)')
    args = parser.parse_args()
    vargs = vars(args)
    print(vargs)
    print()

    train_dataset = CBISDataset(args.train_data, args.batch_size_train, transform['train'], binary=args.binary_classification, oversample=args.oversample)
    test_dataset = CBISDataset(args.val_data, args.batch_size_val, transform['val'], binary=args.binary_classification, oversample=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)
    
    #%%
    
    categories = 2 if args.binary_classification else 3
    pretrained = not args.no_pretrain
    batch_size = (args.batch_size_train, args.batch_size_val)
    
    # List of arguments
    num_tokens = 8         # number of tokens used in transformer step
    depth = 6              # number of transformer layers
    
    model = PretrainedViTResNet(pretrained=pretrained, in_channels=3, num_classes=categories, num_tokens=num_tokens, depth=depth, batch_size=batch_size).to(device)
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
        train_predict, train_target = train(model, optimizer, train_loader, train_loss_history, train_acc_history, train_conf_matrices, epoch, args.binary_classification, tensorboard_writer)
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        eval_predict, eval_target = evaluate(model, test_loader, test_loss_history, test_acc_history, test_conf_matrices, epoch, args.binary_classification, tensorboard_writer)

        if tensorboard_writer:
            tensorboard_writer.add_scalar('time per epoch', time.time() - start_time, epoch)
        
    minutes, seconds = divmod(time.time() - init_time, 60)
    print('Total execution time:', '{:.0f}m {:.1f}s'.format(minutes, seconds))
