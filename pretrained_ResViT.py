# -*- coding: utf-8 -*-
"""
Run a pretrained ResNet-18 model in combination with visual transformer modules on CBIS-DDSM

By Thijs Werrij (thijswerrij)
Based on code by Md Tahmid Hossain (tahmid0007)
"""

import json
import time
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

import PIL
from einops import rearrange
import os

from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Transformer code by Md Tahmid Hossain (tahmid0007), see repository acknowledgements

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        #print(out.size())
        return out



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = True) # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            x = attention(x, mask = mask) # go to attention
            x = mlp(x) #go to MLP_Block
        return x
    
#%%

class PretrainedViTResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, dim = 128, num_tokens = 8, mlp_dim = 256, heads = 8, depth = 6, emb_dropout = 0.1, dropout= 0.1, batch_size=(100,100), pretrained=True, remove_last_block=False):
        #super().__init__(BasicBlock, [3, 3, 3], *args, **kwargs)
        super(PretrainedViTResNet, self).__init__()
        
        self.in_planes = 16
        self.L = num_tokens
        self.cT = dim
        
        resnet = resnet18(pretrained=pretrained, progress=False).to(device)
        modules = list(resnet.children())
        if remove_last_block: # original paper replaces last BasicBlock with VT modules; if remove_last_block = True, last BB is removed
            self.resnet = nn.Sequential(*modules[:-3])
            outsize = 256
        else:
            self.resnet = nn.Sequential(*modules[:-2])
            outsize = 512
        #self.final = modules[-2:]
        #self.apply(_weights_init)
        
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
from cbis_ddsm_train import CBISDataset

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
from cbis_ddsm_train import run, cross_validate, plot

if __name__ == "__main__":

    parser.add_argument('--no-pretrain', action='store_true',
                        help='do not use pretraining (pretrained ResNet is used by default)')
    parser.add_argument('--remove-last-block', action='store_true',
                        help='If set to true, the two Basic Blocks of ResNet-18 are removed')
    
    args = parser.parse_args()
    vargs = vars(args)
    print(vargs)
    print()

    train_dataset = CBISDataset(args.train_data, args.batch_size_train, transform['train'], binary=args.binary_classification, oversample=args.oversample, bp_filter=args.filter)
    test_dataset = CBISDataset(args.val_data, args.batch_size_val, transform['val'], binary=args.binary_classification, oversample=False, bp_filter=args.filter)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)
    
    #%%
    
    categories = 2 if args.binary_classification else 3
    pretrained = not args.no_pretrain
    batch_size = (args.batch_size_train, args.batch_size_val)
    
    # List of arguments
    num_tokens = args.num_tokens    # number of visual tokens
    depth = args.transform_depth    # number of transformer modules
    
    model = PretrainedViTResNet(pretrained=pretrained, num_classes=categories, dim=args.dim, mlp_dim=args.mlp_dim, num_tokens=num_tokens, depth=depth, remove_last_block=args.remove_last_block, batch_size=batch_size).to(device)
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
    if args.cross_val < 1:
        run(model, optimizer, train_loader, test_loader, args.epochs, args.binary_classification, tensorboard_writer)
    else:
        cross_validate(model, optimizer, train_dataset, test_loader, args.cross_val, args.epochs, transform, args.binary_classification, tensorboard_writer)
        
    minutes, seconds = divmod(time.time() - init_time, 60)
    print('Total execution time:', '{:.0f}m {:.1f}s'.format(minutes, seconds))
    
#%% Save model

    if type(args.model) is str:
        
        if not os.path.exists(args.model):
            os.makedirs(args.model)
        
        torch.save(model.state_dict(), f"{args.model}/model.pt")
