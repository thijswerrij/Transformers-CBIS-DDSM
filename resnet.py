# -*- coding: utf-8 -*-
"""
Run ResNet-18 on CBIS-DDSM data
"""
import json
import time
import PIL
import torch
import torchvision
import torch.utils.tensorboard
from torchvision.models import resnet18
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    model = resnet18(pretrained=pretrained, progress=False).to(device)
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
