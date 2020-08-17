import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet_manifold as RN
import pyramidnet as PYRM
# import utils
from utils import get_imbalanced_data
import numpy as np
import inatu_loader as inat2018_loader
import random
import warnings

torch.autograd.set_detect_anomaly(True)
global directory
warnings.filterwarnings("ignore")

def load_data(args):

    if args.dataset.startswith('cifar'):

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            train_data= datasets.CIFAR100(args.data_dir, train=True, download=True, transform=transform_train)
            val_data= datasets.CIFAR100(args.data_dir, train=False, transform=transform_test)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_data= datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform_train)
            val_data= datasets.CIFAR10(args.data_dir, train=False, transform=transform_test)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

        if args.imb_factor<1.0:
            train_data = get_imbalanced_data(args, train_data)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data,batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    elif args.dataset == 'imagenet':
        train_file = os.path.join('/home/rahulahuja/dsta/Temp/rahul/inat2018/train2018.json')
        val_file= os.path.join('/home/rahulahuja/dsta/Temp/rahul/inat2018/val2018.json')
        data_target_dir = os.path.join("/home/rahulahuja/dsta/Temp/rahul/inat2018/")

        train_data = inat2018_loader.INAT(data_target_dir, train_file, is_train=True)
        test_data = inat2018_loader.INAT(data_target_dir, val_file, is_train=False)


        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            test_data,batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

        numberofclass = 8142

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))


    return train_loader, val_loader, numberofclass
