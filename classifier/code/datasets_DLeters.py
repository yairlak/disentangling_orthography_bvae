#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import sys

sys.path.append('../../../')
from utils.datasets import get_dataloaders
from torch.utils.data import TensorDataset


"""The output of torchvision datasets are PILImage images of range [0, 1].
We transform them to Tensors of normalized range [-1, 1].
"""
def get_dletters(batch_size=64):
    # https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader

    train_loader = get_dataloaders('dletters',
                                           batch_size=batch_size,
                                           root='../../../data/dwords/',
                                           file_name='dletters_n5_AB.npz')

    classes_str = list(set(train_loader.dataset.lat_values_str[:, 0]))
    classes_int = train_loader.dataset.lat_values[:, 0]

    testset = train_loader.dataset.imgs

    # por qué cuando en el codigo original se itera sobre este dataloader no es necesario hacer esta transformación
    # ver si lo puedo hacer aca
    # y eso haría mas similar este proceso
    tensor_x = torch.Tensor(testset)/255  # transform to torch tensor
    tensor_x = tensor_x[:, None, :, :]
    tensor_y = torch.tensor(classes_int, dtype=torch.int64)

    my_dataset = TensorDataset(tensor_x, tensor_y)

    trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)


    return trainloader, testloader, classes_str

def get_CIFAR10(batch_size):
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    
    return trainloader, testloader
