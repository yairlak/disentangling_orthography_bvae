#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

"""
import os
import argparse
# My Modules
from dataset import get_CIFAR10
from model import CNNmodel, train_model, eval_model
from viz import plot_loss, plot_samples
# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# PARSER: it is useful for executable code from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--n-epochs', default=5, type=int)
parser.add_argument('--batch-size', default=4, type=int)
# OPTIMIZER
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)

args = parser.parse_args()

torch.manual_seed(args.seed) # For reproducibility

# LOAD DATA
trainloader, testloader = get_CIFAR10(args.batch_size)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
dataiter = iter(trainloader)

# PLOT SOME TRAIN SAMPLES
images, labels = next(dataiter) 
plot_samples(torchvision.utils.make_grid(images),
             '../figures/train_samples.png') # Show these images

print(' '.join(f'{classes[labels[j]]:5s}' for j in range(args.batch_size))) # Print labels

# DEFINE MODEL, LOSS & OPTIMZIER
model = CNNmodel()
criterion = nn.CrossEntropyLoss() # Classification Cross-Entropy loss
optimizer = optim.SGD(model.parameters(),
                      lr=args.lr,
                      momentum=args.momentum) # SGD with momentum
print(model)

###############
# TRAIN MODEL #
###############
losses_epochs = train_model(args.n_epochs,
                            model,
                            trainloader,
                            optimizer,
                            criterion)

# PLOT TRAIN LOSS
plot_loss(losses_epochs,
          fn2save='../figures/train_loss.png') # Plot training evolution

# SAVE MODEL
os.makedirs('../output', exist_ok=True)
fn_model = '../output/cifar_net.pth'
torch.save(model.state_dict(), fn_model) # Save trained model
print('Finished Training')

##############
# EVAL MODEL #
##############

# PLOTTING
dataiter = iter(testloader) # Load test set
images, labels = next(dataiter)
plot_samples(torchvision.utils.make_grid(images),
             '../figures/test_samples.png') # Plot images

print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# EVAL on a few samples
outputs = model(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

# EVAL on all test data
correct, total = eval_model(model, testloader)
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')