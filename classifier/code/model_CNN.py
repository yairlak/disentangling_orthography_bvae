#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNmodel64(nn.Module):
    def __init__(self, n_classes=10):
        super(CNNmodel64, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 13 * 13, 250)
        self.fc3 = nn.Linear(250, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train_model(n_epochs, model, dataloader, optimizer, criterion):
    
    losses = []
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print(f'starting epoch {epoch} of {n_epochs}')
        running_losses = []
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            if next(model.parameters()).is_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            n_print=500
            if i % n_print == n_print-1:    # print every mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_losses.append(running_loss)
                running_loss = 0.0
        losses.append(running_losses)
    return losses


def eval_model(model, testloader):
    correct, total = 0, 0
    # since we're not training, no need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            if next(model.parameters()).is_cuda:
                images = images.cuda()
                labels = labels.cuda()

            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct, total
