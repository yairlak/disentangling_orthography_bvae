#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:20:25 2023

@author: wish_della
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_samples(img, fn2save='samples.png'):
    fig, _ = plt.subplots()
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    # SAVE
    os.makedirs(os.path.dirname(fn2save), exist_ok=True)
    fig.savefig(fn2save)
    
    
def plot_loss(losses, fn2save='loss.png'):
    fig, axs = plt.subplots(1, 2)
    # Per batch
    axs[0].plot([j for i in losses for j in i])
    # Per epoch
    axs[1].errorbar(x=range(1, len(losses)+1),
                y=[np.mean(l) for l in losses],
                yerr=[np.std(l) for l in losses])
    # COSMETICS
    axs[0].set_xlabel('Batch', fontsize=14)
    axs[0].set_ylabel('Loss', fontsize=14)
    axs[1].set_xlabel('Epoch', fontsize=14)
    axs[1].set_ylabel('Mean loss', fontsize=14)
    plt.subplots_adjust(wspace=0.5)
    # SAVE
    os.makedirs(os.path.dirname(fn2save), exist_ok=True)
    fig.savefig(fn2save)