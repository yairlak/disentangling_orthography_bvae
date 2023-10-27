#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:29:23 2022

Based on dsprites data structure

import numpy as np
data = np.load('data/dsprite_train.npz', allow_pickle= True, encoding='latin1')
print(data.files)



@author: aakash
"""
from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import subprocess, shlex, shutil, io, os, random, gc, time
from tqdm import tqdm
import pickle

import numpy as np
words = [w.strip('\n').strip() for w in open('vocab_small.txt', 'r').readlines()]

####
def gen2(savepath=None, text = 'text', index=1, mirror=False,
        invert=False, fontname='Arial', W = 64, H = 64, size=24,
        xshift=0, yshift=0, upper=0, show=None):
    if upper:
        text = text.upper()
    if invert:
        text = text[::-1]
    img = Image.new("L", (W,H), color = (0))
    fnt = ImageFont.truetype(fontname+'.ttf', size)
    draw = ImageDraw.Draw(img)
    w, h = fnt.getsize(text)
    draw.text((xshift + (W-w)/2, yshift + (H-h)/2), text, font=fnt, fill='white')
    return img


path_out='../data/words/',

#define words, sizes, fonts
# wordlist = words
# sizes = range(15,31,2)
# fonts = ['arial', 'times', 'comic', 'cour', 'calibri']
# xshifts = range(-8, 8,2)
# yshifts = range(-8, 8,2)

wordlist = words
sizes = np.arange(15, 16, 1)
fonts = ['arial', 'times', 'comic']
xshifts = np.arange(-1, 2, 1)
yshifts = np.arange(-1, 2, 1)
colours = [0]
uppers = [0, 1]

#for each word, create num_train + num_val exemplars, then split randomly into train and val.
gc.collect()

imgs, latents_classes,latents_values = [], [], []

for w, word in enumerate(wordlist):
    print(f'Generating images for word: {word}')
    for c,col in enumerate(colours):
        for s,size in enumerate(sizes):
            for x,xshift in enumerate(xshifts):
                for y,yshift in enumerate(yshifts):
                    for f,font in enumerate(fonts):
                        for u,upper in enumerate([0, 1]):
                            #print(size, xshift, yshift, font, upper)
                            img = gen2(savepath=None, index=None, # no saving
                                       text=word, fontname=font, size=size,
                                       xshift=xshift, yshift=yshift, upper=upper)
                            imgs.append(np.array(img))
                            latents_classes.append([w,c,s,x,y,f,u])
                            latents_values.append([w, col, size, xshift, yshift, f, upper])
                
       
np.savez('../data/dwords/dwords', imgs = imgs, latents_classes = latents_classes, latents_values = latents_values)
