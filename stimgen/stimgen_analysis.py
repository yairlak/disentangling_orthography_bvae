#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 09:19:19 2022

@author: aakash
"""
from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import subprocess, shlex, shutil, io, os, random, gc, time
from tqdm import tqdm
import pickle

import numpy as np
words = ['a','b','c','d','e','f','g','h','i','j',
         'k','l','m','n','o','p','q','r','s','t',
         'u','v','w','x','y','z']


####
def gen2(savepath=None, text = 'text', index=1, mirror=False,
        invert=False, fontname='Arial', W = 64, H = 64, size=24,
        xshift=0, yshift=0, upper=0, show=None):
    if upper:
        text = text.upper()
    if invert:
        text = text[::-1]
    img = Image.new("RGB", (W,H), color = (255, 255, 255))
    fnt = ImageFont.truetype(fontname+'.ttf', size)
    draw = ImageDraw.Draw(img)
    w, h = fnt.getsize(text)
    draw.text((xshift + (W-w)/2, yshift + (H-h)/2), text, font=fnt, fill='black')
    if savepath:
        img.save(os.path.join(savepath, f'{text}{index}.png'))
    else:
        return img



######################
def CreateWordSet(path_out='../../../data/letters_centred/'):
    
    #define words, sizes, fonts
    wordlist = words
    
    for word in wordlist:
        path = os.path.join(path_out, word)
        os.makedirs(path, exist_ok=True); n = 0
        
        print(f'Generating images for word: {word}')
        gen2(savepath= path, index=n, # no saving
             text=word, fontname='arial', size=24,
             xshift=0, yshift=0, upper=1)

    return 'done'


CreateWordSet()

