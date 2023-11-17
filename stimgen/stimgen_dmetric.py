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
import gc, random,os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile
from itertools import product
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from tqdm import tqdm
# import pickle, subprocess, shlex, shutil, io, , , time

words = ['a', 'b', 'c', 'd', 'e',
         'f', 'g', 'h', 'i', 'j',
         'k', 'l', 'm', 'n', 'o',
         'p', 'q', 'r', 's', 't',
         'u', 'v', 'w', 'x', 'y',
         'z']


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


def generate_ngrams(unigrams, n=1):
    if not (n in [1, 2, 3]):
        print(f'n must be 1, 2 or 3')
        return []

    if n == 2:
        unigrams = ["a", "k", "l", "m",  "v"]
        # unigrams = ["a", "d", "h", "i", "m", "n", "t"] # awraval

    res = []
    if n >= 1:
        res += unigrams
    if n >= 2:
        t = product(unigrams, repeat=2)
        res += ["".join(x) for x in t]
    if n == 3:
        t = product(unigrams, repeat=3)
        res += ["".join(x) for x in t]

    return res


def add_class(x):
    return zip(range(len(x)), x)


def CreateWordSet(path_out = '../data/dletters/dletters',
                  ngrams   = 1,
                  n_train  = 100_000):

    #define words, sizes, fonts
    wordlist = generate_ngrams(words, ngrams)
    sizes = np.arange(15, 31, 3)
    fonts = ['arial', 'times', 'comic']
    xshifts = np.arange(-8, 8, 1)
    yshifts = np.arange(-8, 8, 1)
    colours = [0]
    uppers   = [0, 1]

    gc.collect()

    imgs, latents_classes,latents_values, latents_values_str = [], [], [], []
    latents_names = ["words", "colours", "sizes",
                     "xshifts", "yshifts", "fonts", "uppers"]
    latents_size  = [len(wordlist), len(colours), len(sizes),
                     len(xshifts), len(yshifts), len(fonts), len(uppers)]

    all_stim = product(add_class(colours), add_class(sizes),
                  add_class(xshifts), add_class(yshifts),
                  add_class(fonts), add_class(uppers))
    all_stim = list(all_stim)

    for w, word in enumerate(wordlist):
        # n_thisWord = n_total//len(wordlist)
        # selected   = random.sample(all, n_thisWord) # select some variations for this word
        # print(f'Generating {n_thisWord} images for word: {word}')
        selected = all_stim
        print(f'Generating images for word: {word}')
        for ((c, col), (s, size),
             (x, xshift), (y, yshift),
             (f, font), (u, upper)) in selected:

                img = gen2(savepath=None, index=None, # no saving
                           text=word, fontname=font, size=size,
                           xshift=xshift, yshift=yshift, upper=upper)

                imgs.append(np.array(img))
                latents_classes.append([w, c, s, x, y, f, u])
                latents_values.append([w, col, size, xshift, yshift, f, upper])
                latents_values_str.append([word, col, size, xshift, yshift, font, upper])

    os.makedirs(path_out, exist_ok=True)
    f_name = f'/dletters_n{ngrams}'
    np.savez(path_out + f_name,
             imgs=imgs, latents_classes=latents_classes, latents_values=latents_values,
             latents_names=latents_names, latents_size=latents_size, latents_values_str=latents_values_str)


ngrams = 2
CreateWordSet(f'data/dwords/', ngrams)