# Contains loss/accuracy metrics, mask to RLE, RLE to mask, and post-processing module.

import os
import json
import gc
import cv2
import keras
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from keras.metrics import binary_crossentropy

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true,y_pred):
    return 1-dice_coef(y_true,y_pred)

def BCE_loss(y_true, y_pred):
    return (binary_crossentropy(y_true, y_pred))

def bce_dice(y_true, y_pred):
    return BCE_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

def post_process(probability, threshold=0.5, min_size=3000):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = probability >= 0.5
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
    return predictions


def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(rle, input_shape):
    width, height = input_shape[:2]
    mask= np.zeros( width*height ).astype(np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
    return mask.reshape(height, width).T

def build_masks(rles, input_shape):
    depth = len(rles)
    masks = np.zeros((*input_shape, depth))
    for i, rle in enumerate(rles):
        if type(rle) is str:
            masks[:, :, i] = rle2mask(rle, input_shape)
    return masks

def build_rles(masks):
    width, height, depth = masks.shape
    #rles = [mask2rle(post_process(masks[:, :, i])) for i in range(depth)]
    rles = [mask2rle((masks[:, :, i])) for i in range(depth)]
    return rles