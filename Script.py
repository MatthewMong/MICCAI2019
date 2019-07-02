""" Data Loading and Processing Scripts go here."""
import tensorflow as tf
import os
from PIL import Image
import numpy as np

#Should perform a batch extraction to filter all photos based on maps, will be completed soon
def batch_extract(png_location,jpg_location):
    print('blank right now, check back later')

#this function extracts the image based on the maps we are given, it serves as a helper function to batch_extract
def image_extract(jpg,png):
    img_map=Image.open(png)
    with open(jpg,'rb') as img_file:
        img_file.seek(163)
        a=img_file.read(2)
        height = (a[0] << 8) + a[1]
        a=img_file.read(2)
        width=(a[0]<<8)+a[1]
    img=Image.open(jpg)
    img_map_arr=np.array(img_map,dtype='uint8')
    img_arr=np.array(img,dtype='uint8')
    print(img_arr[12,4])
    pls=Image.fromarray(img_arr)
    pls.show()
    abz=img_map_arr.flatten()
    abz2=img_arr.reshape(width*height,3)
    abz2copy=np.empty(abz2.shape)
    for x in abz.nonzero():
        abz2copy[x]=abz2[x]
    abz2copy=abz2copy.reshape(height,width,3)
    img_arr=abz2copy
    final=Image.fromarray(img_arr.astype('uint8'))
    final.show()


#image_extract('D:/imgs/imgs/slide003_core004.jpg','D:/imgs/Maps1/maps1/slide003_core004_classimg_nonconvex.png')