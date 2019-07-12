""" Data Loading and Processing Scripts go here."""
import os
import glob
from PIL import Image
import numpy as np
DEBUG=True

#Examples of how to use:
#batch_extract('D:/imgs/Test/Maps','D:/imgs/Test/Jpegs')
#image_extract('D:/imgs/imgs/slide003_core004.jpg','D:/imgs/Maps1/maps1/slide003_core004_classimg_nonconvex.png')


#Should perform a batch extraction to filter all photos based on maps, feed it the directory with your png maps, and your jpg locations
def batch_extract(png_location,jpg_location):
    dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6:[]}
    jpgs=os.listdir(jpg_location)
    for image in jpgs:
        value=glob.glob(png_location+'/'+image.strip('.jpg')+'_classimg_nonconvex.png')
        if(len(value)==1):
            eximage=image_extract(jpg_location+'/'+image,png_location+'/'+image.strip('.jpg')+'_classimg_nonconvex.png')
            i=0
            while i<len(eximage):
                dict[eximage[i]].append(eximage[i+1])
                i=i+2
            print(dict)
        elif(len(value)==0):
            print("no file with that name")
        else:
            print('error, more than one file with same name exists')
    return dict

#this function extracts the image based on the maps we are given, it serves as a helper function to batch_extract, it does not differentiate between different levels within an image
#both inputs should be the locations of your png and jpg files
def image_extract(jpg,png):
    img_map=Image.open(png)
    returnval=[]
    with open(jpg,'rb') as img_file:
        img_file.seek(163)
        a=img_file.read(2)
        height = (a[0] << 8) + a[1]
        a=img_file.read(2)
        width=(a[0]<<8)+a[1]
    img=Image.open(jpg)
    img_map_arr=np.array(img_map,dtype='uint8')
    img_arr=np.array(img,dtype='uint8')
    #print(img_arr[12,4])
    #pls=Image.fromarray(img_arr)
    #pls.show()
    abz=img_map_arr.flatten()
    abzcopy=abz.copy()
    abz2=img_arr.reshape(width*height,3)
    abz2copy=np.empty(abz2.shape)
    value=np.unique(abz)
    value=value[value>0]
    if np.array_equal(np.unique(value),[])or value.size==0 or np.array_equal(value,[]):
        returnval.append(0)
        returnval.append(img)
    elif(np.size(value)>1):
        for values in value:
            abzcopy=abz.copy()
            abzcopy[abzcopy!=values]=0
            zeros=abzcopy.nonzero()
            abz2copy = np.empty(abz2.shape)
            for x in zeros:
                abz2copy[x]=abz2[x]
            abz2copy=abz2copy.reshape(height,width,3)
            img_arr=abz2copy
            final=Image.fromarray(img_arr.astype('uint8'))
            if DEBUG==True:
                final.show()
            returnval.append(values)
            returnval.append(final)
    else:
        for x in abz.nonzero():
            abz2copy[x]=abz2[x]
        abz2copy=abz2copy.reshape(height,width,3)
        img_arr=abz2copy
        final=Image.fromarray(img_arr.astype('uint8'))
        if DEBUG==True:
            final.show()
        print(jpg)
        returnval.append(value[0])
        returnval.append(final)
    return returnval

batch_extract('D:/imgs/Maps1/maps1','D:/imgs/imgs')