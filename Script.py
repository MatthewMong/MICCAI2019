""" Data Loading and Processing Scripts go here."""
import glob
import os
from io import BytesIO

import numpy as np
import pandas as pandas
import tensorflow as tf
from PIL import Image
from tensorflow import keras

DEBUG = False
tf.enable_eager_execution()


# The main function you need is tf_data_process(), it takes png files location, and jpg files location and returns a tensorflow dataset.
# please set the compression rate of the photos by yourself by changing image = tf.image.resize(image, [192, 192]) to whatever number you want

# Examples of how to use:
# batch_extract('D:/imgs/Test/Maps','D:/imgs/Test/Jpegs')
# image_extract('D:/imgs/imgs/slide003_core004.jpg','D:/imgs/Maps1/maps1/slide003_core004_classimg_nonconvex.png')


# Should perform a batch extraction to filter all photos based on maps, feed it the directory with your png maps, and your jpg locations
def batch_extract(png_location, jpg_location):
    dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    jpgs = os.listdir(jpg_location)
    for image in jpgs:
        value = glob.glob(png_location + '/' + image.strip('.jpg') + '_classimg_nonconvex.png')
        if (len(value) == 1):
            eximage = image_extract(jpg_location + '/' + image,
                                    png_location + '/' + image.strip('.jpg') + '_classimg_nonconvex.png')
            i = 0
            while i < len(eximage):
                dict[eximage[i]].append(eximage[i + 1])
                i = i + 2
            print(dict)
        elif (len(value) == 0):
            print("no file with that name")
        else:
            print('error, more than one file with same name exists')
    return dict


def batch_extract_to_Arr(png_location, jpg_location):
    Labels = []
    Pictures = []
    jpgs = os.listdir(jpg_location)
    for image in jpgs:
        value = glob.glob(png_location + '/' + image.strip('.jpg') + '_classimg_nonconvex.png')
        if (len(value) == 1):
            eximage = image_extract(jpg_location + '/' + image,
                                    png_location + '/' + image.strip('.jpg') + '_classimg_nonconvex.png')
            i = 0
            while i < len(eximage):
                Labels.append(eximage[i])
                i = i + 1
                Pictures.append(preprocess_image(convertToJpeg(eximage[i])))
                i = i + 1
        elif (len(value) == 0):
            print("no file with that name")
        else:
            print('error, more than one file with same name exists')
    return [Labels, Pictures]


def keras_batch_extract_to_Arr(png_location, jpg_location):
    Labels = []
    Pictures = []
    jpgs = os.listdir(jpg_location)
    for image in jpgs:
        value = glob.glob(png_location + '/' + image.strip('.jpg') + '_classimg_nonconvex.png')
        if (len(value) == 1):
            eximage = image_extract(jpg_location + '/' + image,
                                    png_location + '/' + image.strip('.jpg') + '_classimg_nonconvex.png')
            i = 0
            while i < len(eximage):
                (Labels.append(eximage[i]))
                i = i + 1
                (Pictures.append(preprocess_image(convertToJpeg(eximage[i]))))
                i = i + 1
        elif (len(value) == 0):
            print("no file with that name")
        else:
            print('error, more than one file with same name exists')
    Pictures = np.array(Pictures)
    Labels = np.array(Labels)
    return Labels, Pictures


# this function extracts the image based on the maps we are given, it serves as a helper function to batch_extract, it does not differentiate between different levels within an image
# both inputs should be the locations of your png and jpg files
def image_extract(jpg, png):
    print(jpg)
    img_map = Image.open(png)
    returnval = []
    with open(jpg, 'rb') as img_file:
        img_file.seek(163)
        a = img_file.read(2)
        height = (a[0] << 8) + a[1]
        a = img_file.read(2)
        width = (a[0] << 8) + a[1]
    img = Image.open(jpg)
    img_map_arr = np.array(img_map, dtype='uint8')
    img_arr = np.array(img, dtype='uint8')
    abz = img_map_arr.flatten()
    abz2 = img_arr.reshape(width * height, 3)
    abz2copy = np.empty(abz2.shape)
    value = np.unique(abz)
    value = value[value > 0]
    if np.array_equal(np.unique(value), []) or value.size == 0 or np.array_equal(value, []):
        returnval.append(0)
        returnval.append(img)
    elif (np.size(value) > 1):
        for values in value:
            abzcopy = abz.copy()
            abzcopy[abzcopy != values] = 0
            zeros = abzcopy.nonzero()
            abz2copy = np.empty(abz2.shape)
            for x in zeros:
                abz2copy[x] = abz2[x]
            abz2copy = abz2copy.reshape(height, width, 3)
            img_arr = abz2copy
            final = Image.fromarray(img_arr.astype('uint8'))
            if DEBUG == True:
                final.show()
            returnval.append(values)
            returnval.append(final)
    else:
        for x in abz.nonzero():
            abz2copy[x] = abz2[x]
        abz2copy = abz2copy.reshape(height, width, 3)
        img_arr = abz2copy
        final = Image.fromarray(img_arr.astype('uint8'))
        if DEBUG == True:
            final.show()
        returnval.append(value[0])
        returnval.append(final)
    return returnval


def tf_data_process(png_location, jpg_location):
    arr = batch_extract_to_Arr(png_location, jpg_location)
    pictures = tf.data.Dataset.from_tensor_slices(arr[1])
    labels = tf.data.Dataset.from_tensor_slices(arr[0])
    dataset = tf.data.Dataset.zip({"pictures": pictures, "labels": labels})
    return dataset


def pandas_data_process(png_location, jpg_location):
    dataframe = pandas.DataFrame(data=batch_extract(png_location, jpg_location))
    print('done loading')
    return dataframe


def keras_data_process(png_location, jpg_location):
    labels, pictures = keras_batch_extract_to_Arr(png_location, jpg_location)
    print('done loading')
    # pictures=np.array(pictures)
    print('done pictures')
    # labels=np.array(labels)
    print('done labels')
    return pictures, labels


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [512, 512])
    image = tf.image.rgb_to_grayscale(image)
    image = image.numpy()
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def convertToJpeg(im):
    with BytesIO() as f:
        im.save(f, format='JPEG')
        return f.getvalue()


pictures, labels = keras_data_process('D:/imgs/Test/Maps', 'D:/imgs/Test/Jpegs')
test_pics, test_labels = keras_data_process('D:/imgs/Test/TestMaps', 'D:/imgs/Test/TestJpegs')

# test=tf_data_process('D:/imgs/Test/TestMaps', 'D:/imgs/Test/TestJpegs')
model = keras.Sequential([
    # keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu', input_shape=(256, 256, 1)),
    # keras.layers.BatchNormalization(),
    # keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    # keras.layers.BatchNormalization(axis=3),
    # keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
    # keras.layers.Dropout(0.3),
    # keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu', padding='same'),
    # keras.layers.BatchNormalization(),
    # keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    # keras.layers.BatchNormalization(axis=3),
    # keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
    # keras.layers.Dropout(0.3),
    # keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same'),
    # keras.layers.BatchNormalization(),
    # keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    # keras.layers.BatchNormalization(axis=3),
    # keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
    # keras.layers.Dropout(0.5),
    # keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', padding='same'),
    # keras.layers.BatchNormalization(),
    # keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    # keras.layers.BatchNormalization(axis=3),
    # keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
    # keras.layers.Dropout(0.5),
    # keras.layers.Conv2D(512, kernel_size=(3, 3), strides=1, activation='relu', padding='same'),
    # keras.layers.BatchNormalization(),
    # keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
    # keras.layers.BatchNormalization(axis=3),
    # keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
    # keras.layers.Dropout(0.5),
    # keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=1, activation='relu', padding='same'),
    # keras.layers.BatchNormalization(),
    # keras.layers.Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same'),
    # keras.layers.BatchNormalization(axis=3),
    # keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
    # keras.layers.Dropout(0.5),
    # keras.layers.Flatten(),
    # keras.layers.Dense(units=512, activation='relu'),
    # keras.layers.BatchNormalization(),
    # keras.layers.Dropout(0.5),
    # keras.layers.Dense(units=128, activation='relu'),
    # keras.layers.Dropout(0.25),
    # keras.layers.Dense(7, activation='softmax')
    keras.layers.SeparableConv2D(32, (3, 3), padding="same",
                                 input_shape=(512, 512, 1)),
    keras.layers.Activation("relu"),
    keras.layers.BatchNormalization(axis=-1),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.SeparableConv2D(64, (3, 3), padding="same"),
    keras.layers.Activation("relu"),
    keras.layers.BatchNormalization(axis=-1),
    keras.layers.SeparableConv2D(64, (3, 3), padding="same"),
    keras.layers.Activation("relu"),
    keras.layers.BatchNormalization(axis=-1),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(256),
    keras.layers.Activation("relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(7),
    keras.layers.Activation("softmax")
    # keras.layers.Flatten(input_shape=(256,256,1)),
    # #keras.layers.SeparableConv2D(32,(3,3), padding="same"),
    # keras.layers.Activation("relu"),
    # #keras.layers.MaxPool2D(pool_size=(2,2)),
    # keras.layers.Dropout(.25),
    # #keras.layers.Convolution1D(filters=5,kernel_size=3),
    # keras.layers.Dense(128, activation=tf.nn.relu),
    # keras.layers.Dense(7, activation=tf.nn.softmax),

])

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(pictures, labels, epochs=12, use_multiprocessing=True)
# model.fit(pictures,labels, epochs=9)
# model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_pics, test_labels,use_multiprocessing=True)

print('Test accuracy:', test_acc)
print(test_labels)
print(model.predict(test_pics,verbose=1,use_multiprocessing=True))