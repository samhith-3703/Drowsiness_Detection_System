import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# Image dimensions and batch size
img_height, img_width = 128, 128
batch_size = 32


# Paths to  directories
train_dir = r'D:\VS\Capstone_project\Trials\Driver_Drowsiness_Dataset\train'
val_dir = r'D:\VS\Capstone_project\Trials\Driver_Drowsiness_Dataset\validation'
test_dir = r'D:\VS\Capstone_project\Trials\Driver_Drowsiness_Dataset\test'


# Data augmentation and preprocessing for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  
    rotation_range=20,    
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,        
    horizontal_flip=True  
)




val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0    
)




train_data = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'  
)




val_data = val_test_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)


# Load testing data (no augmentation)
test_data = val_test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  
)


print("Data preprocessing complete.")



