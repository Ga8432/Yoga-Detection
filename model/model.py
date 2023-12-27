import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,BatchNormalization,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers.schedules import ExponentialDecay
from PIL import ImageFile



lr_schedual=ExponentialDecay(initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9)

# load data
train='/content/DATASET/TRAIN'
test='/content/DATASET/TEST'
#Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_datagen=ImageDataGenerator(rescale=1./255.)

img_size=(255,255)
batch_size=32
#generator
train_generator = train_datagen.flow_from_directory(train,
                                                    target_size=img_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test,
                                                    target_size=img_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

# model
model=Sequential()
model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),strides=2,padding='valid'))


model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),strides=2,padding='valid'))

model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),strides=2,padding='valid'))

model.add(Conv2D(512,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),strides=2,padding='valid'))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(len(train_generator.class_indices), activation='softmax'))





model.compile(optimizer= Adam(learning_rate=lr_schedual),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)


# train
ImageFile.LOAD_TRUNCATED_IMAGES = True
history = model.fit(train_generator,
                    epochs=50,
                    validation_data=test_generator,
                    callbacks=[early_stopping, reduce_lr, model_checkpoint])