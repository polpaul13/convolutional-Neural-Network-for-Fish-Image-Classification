# coding: utf-8

# In[1]:


# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# loading the directories 
training_dir = "C:/FISH2/training"
validation_dir = "C:/FISH2/testing"

#get number of files
image_files = glob(training_dir + '/*/*.*')
valid_image_files = glob(validation_dir + '/*/*.*')


# In[2]:


# get number of classes (types of fish)
folders = glob(training_dir + '/*')
num_classes = len(folders)
print ('Total Classes = ' + str(num_classes))


# In[3]:


# importing the libraries for model training
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications import VGG16


IMAGE_SIZE = [64, 64]   

# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
# input_shape = (64,64,3) as
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  

# this will exclude the initial layers from training phase as there are already been trained.
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(128, activation = 'relu')(x)  
x = Dense(num_classes, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

model = Model(inputs = vgg.input, outputs = x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[4]:


model.summary()


# In[13]:


# importing the libraries image Augmentation
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

training_datagen = ImageDataGenerator(
                                    rescale=1./255,   # all pixel values will be between 0 an 1
                                    shear_range=0.2, 
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    preprocessing_function=preprocess_input)

validation_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)

training_generator = training_datagen.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')


# In[14]:


# The labels are stored in class_indices in dictionary form. 
# checking the labels
training_generator.class_indices


# In[15]:


training_images = 37458
validation_images = 4006

history = model.fit_generator(training_generator,
                   steps_per_epoch = 5000,   
                   epochs = 10, 
                   validation_data = validation_generator,
                   validation_steps = 1500)  


# In[16]:


print ('Training Accuracy = ' + str(history.history['acc']))
print ('Validation Accuracy = ' + str(history.history['val_acc']))


# In[17]:


# Plot the train and validation loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Plot the train and validation accuracies
plt.plot(history.history['acc'], label='train acc')
plt.plot(history.history['val_acc'], label='val acc')
plt.legend()
plt.show()
