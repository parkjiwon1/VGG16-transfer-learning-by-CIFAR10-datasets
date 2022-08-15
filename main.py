import os
os.add_dll_directory('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin')
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

(train_X,train_Y), (test_X,test_Y) = tf.keras.datasets.cifar10.load_data()

train_X = tf.keras.applications.vgg16.preprocess_input(train_X)
test_X = tf.keras.applications.vgg16.preprocess_input(test_X)

train_x ,test_x = np.zeros((15000,32,32,3)),np.zeros((3000,32,32,3))
train_y,test_y = np.zeros(15000,),np.zeros((3000,))

k = 0
for i in range(len(train_X)):
    if (train_Y[i] == 0):
        train_x[k] = train_X[i]
        train_y[k] = 0
        k += 1
    elif (train_Y[i] == 1):
        train_x[k] = train_X[i]
        train_y[k] = 1
        k += 1
    elif (train_Y[i] == 8):
        train_x[k] = train_X[i]
        train_y[k] = 2
        k += 1

l = 0
for i in range(len(test_X)):
    if (test_Y[i] == 0):
        test_x[l] = test_X[i]
        test_y[l] = 0
        l += 1
    elif (test_Y[i] == 1):
        test_x[l] = test_X[i]
        test_y[l] = 1
        l += 1
    elif (test_Y[i] == 8):
        test_x[l] = test_X[i]
        test_y[l] = 2
        l += 1

model = tf.keras.applications.VGG16(include_top = False)
model.summary()


base_model = tf.keras.applications.VGG16(input_shape = [32,32,3], include_top = False, weights = 'imagenet')

x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64,activation= 'relu')(x)

predictions = tf.keras.layers.Dense(3, activation = 'softmax')(x)

model = tf.keras.Model(inputs = base_model.input,outputs = predictions)

model.summary()

for layer in model.layers[:19]:
    layer.trainable = True
for layer in model.layers[19:]:
    layer.trainable = True

model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(),loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(train_x,train_y,batch_size = 36,epochs = 10,validation_split = 0.25)

# Evaluate the model using the test dataset
model.evaluate(train_x,train_y)
model.evaluate(test_x,test_y)
