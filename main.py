import os
os.add_dll_directory('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin')
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

(train_X,train_Y), (test_X,test_Y) = tf.keras.datasets.cifar10.load_data()

idx = np.where((train_Y == 0)|(train_Y == 1)|(train_Y==8))
idx2 = np.where((test_Y == 0)|(test_Y == 1)|(test_Y==8))

train_x = train_X[idx[0],:]; test_x = test_X[idx2[0],:]
train_y = train_Y[idx]; test_y = test_Y[idx2]

idx = np.where(train_y == 8); idx2 = np.where(test_y ==8)
train_y[idx] = 2; test_y[idx2] = 2

train_x = tf.keras.applications.vgg16.preprocess_input(train_x)
test_x = tf.keras.applications.vgg16.preprocess_input(test_x)


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
