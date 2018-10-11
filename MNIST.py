
#Import the various gym, keras, numpy and libraries we will require


import numpy as np
import matplotlib.pyplot as plt
import random
import time
import struct

from keras.layers import Flatten, Dense
from keras.models import Sequential, Model
from keras import optimizers

def build_model(num_input_nodes, num_output_nodes, lr, size):
    
    model = Sequential()
    
    model.add(Dense(size[0], input_shape = (32,num_input_nodes), activation = 'relu'))
    
    for i in range(1,len(size)):
        model.add(Dense(size[i], activation = 'relu'))
    
    model.add(Dense(num_output_nodes, activation = 'softmax')) 
    
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
    
    model.compile(loss = 'mse', optimizer = adam)
    
    model.summary()
    
    return model
        
# The MNIST data is provided in a file format that needs to be read before the data can be fed to the neural network.     
def read_data(image_filename, label_filename):
    
    #Create arrays to store labels and pixel data
    label_array = []
    img_array = []
    
    # Load everything in some numpy arrays
    with open(label_filename, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(image_filename, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    print(len(lbl))

    for i in range(100):
        label_array.append(lbl[i])
        img_array.append(np.ndarray.flatten(img[i]))
    
    return(label_array, img_array)

if __name__ == '__main__':
    
    num_input_nodes = 784   #Number of input pixels
    num_output_nodes = 10   #Number of output pixels
    
    #Build Model
    model = build_model(num_input_nodes, num_output_nodes, 0.001, [16,16])
    
    
    train_labels, train_images = read_data('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
    print('labels = ',np.shape(train_labels))
    print('images = ', np.shape(train_images))
    
    model.fit(train_images[0], train_labels[0], batch_size=32, epochs=1, verbose=1)
    
        
    

    

