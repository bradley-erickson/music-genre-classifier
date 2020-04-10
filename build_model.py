# -*- coding: utf-8 -*-
"""
@author: Brad Erickson
@class: CS 495: Research Seminar
@date: Spring 2020
@title: Model builder
"""

# imports
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

## keras imports
from keras.models import Sequential, model_from_json
from keras.layers import Dense, BatchNormalization
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

## sklearn imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# methods
## data
def transform_data(data):
    """ return a yeo-johnson scaled version of our data """
    pt = preprocessing.PowerTransformer(method='yeo-johnson')
    data = pt.fit_transform(data)
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


## model
def initialize_model(input_shape, classes):
    """ initialize the model parameters and layers """
    
    cnn = Sequential()
    
    cnn.add(Dense(64, activation = 'relu', input_shape = input_shape))
    cnn.add(BatchNormalization())
    cnn.add(Dense(32, activation = 'relu'))
    cnn.add(Dense(16, activation = 'relu'))
    cnn.add(Dense(classes, activation = 'softmax'))
    
    cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #cnn.summary()
    
    return cnn


## train
## plot
def plot_model(info):
    """ display a plot of the model """
    
    plt.figure()
    plt.plot(info.history['acc'])
    plt.plot(info.history['val_acc'])
    plt.title('Accuracy of model')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
    
#    plt.plot(info.history['loss'])
#    plt.plot(info.history['val_loss'])
#    plt.title('Loss of Model')
#    plt.ylabel('Loss')
#    plt.xlabel('Epoch')
#    plt.legend(['Training', 'Validation'], loc='upper right')
#    plt.show()
    
def load_model_info(file):
    """ load the model and weights from a file """
    
    print ('Reading in: ' + file)
    model_file = open(file + '.json', 'r')
    loaded_model = model_file.read()
    model_file.close()
    model = model_from_json(loaded_model)    
    model.load_weights(file + '.h5')
    return model

# run model
def build_model(data, model_parameters, model_name):
    """ transforms data and trains model """
    # prep data
    x = data.loc[:, data.columns != 'genre']
    y = data['genre'].values
    classes = list(data['genre'].unique())
    x_trans = transform_data(x)
    
    for i in range(len(y)):
        y[i] = classes.index(y[i])
    y = to_categorical(y)
    print (classes)
    
    
    
    valid_acc_list = []
    valid_max = 0
    x_split, x_test, y_split, y_test = train_test_split(x_trans, y, train_size=0.8, stratify=y)
    for i in range(20):
        print ('\n==================================================================')
        print ('Iteration: ', str(i))
        print ('==================================================================\n')
        x_train, x_valid, y_train, y_valid = train_test_split(x_split, y_split, train_size=0.75, stratify=y_split)
        
        
        # create model
        input_shape = x_trans[0].shape
        model = initialize_model(input_shape, 10)
        
        fitted = model.fit(x_train, y_train, verbose=0, epochs=model_parameters[1], batch_size=model_parameters[0], validation_data=(x_valid, y_valid))
        
        print ('Train accuracy: ', np.max(fitted.history['acc']))
        print ('Valid accuracy: ', np.max(fitted.history['val_acc']))
        #plot_model(fitted)
        
        valid_acc = np.max(fitted.history['val_acc'])
        valid_acc_list.append(valid_acc)
        if (valid_acc > valid_max):
            print ('Old: ', valid_max, ' New: ', valid_acc)
            valid_max = valid_acc
            model_json = model.to_json()
            with open(model_name+".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(model_name+".h5")
        
    print ('Average validation accuracy: ', str(np.mean(valid_acc_list)))
    loaded_model = load_model_info(model_name)
    loaded_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
    y_pred = loaded_model.predict(x_test)
    print ('Test accuracy: ', test_acc)
    
    predict_class = np.argmax(y_pred, axis=1)
    test_class = np.argmax(y_test, axis=1)
    conf = confusion_matrix(test_class, predict_class)
    print (conf)
    plot_conf = pd.DataFrame(conf, index=classes, columns=classes)
    plt.figure()
    cmap = plt.get_cmap('Blues_r')
    heat = sn.heatmap(plot_conf, annot=True, annot_kws={"size": 16}, cmap=cmap, cbar=False, vmin=0, vmax=20)
    title = ''
    if (model_name == 'spotify'):
        title = 'Third-party confusion matrix'
    else:
        title = 'Self-extracted confusion matrix'
    heat.set(xlabel='Predicted', ylabel='Actual', title=title)
    plt.tight_layout()
    plt.show()
    


# Go
# parameters will be used as [batch_size, num_epochs]
parameters = [64, 20]
spotify = pd.read_csv('spotify_data.csv')
extract = pd.read_csv('extracted_data.csv')

build_model(spotify, parameters, 'spotify')
build_model(extract, parameters, 'extract')
