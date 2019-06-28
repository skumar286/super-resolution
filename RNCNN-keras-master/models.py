# -*- coding: utf-8 -*-

import numpy as np
from keras.utils.vis_utils import plot_model
#import tensorflow as tf
from keras.models import *
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Lambda,Subtract,add,MaxPool2D,Concatenate
#from keras import backend as K


def RDBlocks(x,count = 6 , g=32):
    li = [x]
    #pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu' , name = name+'_conv1')(x)
    pas = Conv2D(filters=g, kernel_size=(3,3),strides=(1, 1),padding="same")(x)
    pas = BatchNormalization()(pas)
    pas = Activation("relu")(pas)

    for i in range(2 , count+1):
        li.append(pas)
        out =  Concatenate(axis = 3)(li) # conctenated out put
        #pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_conv'+str(i))(out)
        pas = Conv2D(filters=g, kernel_size=(3,3),strides=(1, 1), padding="same")(out)
        pas = BatchNormalization()(pas)
        pas = Activation("relu")(pas)   
    # feature extractor from the dense net
    li.append(pas)
    out = Concatenate(axis = 3)(li)
    #feat = Convolution2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='same',activation='relu' , name = name+'_Local_Conv')(out)
    feat = Conv2D(filters=64, kernel_size=(3,3),strides=(1, 1), padding="same")(out)
    feat = BatchNormalization()(feat)
    feat = Activation("relu")(feat)     
    feat = Subtract()([x , feat])
    return feat


def DnCNN():
    inp = Input(shape = (None , None , 1))
    #pass1 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(inp)
    x = Conv2D(filters=64, kernel_size=(3,3),strides=(1, 1), kernel_initializer="he_normal",padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)    
    #pass2 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(pass1)
    x1 = Conv2D(filters=64, kernel_size=(3,3),strides=(1, 1), kernel_initializer="he_normal",padding="same")(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)    

    
    RDB = RDBlocks(x1)
    RDBlocks_list = [RDB,]
    for i in range(2,10):
        RDB = RDBlocks(RDB)
        RDBlocks_list.append(RDB)
    out = Concatenate(axis = 3)(RDBlocks_list)
    out = Conv2D(filters=1 , kernel_size=(1,1) , strides=(1,1) , padding='same')(out)
    x = Subtract()([inp , out])
    model = Model(inputs=inp, outputs=x)
    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model