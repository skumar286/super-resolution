import cv2
import os

'''

This is the standard DENSE NET with Global and Local Feature fusion + learning and 
upscalling using subpixel convolution. That is just better way of space filling by
learnable parameters in the deconvolution layer which has gradient during
backpropagation instead of the stadard 0 gradient.

Now we add the infamous PERCEPTUAL Loss in a more controlled fashion so that
content is maximum and color is kept but the texture i.e the strokes are transfered

We will try with bothh VGG and INCEPTION as loss factor
This one will rely on stanard VGG

'''


'''
train where the train data is build somewhere and you are pointing to it
python main.py --ep 1 --to 200 --bs 16 --lr 0.0001 --gpu 1 --sample 16384 --scale 4 --data ../Data --test_image 0016.png --l1_factor 0.9 --lambda_content 0

testing on the data

python main.py --test_only True --gpu 1 --chk 111 --scale 2 --test_image some.png

use the above to run the file. this is the orinal configuration as per the paper

'''

import argparse
parser = argparse.ArgumentParser(description='control RDNSR')
parser.add_argument('--to', action="store",dest="tryout", default=200)
parser.add_argument('--ep', action="store",dest="epochs", default=100)
parser.add_argument('--bs', action="store",dest="batch_size", default=32)
parser.add_argument('--lr', action="store",dest="learning_rate", default=0.01)
parser.add_argument('--gpu', action="store",dest="gpu", default=-1)
parser.add_argument('--chk',action="store",dest="chk",default=-1)
parser.add_argument('--sample',action='store',dest="sample",default=512)
# parser.add_argument('--test_sample', action="store",dest="test_sample",default=190)
parser.add_argument('--scale', action='store' , dest = 'scale' , default = 2)
parser.add_argument('--data', action='store' , dest = 'folder' , default = '.')
parser.add_argument('--test_image', action = 'store' , dest = 'test_image' , default = 'baby.png')
parser.add_argument('--test_only' , action = 'store', dest = 'test_only' , default = False)
parser.add_argument('--zoom',action='store',dest = 'zoom' , default = False)

values = parser.parse_args()
learning_rate = float(values.learning_rate)
batch_size = int(values.batch_size)
epochs = int(values.epochs)
tryout = int(values.tryout)
gpu=int(values.gpu)
sample = int(values.sample)
# test_sample = int(values.test_sample)
scale = int(values.scale)
folder = values.folder
test_only = values.test_only

zoom = values.zoom

os.environ["CUDA_VISIBLE_DEVICES"]="0"

chk = int(values.chk)

import sys
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow import keras as K
from DATA_BUILDER import DATA
from keras.models import Model
from keras.layers import Input,MaxPool2D,Deconvolution2D ,Convolution2D , Add, Dense , AveragePooling2D , UpSampling2D , Reshape , Flatten , Subtract , Concatenate , Lambda
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import backend as k
from keras.applications import vgg16
from keras.utils import multi_gpu_model
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from keras.utils import plot_model
from subpixel import Subpixel
from PIL import Image
import cv2


CHANNEL = 3

DATA = DATA(folder = folder , patch_size = int(scale * 32))

out_patch_size =  DATA.patch_size 
inp_patch_size = int(out_patch_size/ scale)
if not test_only:
    DATA.load_data(folder=folder)
    if scale == 2:
        x = DATA.training_patches_2x
    elif scale == 4:
        x = DATA.training_patches_4x
    elif scale == 8:
        x = DATA.training_patches_8x



vgg_inp = Input(shape = (32*scale, 32*scale, 3) , name='vgg_net_input')
vgg = vgg16.VGG16(input_tensor = vgg_inp, input_shape=(32*scale,32*scale,3) , weights='imagenet' , include_top=False)
for l in vgg.layers: l.trainable=False

## Content loss
content_loss_layer = 'block3_conv3'

content_loss_output = [vgg.get_layer(content_loss_layer).output]

vgg_content= Model(inputs=vgg_inp , outputs = content_loss_output)


def PSNRLossnp(y_true,y_pred):
        return 10* np.log(255*2 / (np.mean(np.square(y_pred - y_true))))

def perceptual_loss(y_true,y_pred):
#     mse=K.losses.mean_squared_error(y_true,y_pred)
    y_t=vgg_content(y_true)
    y_p=vgg_content(y_pred)
    loss=tf.losses.mean_squared_error(y_t,y_p)
    return loss

def SSIM( y_true,y_pred):
    u_true = k.mean(y_true)
    u_pred = k.mean(y_pred)
    var_true = k.var(y_true)
    var_pred = k.var(y_pred)
    std_true = k.sqrt(var_true)
    std_pred = k.sqrt(var_pred)
    c1 = k.square(0.01*7)
    c2 = k.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def PSNRLoss(y_true, y_pred):
        return 10* k.log(255**2 /(k.mean(k.square(y_pred - y_true))))

class PERCEPTUAL_RDN:
    
    def RDBlocks(self,x,name , count = 6 , g=32):
        ## 6 layers of RDB block
        ## this thing need to be in a damn loop for more customisability
            li = [x]
            pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu' , name = name+'_conv1')(x)
            
            for i in range(2 , count+1):
                li.append(pas)
                out =  Concatenate(axis = self.channel_axis)(li) # conctenated out put
                pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_conv'+str(i))(out)
            
            # feature extractor from the dense net
            li.append(pas)
            out = Concatenate(axis = self.channel_axis)(li)
            feat = Convolution2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='same',activation='relu' , name = name+'_Local_Conv')(out)
            
            feat = Add()([feat , x])
            return feat


             
    def get_inference_model(self):
            return self.inference_model
    

    def __init__(self , channel = 3 , lr=0.0001 , patch_size=32 , RDB_count=20 ,chk = -1 , scale = 4):
            self.channel_axis = 3
            inp = Input(shape = (patch_size , patch_size , channel) , name='rdn_input')

            pass1 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(inp)

            pass2 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(pass1)

            
            RDB = self.RDBlocks(pass2 , 'RDB1')
            RDBlocks_list = [RDB,]

            for i in range(2,RDB_count+1):
                RDB = self.RDBlocks(RDB ,'RDB'+str(i))
                RDBlocks_list.append(RDB)
            
            out = Concatenate(axis = self.channel_axis)(RDBlocks_list)
            out = Convolution2D(filters=64 , kernel_size=(1,1) , strides=(1,1) , padding='same')(out)
            out = Convolution2D(filters=64 , kernel_size=(3,3) , strides=(1,1) , padding='same')(out)

            output = Add()([out , pass1])
            
            if scale >= 2:
                output = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(output)
            if scale >= 4:
                output = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(output)
            if scale >= 8:
                output = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(output)
            
            output = Convolution2D(filters =3 , kernel_size=(3,3) , strides=(1 , 1) , padding='same' , name='l1_output')(output)

            self.inference_model = Model(inputs=inp , outputs = output)

            if not test_only:
                adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None)
                self.inference_model.compile(loss=perceptual_loss, optimizer=adam , metrics=[PSNRLoss,SSIM] )

            
            if chk >=0 :
                print("loading existing weights !!!")
                self.inference_model.load_weights('model_'+str(scale)+'x_iter'+str(chk)+'.h5')
    

    def fit(self , X , Y ,batch_size=16 , epoch = 100 ):
            zero = np.zeros(Y.shape[0])    
            hist = self.inference_model.fit(X,Y, batch_size = batch_size , validation_split=0.3,verbose =1 , epochs=epoch)
            return hist.history
    

net = PERCEPTUAL_RDN(lr = learning_rate,scale = scale , chk = chk)

image_name = values.test_image
try:
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
except cv2.error as e:
    print("Bad image path check the name or path !!")
    exit()

r = DATA.patch_size - img.shape[0] % DATA.patch_size
c = DATA.patch_size - img.shape[1] % DATA.patch_size
img = np.pad(img, [(0,r),(0,c),(0,0)] , 'constant')
Image.fromarray(img).save("test_image_padded.png")
lr_img = cv2.resize(img , (int(img.shape[1]/scale),int(img.shape[0]/scale)) ,cv2.INTER_CUBIC)
Image.fromarray(lr_img).save("test_"+str(scale)+"x_lr_padded.png")
hr_img_bi = cv2.resize(lr_img ,(int(img.shape[1]),int(img.shape[0])),cv2.INTER_CUBIC)
Image.fromarray(hr_img_bi).save("test_"+str(scale)+"x_hr_bicubic_padded.png")
p , r , c = DATA.patchify(lr_img,scale=scale) 
p , r , c = DATA.patchify(lr_img,scale=scale) 

if not test_only:
    for i in range(chk+1,tryout):
        print("tryout no: ",i)   
        
        samplev = np.random.random_integers(0 , x.shape[0]-1 , sample)
        Y = DATA.training_patches_Y[samplev]
        net.fit(x[samplev] , Y , batch_size , epochs )
        
        net.get_inference_model().save_weights('model_'+str(scale)+'x_iter'+str(i)+'.h5')
        g = net.get_inference_model().predict(np.array(p))
        gen = DATA.reconstruct(g , r , c , scale=1)
        #gen[gen > 255] = 255
        #gen[gen < 0] = 0
        Image.fromarray(gen).save("test_"+str(scale)+"x_gen_"+str(i)+".png")
        print("Reconstruction Gain:", PSNRLossnp(img , gen))
else:

    if zoom:
        gz , r2 , c2 = DATA.patchify(img , scale = scale)
        gz = net.get_inference_model().predict(np.array(gz))      
        genz = DATA.reconstruct(gz , r2 , c2 , scale = 1)
        Image.fromarray(genz).save("test_image_zoomed_"+str(scale)+"x.png")     
    else:
        g = net.get_inference_model().predict(np.array(p))
        gen = DATA.reconstruct(g , r , c , scale=1)     
        Image.fromarray(gen).save("test_"+str(scale)+"x_gen_.png")
        print("Reconstruction Gain:", PSNRLossnp(img , gen))
