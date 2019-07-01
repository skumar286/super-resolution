import cv2,glob
import os
import argparse
parser = argparse.ArgumentParser(description='control RDNSR')
parser.add_argument('--to', action="store",dest="tryout", default=200)
parser.add_argument('--ep', action="store",dest="epochs", default=100)
parser.add_argument('--bs', action="store",dest="batch_size", default=20)
parser.add_argument('--lr', action="store",dest="learning_rate", default=0.0001)
parser.add_argument('--gpu', action="store",dest="gpu", default=3)
parser.add_argument('--chk',action="store",dest="chk",default=-1)
parser.add_argument('--sample',action='store',dest="sample",default=512)
# parser.add_argument('--test_sample', action="store",dest="test_sample",default=190)
parser.add_argument('--scale', action='store' , dest = 'scale' , default = 2)
parser.add_argument('--data', action='store' , dest = 'folder' , default = '.')
parser.add_argument('--test_image', action = 'store' , dest = 'test_image' , default = 'butterfly.png')
parser.add_argument('--test_only' , action = 'store', dest = 'test_only' , default = False)
parser.add_argument('--zoom' , action = 'store' , dest = 'zoom' , default = False)
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
chk = int(values.chk)
zoom = values.zoom
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras.applications.vgg19 import VGG19
import sys
import numpy as np
import matplotlib.pyplot as plt
from DATA_BUILDER import DATA
from keras.models import Model
from keras.layers import Input,MaxPool2D,Deconvolution2D ,Convolution2D , Add, Dense , AveragePooling2D , UpSampling2D , Reshape , Flatten , Subtract , Concatenate
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import backend as k
from keras.applications.vgg19 import preprocess_input
from keras.utils import multi_gpu_model

import tensorflow as tf
from keras.utils import plot_model
from subpixel import Subpixel
from PIL import Image
import cv2
test_dir=glob.glob('./Set5/*.*',recursive=True)

def perceptual_loss(y_true, y_pred):  # y_true and y_pred's pixels are scaled between 0 to 255
    y_true = preprocess_input(y_true)
    y_pred = preprocess_input(y_pred)
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(128,128,3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return k.mean(k.square(loss_model(y_true)-loss_model(y_pred)))



def PSNRLossnp(y_true,y_pred):
        return 10* np.log(255*2 / (np.mean(np.square(y_pred - y_true))))

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


class SRResnet:
    def L1_loss(self , y_true , y_pred):
        return k.mean(k.abs(y_true - y_pred))
    
    #def L1_plus_PSNR_loss(self,y_true, y_pred):
        #return self.L1_loss(y_true , y_pred) - 0.0001*PSNRLoss(y_true , y_pred)
    
    def RDBlocks(self,x,name , count = 6 , g=32):
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
        
    def visualize(self):
            plot_model(self.model, to_file='model.png' , show_shapes = True)
    
    def get_model(self):
        return self.model
    
    def __init__(self , channel = 3 , lr=0.0001 , patch_size=32 , RDB_count=20 ,chk = -1 , scale = 2):
            self.channel_axis = 3
            inp = Input(shape = (patch_size , patch_size , channel))
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
            
            output = Convolution2D(filters =3 , kernel_size=(3,3) , strides=(1 , 1) , padding='same')(output)

            model = Model(inputs=inp , outputs = output)
            adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=lr/2, amsgrad=False)

            model.compile(loss=perceptual_loss, optimizer=adam , metrics=[PSNRLoss,SSIM])
            
            if chk >=0 :
                print("loading existing weights !!!")
                model.load_weights('model_'+str(scale)+'x_iter'+str(chk)+'.h5')
            self.model = model
            
    def fit(self , X , Y ,batch_size=16 , epoch = 100 ):
            # with tf.device('/gpu:'+str(gpu)):    
            hist = self.model.fit(X, Y , batch_size = batch_size , verbose =1 , epochs=epoch)
            return hist.history


if __name__ == '__main__':
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

		

	net = SRResnet(lr = learning_rate,scale = scale , chk = chk)
	if not test_only:
		net.visualize()
		net.get_model().summary()

	if not test_only:
		for i in range(chk+1,tryout):
		    print("tryout no: ",i)   
		    
		    samplev = np.random.random_integers(0 , x.shape[0]-1 , sample)
		    net.fit(x[samplev] , DATA.training_patches_Y[samplev] , batch_size , epochs )
		    
		    net.get_model().save_weights('model_'+str(scale)+'x_iter'+str(i)+'.h5')
		    g = net.get_model().predict(np.array(p))
		    gen = DATA.reconstruct(g , r , c , scale=1)
		    Image.fromarray(gen).save("test_"+str(scale)+"x_gen_"+str(i)+".png")
		    print("Reconstruction Gain:", PSNRLossnp(img , gen))
	else :
	    sum,cnt,tot_psnr=0,0,0
	    for i in range(len(test_dir)):
	    	cnt+=1
	    	img=test_dir[i]
	    	img = cv2.imread(img)
	    	img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
	    	r = DATA.patch_size - img.shape[0] % DATA.patch_size
	    	c = DATA.patch_size - img.shape[1] % DATA.patch_size
	    	img = np.pad(img, [(0,r),(0,c),(0,0)] , 'constant')
	    	lr_img = cv2.resize(img , (int(img.shape[1]/scale),int(img.shape[0]/scale)) ,cv2.INTER_CUBIC)
	    	hr_img_bi = cv2.resize(lr_img ,(int(img.shape[1]),int(img.shape[0])),cv2.INTER_CUBIC)
	    	p , r , c = DATA.patchify(lr_img,scale=scale) 
	    	g = net.get_model().predict(np.array(p))
	    	gen = DATA.reconstruct(g , r , c , scale=1)
	    	nm=test_dir[i].split("/")
	    	nm=nm[-1].split(".")[0]
	    	Image.fromarray(gen).save(nm+str(scale)+"x_gen_.png")
	    	print("Reconstruction Gain:", PSNRLossnp(img , gen))
	    	tot_psnr+=PSNRLossnp(img , gen)
	    avg_psnr=tot_psnr/cnt
	    print("Avg PSNR",avg_psnr)
