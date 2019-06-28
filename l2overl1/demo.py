import argparse, os,glob
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="PyTorch VDSR Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="/media/DiskE/vdsr2x4x/custom loss/l2overl1/checkpoint/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


    
def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")


model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
data_hr=glob.glob('/media/DiskE/vdsr2x4x/data/Urban100/*.*',recursive=True)
data_bic=glob.glob('/media/DiskE/vdsr2x4x/data/Urban100bicubic/*.*',recursive=True)
data_hr=sorted(data_hr)
data_bic=sorted(data_bic)
for dt in range(len(data_hr)):
	im_gt_ycbcr = imread(data_hr[dt], mode="YCbCr")
	im_b_ycbcr = imread(data_bic[dt], mode="YCbCr")
	    
	im_gt_y = im_gt_ycbcr[:,:,0].astype(float)
	im_b_y = im_b_ycbcr[:,:,0].astype(float)


	im_input = im_b_y/255.

	im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

	if cuda:
	    model = model.cuda()
	    im_input = im_input.cuda()
	else:
	    model = model.cpu()

	start_time = time.time()
	out = model(im_input)
	elapsed_time = time.time() - start_time

	out = out.cpu()

	im_h_y = out.data[0].numpy().astype(np.float32)

	im_h_y = im_h_y * 255.
	im_h_y[im_h_y < 0] = 0
	im_h_y[im_h_y > 255.] = 255.

	im_h = colorize(im_h_y[0,:,:], im_b_ycbcr)
	im_gt = Image.fromarray(im_gt_ycbcr, "YCbCr").convert("RGB")
	im_b = Image.fromarray(im_b_ycbcr, "YCbCr").convert("RGB")
	print(data_hr[dt],data_bic[dt])
	c=data_hr[dt].split('/')
	c=c[-1].split(".")[0]
	im_h.save("/media/DiskE/vdsr2x4x/custom loss/l2overl1/Urban100gen/"+c+ "_gn"+str(opt.scale)+".png")
