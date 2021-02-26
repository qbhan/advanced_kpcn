import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import pyexr
import OpenEXR

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import random
from random import randint

import glob
import os
import time

from preprocess import *
from generate_patch import *
from dataset import *
from utils import *
from train import *
from model import *

# more utility funcitons
# def get_cropped_patches(sample_file, gt_file):
#   data = preprocess_input(sample_file, gt_file)
#   patches = importanceSampling(data)
#   cropped = list(crop(data, tuple(pos), patch_size) for pos in patches)
  
#   return cropped

if __name__ == '__main__':

  torch.multiprocessing.freeze_support()

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  patch_size = 64 # patches are 64x64
  n_patches = 400
  eps = 0.00316

  # data = preprocess_input("sample_data/sample.exr", "sample_data/gt.exr", debug=True)

  # # coordinates of chosen patches
  # patches = importanceSampling(data, debug=False)

  # cropped = list(crop(data, tuple(pos), patch_size) for pos in patches)

  # for f in glob.glob('sample_data/sample*.exr'):
  #   if f == 'sample_data/sample.exr': continue
  #   num = f[len('sample_data/sample'):f.index('.')]
  #   sample_name = 'sample_data/sample{}.exr'.format(num)
  #   gt_name = 'sample_data/gt{}.exr'.format(num)
  #   print(sample_name, gt_name)
  #   cropped += get_cropped_patches(sample_name, gt_name)

  # print(len(cropped))
  # # save the training data
  # for i, v in enumerate(cropped):
  #   torch.save(v, 'data/sample'+str(i+1)+'.pt')
  #   print('SAVED data/sample'+str(i+1)+'.pt')


  # # # Check sizes of data
  # for k, v in cropped[0].items():
  #   print(k, getsize(v))
    
  # print(getsize(cropped) / 1024 / 1024, "MiB")

  # If already saved the data, load them
  cropped = []
  for patch in os.listdir('data/'):
      # print('Loading patch : ' + patch)
      cropped.append(torch.load('data/'+ patch))

  # print('Patches cropped : ' + str(len(cropped)))

  # # Make dataset
  cropped = to_torch_tensors(cropped)
  cropped = send_to_device(cropped, device)
  dataset = KPCNDataset(cropped)


  print("Current device : {}".format(device))


  # mode = 'KPCN' # 'KPCN' or 'DPCN'

  recon_kernel_size = 21

  # # some network parameters
  L = 9 # number of convolutional layers
  n_kernels = 100 # number of kernels in each layer
  kernel_size = 5 # size of kernel (square)

  input_channels = dataset[0]['X_diff'].shape[-1]
  hidden_channels = 100

  print("Input channels:", input_channels)

  # # BHWC -> BCHW


  ddiffuseNet, dspecularNet, dlDiff, dlSpec, dlFinal = train(dataset, input_channels, device, feat=False, mode='DPCN', epochs=40, learning_rate=1e-5)
  torch.save(ddiffuseNet.state_dict(), 'trained_model/ddiffuseNet.pt')
  torch.save(dspecularNet.state_dict(), 'trained_model/dspecularNet.pt')

  kdiffuseNet, kspecularNet, klDiff, klSpec, klFinal = train(dataset, input_channels, device, feat=False, mode='KPCN', epochs=40, learning_rate=1e-5)
  torch.save(kdiffuseNet.state_dict(), 'trained_model/kdiffuseNet.pt')
  torch.save(kspecularNet.state_dict(), 'trained_model/kspecularNet.pt')


  # # test

  # ddiffuseNet = make_net(9, 28, 100, 5, 'DPCN')
  # dspecularNet = make_net(9, 28, 100, 5, 'DPCN')
  # # ddiffuseNet = make_feat_net(9, 28, 100, 5, 'DPCN')
  # # dspecularNet = make_feat_net(9, 28, 100, 5, 'DPCN')
  # ddiffuseNet.load_state_dict(torch.load('trained_model/ddiffuseNet.pt'))
  # diffuseNet = ddiffuseNet.to(device)
  # dspecularNet.load_state_dict(torch.load('trained_model/dspecularNet.pt'))
  # dspecularNet = dspecularNet.to(device)

  # kdiffuseNet = make_net(9, 28, 100, 5, 'KPCN')
  # kspecularNet = make_net(9, 28, 100, 5, 'KPCN')
  # # kdiffuseNet = make_feat_net(9, 28, 100, 5, 'KPCN')
  # # kspecularNet = make_feat_net(9, 28, 100, 5, 'KPCN')
  # kdiffuseNet.load_state_dict(torch.load('trained_model/kdiffuseNet.pt'))
  # kspecularNet.load_state_dict(torch.load('trained_model/kspecularNet.pt'))
  # kdiffuseNet = kdiffuseNet.to(device)
  # kspecularNet = kspecularNet.to(device)

  # eval_data = preprocess_input("sample_data/eval1.exr", "sample_data/evalref1.exr")
  # eval_data = crop(eval_data, (1280//2, 720//2), 300)

  # print('KPCN RESULTS')
  # denoise(kdiffuseNet, kspecularNet, eval_data, device)
  # print('DPCN RESULTS')
  # denoise(ddiffuseNet, dspecularNet, eval_data, device)