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
import csv

from preprocess import *
from generate_patch import *
from dataset import *
from utils import *
from train import *
from model import *


def image_preprocess(patch_size, n_patches):
  cropped = []
  cropped += get_cropped_patches("sample_data/sample.exr", "sample_data/gt.exr", patch_size, n_patches)
  for f in glob.glob('sample_data/sample*.exr'):

    if f == 'sample_data/sample.exr': continue

    num = f[len('sample_data/sample'):f.index('.')]
    sample_name = 'sample_data/sample{}.exr'.format(num)
    gt_name = 'sample_data/gt{}.exr'.format(num)
    print(sample_name, gt_name)
    cropped += get_cropped_patches(sample_name, gt_name, patch_size, n_patches)

  print('Patches cropped : ' + str(len(cropped)))
  print('Saving patches')
  # save the training data
  for i, v in enumerate(cropped):
    torch.save(v, 'data/sample'+str(i+1)+'.pt')
    # print('SAVED data/sample'+str(i+1)+'.pt')

  # Check sizes of data
  for k, v in cropped[0].items():
    print(k, getsize(v))
    
  print(getsize(cropped) / 1024 / 1024, "MiB")


def load_dataset(device):
  # If already saved the data, load them
  cropped = []
  for patch in os.listdir('data/'):
    print('Loading patch : ' + patch)
    cropped.append(torch.load('data/'+ patch))
  
  # load to device and make dataset
  cropped = to_torch_tensors(cropped)
  cropped = send_to_device(cropped, device)
  dataset = KPCNDataset(cropped)
  return dataset


def train_dpcn(dataset, n_layers, n_kernels, size_kernel, in_channels, hidden_channels, save_dir=None):
  pass
  ddiffuseNet, dspecularNet, dlDiff, dlSpec, dlFinal = train(dataset, input_channels, device, feat=False, mode='DPCN', epochs=40, learning_rate=1e-5)
  torch.save(ddiffuseNet.state_dict(), 'trained_model/ddiffuseNet.pt')
  torch.save(dspecularNet.state_dict(), 'trained_model/dspecularNet.pt')
  with open('plot/dpcn.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(dlDiff)
    writer.writerow(dlSpec)
  return ddiffuseNet, dspecularNet, dlDiff, dlSpec, dlFinal



def train_kpcn(dataset, n_layers, n_kernels, size_kernel, in_channels, hidden_channels, recon_kernel_size, save_dir=None):
  pass
  kdiffuseNet, kspecularNet, klDiff, klSpec, klFinal = train(dataset, input_channels, device, feat=False, mode='KPCN', epochs=40, learning_rate=1e-5)
  torch.save(kdiffuseNet.state_dict(), 'trained_model/kdiffuseNet.pt')
  torch.save(kspecularNet.state_dict(), 'trained_model/kspecularNet.pt')
  with open('plot/kpcn.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(klDiff)
    writer.writerow(klSpec)
  return kdiffuseNet, kspecularNet, klDiff, klSpec, klFinal


def train_feat_dpcn(dataset, n_layers, n_kernels, size_kernel, in_channels, hidden_channels, save_dir=None):
  pass
  ddiffuseNet, dspecularNet, dlDiff, dlSpec, dlFinal = train(dataset, input_channels, device, feat=True, mode='DPCN', epochs=40, learning_rate=1e-5)
  torch.save(ddiffuseNet.state_dict(), 'trained_model/feat_ddiffuseNet.pt')
  torch.save(dspecularNet.state_dict(), 'trained_model/feat_dspecularNet.pt')
  with open('plot/feat_dpcn.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(dlDiff)
    writer.writerow(dlSpec)
  return ddiffuseNet, dspecularNet, dlDiff, dlSpec, dlFinal


def train_feat_kpcn(dataset, n_layers, n_kernels, size_kernel, in_channels, hidden_channels, recon_kernel_size, save_dir=None):
  pass
  kdiffuseNet, kspecularNet, klDiff, klSpec, klFinal = train(dataset, input_channels, device, feat=True, mode='KPCN', epochs=40, learning_rate=1e-5)
  torch.save(kdiffuseNet.state_dict(), 'trained_model/feat_kdiffuseNet.pt')
  torch.save(kspecularNet.state_dict(), 'trained_model/feat_kspecularNet.pt')
  with open('plot/feat_kpcn.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(klDiff)
    writer.writerow(klSpec)
  return kdiffuseNet, kspecularNet, klDiff, klSpec, klFinal


def test_model(diffuseNet, specularNet, device, data=None):
  pass
  diffuseNet.to(device)
  specularNet.to(device)
  if not data:
    eval_data = preprocess_input("sample_data/eval1.exr", "sample_data/evalref1.exr")
    eval_data = crop(eval_data, (1280//2, 720//2), 300)
  denoise(diffuseNet, specularNet, eval_data, device)

if __name__ == '__main__':

  torch.multiprocessing.freeze_support()

  # # some network parameters
  L = 9 # number of convolutional layers
  n_kernels = 100 # number of kernels in each layer
  kernel_size = 5 # size of kernel (square)
  recon_kernel_size = 21

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("Current device : {}".format(device))

  patch_size = 64 # patches are 64x64
  n_patches = 400
  eps = 0.00316

  # image_preprocess(patch_size, n_patches)
  dataset = load_dataset(device)

  input_channels = dataset[0]['X_diff'].shape[-1]  #28
  # input_channels = 28
  hidden_channels = 100


  # TRAIN

  ddiffuseNet, dspecularNet, dlDiff, dlSpec, dlFinal = train_dpcn(dataset, L, n_kernels, kernel_size, input_channels, hidden_channels)
  plot_training(dlDiff, dlSpec, 'ddiffuse')
  # kdiffuseNet, kspecularNet, klDiff, klSpec, klFinal = train_kpcn(dataset, L, n_kernels, kernel_size, input_channels, hidden_channels)

  # feat_ddiffuseNet, feat_dspecularNet, feat_dlDiff, feat_dlSpec, feat_dlFinal = train_feat_dpcn(dataset, L, n_kernels, kernel_size, input_channels, hidden_channels, recon_kernel_size)
  # feat_kdiffuseNet, feat_kspecularNet, feat_klDiff, feat_klSpec, feat_klFinal = train_feat_kpcn(dataset, L, n_kernels, kernel_size, input_channels, hidden_channels, recon_kernel_size)


  # TEST

  # ddiffuseNet = make_net(9, 28, 100, 5, 'DPCN')
  # dspecularNet = make_net(9, 28, 100, 5, 'DPCN')
  # ddiffuseNet.load_state_dict(torch.load('trained_model/ddiffuseNet.pt'))
  # dspecularNet.load_state_dict(torch.load('trained_model/dspecularNet.pt'))
  # test_model(ddiffuseNet, dspecularNet, device)

  # feat_ddiffuseNet = make_feat_net(9, 28, 100, 5, 'DPCN')
  # feat_dspecularNet = make_feat_net(9, 28, 100, 5, 'DPCN')
  # feat_ddiffuseNet.load_state_dict(torch.load('trained_model/feat_ddiffuseNet.pt'))
  # feat_dspecularNet.load_state_dict(torch.load('trained_model/feat_dspecularNet.pt'))
  # test_model(feat_ddiffuseNet, feat_dspecularNet, device)

  # kdiffuseNet = make_net(9, 28, 100, 5, 'KPCN')
  # kspecularNet = make_net(9, 28, 100, 5, 'KPCN')
  # kdiffuseNet.load_state_dict(torch.load('trained_model/kdiffuseNet.pt'))
  # kspecularNet.load_state_dict(torch.load('trained_model/kspecularNet.pt'))
  # test_model(kdiffuseNet, kspecularNet, device)

  # feat_kdiffuseNet = make_feat_net(9, 28, 100, 5, 'KPCN')
  # feat_kspecularNet = make_feat_net(9, 28, 100, 5, 'KPCN')
  # feat_kspecularNet.load_state_dict(torch.load('trained_model/feat_kspecularNet.pt'))
  # feat_kdiffuseNet.load_state_dict(torch.load('trained_model/feat_kdiffuseNet.pt'))
  # test_model(feat_kdiffuseNet, feat_kspecularNet, device) 
  