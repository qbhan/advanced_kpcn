import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import numpy as np

from utils import *
from model import *
from visualize import *

L = 9 # number of convolutional layers
n_kernels = 100 # number of kernels in each layer
kernel_size = 5 # size of kernel (square)

# input_channels = dataset[0]['X_diff'].shape[-1]
hidden_channels = 100

permutation = [0, 3, 1, 2]
eps = 0.00316

def validation(diffuseNet, specularNet, dataloader, criterion, device, mode='KPCN'):
  pass
  lossDiff = 0
  lossSpec = 0
  lossFinal = 0
  for batch_idx, data in enumerate(dataloader):
    X_diff = data['X_diff'].permute(permutation).to(device)
    Y_diff = data['Reference'][:,:,:,:3].permute(permutation).to(device)

    outputDiff = diffuseNet(X_diff)
    if mode == 'KPCN':
      X_input = crop_like(X_diff, outputDiff)
      outputDiff = apply_kernel(outputDiff, X_input, device)

    Y_diff = crop_like(Y_diff, outputDiff)
    lossDiff += criterion(outputDiff, Y_diff).item()

    X_spec = data['X_spec'].permute(permutation).to(device)
    Y_spec = data['Reference'][:,:,:,3:6].permute(permutation).to(device)
    
    outputSpec = specularNet(X_spec)
    if mode == 'KPCN':
      X_input = crop_like(X_spec, outputSpec)
      outputSpec = apply_kernel(outputSpec, X_input, device)

    Y_spec = crop_like(Y_spec, outputSpec)
    lossSpec += criterion(outputSpec, Y_spec).item()

    # calculate final ground truth error
    albedo = data['origAlbedo'].permute(permutation).to(device)
    albedo = crop_like(albedo, outputDiff)
    outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

    Y_final = data['finalGt'].permute(permutation).to(device)
    Y_final = crop_like(Y_final, outputFinal)
    lossFinal += criterion(outputFinal, Y_final).item()

  return lossDiff/len(dataloader), lossSpec/len(dataloader), lossFinal/len(dataloader)

def train(dataset, input_channels, device, feat=False, validDataSet=None, mode='KPCN', epochs=20, learning_rate=1e-4, show_images=False):
  print('TRAINING WITH VALIDDATASET : {}'.format(validDataSet))
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                          shuffle=True, num_workers=4)

  if validDataSet is not None:
    validDataloader = torch.utils.data.DataLoader(validDataSet, batch_size=4, num_workers=4)

  # instantiate networks
  if feat:
    diffuseNet = make_feat_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
    specularNet = make_feat_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
  else:
    diffuseNet = make_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
    specularNet = make_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)

  print(diffuseNet, "CUDA:", next(diffuseNet.parameters()).is_cuda)
  print(specularNet, "CUDA:", next(specularNet.parameters()).is_cuda)
  
  criterion = nn.L1Loss()

  optimizerDiff = optim.Adam(diffuseNet.parameters(), lr=learning_rate)
  optimizerSpec = optim.Adam(specularNet.parameters(), lr=learning_rate)

  accuLossDiff = 0
  accuLossSpec = 0
  accuLossFinal = 0
  
  lDiff = []
  lSpec = []
  lFinal = []
  valLDiff = []
  valLSpec = []
  valLFinal = []

  writer = SummaryWriter('runs/'+mode+'_2')
  total_epoch = 0


  import time

  start = time.time()

  for epoch in range(epochs):
    for i_batch, sample_batched in enumerate(dataloader):
      #print(i_batch)

      # get the inputs
      X_diff = sample_batched['X_diff'].permute(permutation).to(device)
      Y_diff = sample_batched['Reference'][:,:,:,:3].permute(permutation).to(device)

      # zero the parameter gradients
      optimizerDiff.zero_grad()

      # forward + backward + optimize
      outputDiff = diffuseNet(X_diff)

      # print(outputDiff.shape)

      if mode == 'KPCN':
        X_input = crop_like(X_diff, outputDiff)
        outputDiff = apply_kernel(outputDiff, X_input, device)

      Y_diff = crop_like(Y_diff, outputDiff)

      lossDiff = criterion(outputDiff, Y_diff)
      lossDiff.backward()
      optimizerDiff.step()

      # get the inputs
      X_spec = sample_batched['X_spec'].permute(permutation).to(device)
      Y_spec = sample_batched['Reference'][:,:,:,3:6].permute(permutation).to(device)

      # zero the parameter gradients
      optimizerSpec.zero_grad()

      # forward + backward + optimize
      outputSpec = specularNet(X_spec)

      if mode == 'KPCN':
        X_input = crop_like(X_spec, outputSpec)
        outputSpec = apply_kernel(outputSpec, X_input, device)

      Y_spec = crop_like(Y_spec, outputSpec)

      lossSpec = criterion(outputSpec, Y_spec)
      lossSpec.backward()
      optimizerSpec.step()

      # calculate final ground truth error
      with torch.no_grad():
        albedo = sample_batched['origAlbedo'].permute(permutation).to(device)
        albedo = crop_like(albedo, outputDiff)
        outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

        # if False:#i_batch % 500:
        #   print("Sample, denoised, gt")
        #   sz = 3
        #   orig = crop_like(sample_batched['finalInput'].permute(permutation), outputFinal)
        #   orig = orig.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
        #   show_data(orig, figsize=(sz,sz), normalize=True)
        #   img = outputFinal.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
        #   show_data(img, figsize=(sz,sz), normalize=True)
        #   gt = crop_like(sample_batched['finalGt'].permute(permutation), outputFinal)
        #   gt = gt.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
        #   show_data(gt, figsize=(sz,sz), normalize=True)

        Y_final = sample_batched['finalGt'].permute(permutation).to(device)

        Y_final = crop_like(Y_final, outputFinal)

        lossFinal = criterion(outputFinal, Y_final)

        accuLossFinal += lossFinal.item()

      accuLossDiff += lossDiff.item()
      accuLossSpec += lossSpec.item()

      writer.add_scalar('total loss', accuLossFinal if accuLossFinal != float('inf') else 0, epoch * len(dataloader) + i_batch)
      writer.add_scalar('diffuse loss', accuLossDiff if accuLossDiff != float('inf') else 0, epoch * len(dataloader) + i_batch)
      writer.add_scalar('specular loss', accuLossSpec if accuLossSpec != float('inf') else 0, epoch * len(dataloader) + i_batch)
      
    print('VALIDATION WORKING!')
    validLossDiff, validLossSpec, validLossFinal = validation(diffuseNet, specularNet, validDataloader, criterion, device, mode)
    writer.add_scalar('Valid total loss', np.log(np.abs(validLossFinal)) if accuLossFinal != float('inf') else 0, epoch * len(dataloader) + i_batch)
    writer.add_scalar('Valid diffuse loss', validLossDiff if accuLossDiff != float('inf') else 0, epoch * len(dataloader) + i_batch)
    writer.add_scalar('Valid specular loss', validLossSpec if accuLossSpec != float('inf') else 0, epoch * len(dataloader) + i_batch)

    print("Epoch {}".format(epoch + 1))
    print("LossDiff: {}".format(accuLossDiff))
    print("LossSpec: {}".format(accuLossSpec))
    print("LossFinal: {}".format(accuLossFinal))
    print("ValidLossDiff: {}".format(validLossDiff))
    print("ValidLossSpec: {}".format(validLossSpec))
    print("ValidLossFinal: {}".format(validLossFinal))

    lDiff.append(accuLossDiff)
    lSpec.append(accuLossSpec)
    lFinal.append(accuLossFinal)
    valLDiff.append(validLossDiff)
    valLSpec.append(validLossSpec)
    valLFinal.append(validLossFinal)

    total_epoch += 1
    # if len(valLFinal) > 10 and valLFinal[-1] >= valLFinal[-2]:
    #   print('EARLY STOPPING!')
    #   break
    
    accuLossDiff = 0
    accuLossSpec = 0
    accuLossFinal = 0

  writer.close()
  print('Finished training in mode, {} with epoch {}'.format(mode, total_epoch))
  print('Took', time.time() - start, 'seconds.')
  
  return diffuseNet, specularNet, lDiff, lSpec, lFinal


def unsqueeze_all(d):
  for k, v in d.items():
    d[k] = torch.unsqueeze(v, dim=0)
  return d


def denoise(diffuseNet, specularNet, data, device, debug=False):
  with torch.no_grad():
    out_channels = diffuseNet[len(diffuseNet)-1].out_channels
    mode = 'DPCN' if out_channels == 3 else 'KPCN'
    criterion = nn.L1Loss()
    
    if debug:
      print("Out channels:", out_channels)
      print("Detected mode", mode)
    
    # make singleton batch
    data = send_to_device(to_torch_tensors(data), device)
    if len(data['X_diff'].size()) != 4:
      data = unsqueeze_all(data)
    
    print(data['X_diff'].size())
    
    X_diff = data['X_diff'].permute(permutation).to(device)
    Y_diff = data['Reference'][:,:,:,:3].permute(permutation).to(device)

    # forward + backward + optimize
    outputDiff = diffuseNet(X_diff)

    # print(outputDiff.shape)

    if mode == 'KPCN':
      X_input = crop_like(X_diff, outputDiff)
      outputDiff = apply_kernel(outputDiff, X_input, device)

    Y_diff = crop_like(Y_diff, outputDiff)

    lossDiff = criterion(outputDiff, Y_diff).item()

    # get the inputs
    X_spec = data['X_spec'].permute(permutation).to(device)
    Y_spec = data['Reference'][:,:,:,3:6].permute(permutation).to(device)

    # forward + backward + optimize
    outputSpec = specularNet(X_spec)

    if mode == 'KPCN':
      X_input = crop_like(X_spec, outputSpec)
      outputSpec = apply_kernel(outputSpec, X_input, device)

    Y_spec = crop_like(Y_spec, outputSpec)

    lossSpec = criterion(outputSpec, Y_spec).item()

    # calculate final ground truth error
    albedo = data['origAlbedo'].permute(permutation).to(device)
    albedo = crop_like(albedo, outputDiff)
    outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

    if debug:
      writer = SummaryWriter('runs/results_'+mode)
      print("Sample, denoised, gt")
      sz = 15
      orig = crop_like(data['finalInput'].permute(permutation), outputFinal)
      orig = orig.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
      show_data(orig, figsize=(sz,sz), normalize=True)
      print(orig.shape)
      writer.add_image('noisy', np.transpose(orig, (1,2,0)))
      img = outputFinal.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
      show_data(img, figsize=(sz,sz), normalize=True)
      writer.add_image('denoised', img)
      gt = crop_like(data['finalGt'].permute(permutation), outputFinal)
      gt = gt.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
      show_data(gt, figsize=(sz,sz), normalize=True)
      writer.add_image('clean', img)

    Y_final = data['finalGt'].permute(permutation).to(device)

    Y_final = crop_like(Y_final, outputFinal)
    
    lossFinal = criterion(outputFinal, Y_final).item()
    
    # if debug:
    print("LossDiff:", lossDiff)
    print("LossSpec:", lossSpec)
    print("LossFinal:", lossFinal)