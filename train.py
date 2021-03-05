import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import csv

from utils import *
from model import *
from visualize import *
from dataset import *

# L = 9 # number of convolutional layers
# n_kernels = 100 # number of kernels in each layer
# kernel_size = 5 # size of kernel (square)

# # input_channels = dataset[0]['X_diff'].shape[-1]
# hidden_channels = 100

permutation = [0, 3, 1, 2]
# eps = 0.00316

parser = argparse.ArgumentParser(description='Train the model')

'''
Needed parameters
1. Data & Model specifications
device : which device will the data & model should be loaded
mode : which kind of model should it train
input_channel : input channel
hidden_channel : hidden channel
num_layer : number of layers / depth of models
'''
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--mode', default='kpcn')
parser.add_argument('--num_layers', default=9, type=int)
parser.add_argument('--input_channels', default=28, type=int)
parser.add_argument('--hidden_channels', default=100, type=int)
parser.add_argument('--kernel_size', default=5, type=int)

'''
2. Preprocessing specifications
eps
'''
parser.add_argument('--eps', default=0.00316, type=float)

'''
3. Training Specification
val : should it perform validation
early_stopping : should it perform early stopping
trainset : dataset for training
valset : dataset for validation
lr : learning rate
epoch : epoch
criterion : which loss function should it use
'''
parser.set_defaults(do_val=False)
parser.add_argument('--do_val', dest='do_val', action='store_true')
parser.set_defaults(do_early_stopping=False)
parser.add_argument('--do_early_stopping', dest='do_early_stopping', action='store_true')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--loss', default='L1')




def validation(diffuseNet, specularNet, dataloader, eps, criterion, device, mode='kpcn'):
  pass
  lossDiff = 0
  lossSpec = 0
  lossFinal = 0
  for batch_idx, data in enumerate(dataloader):
    X_diff = data['X_diff'].permute(permutation).to(device)
    Y_diff = data['Reference'][:,:,:,:3].permute(permutation).to(device)

    outputDiff = diffuseNet(X_diff)
    # if mode == 'KPCN':
    if 'kpcn' in mode:
      X_input = crop_like(X_diff, outputDiff)
      outputDiff = apply_kernel(outputDiff, X_input, device)

    Y_diff = crop_like(Y_diff, outputDiff)
    lossDiff += criterion(outputDiff, Y_diff).item()

    X_spec = data['X_spec'].permute(permutation).to(device)
    Y_spec = data['Reference'][:,:,:,3:6].permute(permutation).to(device)
    
    outputSpec = specularNet(X_spec)
    # if mode == 'KPCN':
    if 'kpcn' in mode:
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

def train(mode, device, trainset, validset, eps, L, input_channels, hidden_channels, kernel_size, epochs, learning_rate, loss, do_early_stopping, show_images=False):
  # print('TRAINING WITH VALIDDATASET : {}'.format(validset))
  dataloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                          shuffle=True, num_workers=4)

  if validset is not None:
    validDataloader = torch.utils.data.DataLoader(validset, batch_size=4, num_workers=4)

  # instantiate networks
  print(L, input_channels, hidden_channels, kernel_size, mode)
  if 'feat' in mode:
    diffuseNet = make_feat_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
    specularNet = make_feat_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
  else:
    print(mode)
    diffuseNet = make_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
    specularNet = make_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)

  print(diffuseNet, "CUDA:", next(diffuseNet.parameters()).is_cuda)
  print(specularNet, "CUDA:", next(specularNet.parameters()).is_cuda)

  if loss == 'L1':
    criterion = nn.L1Loss()
  else:
    print('Loss Not Supported')
    return

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
      # print(X_diff.shape, Y_diff.shape)
      # zero the parameter gradients
      optimizerDiff.zero_grad()

      # forward + backward + optimize
      outputDiff = diffuseNet(X_diff)

      # print()

      # if mode == 'KPCN':
      if 'kpcn' in mode:
        X_input = crop_like(X_diff, outputDiff)
        outputDiff = apply_kernel(outputDiff, X_input, device)

      Y_diff = crop_like(Y_diff, outputDiff)
      # print(outputDiff.shape, Y_diff.shape)

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

      # if mode == 'KPCN':
      if 'kpcn' in mode:
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
    validLossDiff, validLossSpec, validLossFinal = validation(diffuseNet, specularNet, validDataloader, eps, criterion, device, mode)
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
    if do_early_stopping and len(valLFinal) > 10 and valLFinal[-1] >= valLFinal[-2]:
      print('EARLY STOPPING!')
      break
    
    accuLossDiff = 0
    accuLossSpec = 0
    accuLossFinal = 0

  writer.close()
  print('Finished training in mode, {} with epoch {}'.format(mode, total_epoch))
  print('Took', time.time() - start, 'seconds.')
  
  return diffuseNet, specularNet, lDiff, lSpec, lFinal


def load_dataset(device, val=False):
  # If already saved the data, load them
  cropped = []
  if not val:
    for patch in os.listdir('data/'):
      print('Loading patch : ' + patch)
      cropped.append(torch.load('data/'+ patch))
  else:
    for patch in os.listdir('val/'):
      print('Loading patch : ' + patch)
      cropped.append(torch.load('val/'+ patch))
    
    # load to device and make dataset
  cropped = to_torch_tensors(cropped)
  cropped = send_to_device(cropped, device)
  dataset = KPCNDataset(cropped)
  return dataset


def train_dpcn(dataset, validset, device, eps, n_layers, size_kernel, in_channels, hidden_channels, epochs, lr, loss, do_early_stopping, save_dir=None):
  pass
  ddiffuseNet, dspecularNet, dlDiff, dlSpec, dlFinal = train('dpcn', device, dataset, validset, eps, n_layers, in_channels, hidden_channels, size_kernel, epochs, lr, loss, do_early_stopping)
  torch.save(ddiffuseNet.state_dict(), 'trained_model/ddiffuseNet.pt')
  torch.save(dspecularNet.state_dict(), 'trained_model/dspecularNet.pt')
  with open('plot/dpcn.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(dlDiff)
    writer.writerow(dlSpec)
    writer.writerow(dlFinal)
  return ddiffuseNet, dspecularNet, dlDiff, dlSpec, dlFinal



def train_kpcn(dataset, validset, device, eps, n_layers, size_kernel, in_channels, hidden_channels, epochs, lr, loss, do_early_stopping, save_dir=None):
  pass
  kdiffuseNet, kspecularNet, klDiff, klSpec, klFinal = train('kpcn', device, dataset, validset, eps, n_layers, in_channels, hidden_channels, size_kernel, epochs, lr, loss)
  torch.save(kdiffuseNet.state_dict(), 'trained_model/kdiffuseNet.pt')
  torch.save(kspecularNet.state_dict(), 'trained_model/kspecularNet.pt')
  with open('plot/kpcn.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(klDiff)
    writer.writerow(klSpec)
    writer.writerow(klFinal)
  return kdiffuseNet, kspecularNet, klDiff, klSpec, klFinal


def train_feat_dpcn(dataset, validset, device, eps, n_layers, size_kernel, in_channels, hidden_channels, epochs, lr, loss, do_early_stopping, save_dir=None):
  pass
  ddiffuseNet, dspecularNet, dlDiff, dlSpec, dlFinal = train('feat_dpcn', device, dataset, validset, eps, n_layers, in_channels, hidden_channels, size_kernel, epochs, lr, loss)
  torch.save(ddiffuseNet.state_dict(), 'trained_model/feat_ddiffuseNet.pt')
  torch.save(dspecularNet.state_dict(), 'trained_model/feat_dspecularNet.pt')
  with open('plot/feat_dpcn.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(dlDiff)
    writer.writerow(dlSpec)
    writer.writerow(dlFinal)
  return ddiffuseNet, dspecularNet, dlDiff, dlSpec, dlFinal


def train_feat_kpcn(dataset, validset, device, eps, n_layers, size_kernel, in_channels, hidden_channels, epochs, lr, loss, do_early_stopping, save_dir=None):
  pass
  kdiffuseNet, kspecularNet, klDiff, klSpec, klFinal = train('feat_kpcn', device, dataset, validset, eps, n_layers, in_channels, hidden_channels, size_kernel, epochs, lr, loss)
  torch.save(kdiffuseNet.state_dict(), 'trained_model/feat_kdiffuseNet.pt')
  torch.save(kspecularNet.state_dict(), 'trained_model/feat_kspecularNet.pt')
  with open('plot/feat_kpcn.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(klDiff)
    writer.writerow(klSpec)
    writer.writerow(klFinal)
  return kdiffuseNet, kspecularNet, klDiff, klSpec, klFinal


def main():
  args = parser.parse_args()
  print(args)
  
  # Load Train & Validation Dataset
  trainset = load_dataset(args.device)
  validset = None
  # print(args.do_val)
  if args.do_val:
    validset = load_dataset(args.device, args.do_val)

  assert(trainset[0]['X_diff'].shape[-1] == validset[0]['X_diff'].shape[-1])
  input_channels = trainset[0]['X_diff'].shape[-1]

  diffuseNet, specularNet, Diff, Spec, Final = None, None, None, None, None

  if args.mode == 'dpcn':
    pass
    diffuseNet, specularNet, Diff, Spec, Final = train_dpcn(trainset, validset, args.device, args.eps, args.num_layers, args.kernel_size, input_channels, args.hidden_channels, args.epochs, args.lr, args.loss, args.do_early_stopping)

  elif args.mode == 'kpcn':
    pass
    diffuseNet, specularNet, Diff, Spec, Final = train_kpcn(trainset, validset, args.device, args.eps, args.num_layers, args.kernel_size, input_channels, args.hidden_channels, args.epochs, args.lr, args.loss, args.do_early_stopping)

  elif args.mode == 'feat_dpcn':
    pass
    diffuseNet, specularNet, Diff, Spec, Final = train_feat_dpcn(trainset, validset, args.device, args.eps, args.num_layers, args.kernel_size, input_channels, args.hidden_channels, args.epochs, args.lr, args.los, args.do_early_stoppings)

  elif args.mode == 'feat_kpcn':
    pass
    diffuseNet, specularNet, Diff, Spec, Final = train_feat_kpcn(trainset, validset, args.device, args.eps, args.num_layers, args.kernel_size, input_channels, args.hidden_channels, args.epochs, args.lr, args.loss, args.do_early_stopping)

  else:
    assert(False)

  # plot_training(Diff, Spec, args.mode)
  


if __name__ == '__main__':
  main()