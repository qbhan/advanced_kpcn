import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from model import *

L = 9 # number of convolutional layers
n_kernels = 100 # number of kernels in each layer
kernel_size = 5 # size of kernel (square)

# input_channels = dataset[0]['X_diff'].shape[-1]
hidden_channels = 100

permutation = [0, 3, 1, 2]
eps = 0.00316
def train(dataset, input_channels, device, mode='KPCN', epochs=20, learning_rate=1e-4, show_images=False):
#   dataset = KPCNDataset()
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                          shuffle=True, num_workers=4)

  # instantiate networks
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

        if False:#i_batch % 500:
          print("Sample, denoised, gt")
          sz = 3
          orig = crop_like(sample_batched['finalInput'].permute(permutation), outputFinal)
          orig = orig.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
          show_data(orig, figsize=(sz,sz), normalize=True)
          img = outputFinal.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
          show_data(img, figsize=(sz,sz), normalize=True)
          gt = crop_like(sample_batched['finalGt'].permute(permutation), outputFinal)
          gt = gt.cpu().permute([0, 2, 3, 1]).numpy()[0,:]
          show_data(gt, figsize=(sz,sz), normalize=True)

        Y_final = sample_batched['finalGt'].permute(permutation).to(device)

        Y_final = crop_like(Y_final, outputFinal)

        lossFinal = criterion(outputFinal, Y_final)

        accuLossFinal += lossFinal.item()

      accuLossDiff += lossDiff.item()
      accuLossSpec += lossSpec.item()

    print("Epoch {}".format(epoch + 1))
    print("LossDiff: {}".format(accuLossDiff))
    print("LossSpec: {}".format(accuLossSpec))
    print("LossFinal: {}".format(accuLossFinal))

    lDiff.append(accuLossDiff)
    lSpec.append(accuLossSpec)
    lFinal.append(accuLossFinal)
    
    accuLossDiff = 0
    accuLossSpec = 0
    accuLossFinal = 0

  print('Finished training in mode', mode)
  print('Took', time.time() - start, 'seconds.')
  
  return diffuseNet, specularNet, lDiff, lSpec, lFinal