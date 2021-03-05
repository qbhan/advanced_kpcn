import torch
from tensorboardX import SummaryWriter

import argparse

from utils import *
from model import *
from visualize import *
from generate_patch import crop
from preprocess import preprocess_input

# L = 9 # number of convolutional layers
# n_kernels = 100 # number of kernels in each layer
# kernel_size = 5 # size of kernel (square)

# # input_channels = dataset[0]['X_diff'].shape[-1]
# hidden_channels = 100

permutation = [0, 3, 1, 2]
eps = 0.00316

parser = argparse.ArgumentParser(description='Test the model')


parser.add_argument('--device', default='cuda:0')
parser.add_argument('--mode', default='kpcn')
parser.add_argument('--num_layers', default=9, type=int)
parser.add_argument('--input_channels', default=28, type=int)
parser.add_argument('--hidden_channels', default=100, type=int)
parser.add_argument('--kernel_size', default=5, type=int)

parser.add_argument('--diffuse_model')
parser.add_argument('--specular_model')

parser.add_argument('--test_noisy', default='sample_data/eval1.exr')
parser.add_argument('--test_clean', default='sample_data/evalref1.exr')


def unsqueeze_all(d):
  for k, v in d.items():
    d[k] = torch.unsqueeze(v, dim=0)
  return d


def denoise(diffuseNet, specularNet, data, device, debug=False):
  with torch.no_grad():
    out_channels = diffuseNet[len(diffuseNet)-1].out_channels
    mode = 'dpcn' if out_channels == 3 else 'kpcn'
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

    if 'kpcn' in mode:
      X_input = crop_like(X_diff, outputDiff)
      outputDiff = apply_kernel(outputDiff, X_input, device)

    Y_diff = crop_like(Y_diff, outputDiff)

    lossDiff = criterion(outputDiff, Y_diff).item()

    # get the inputs
    X_spec = data['X_spec'].permute(permutation).to(device)
    Y_spec = data['Reference'][:,:,:,3:6].permute(permutation).to(device)

    # forward + backward + optimize
    outputSpec = specularNet(X_spec)

    if 'kpcn' in mode:
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


def test_model(diffuseNet, specularNet, device, test_noisy, test_clean):
  pass
  diffuseNet.to(device)
  specularNet.to(device)
  eval_data = preprocess_input(test_noisy, test_clean)
  eval_data = crop(eval_data, (1280//2, 720//2), 300)
  denoise(diffuseNet, specularNet, eval_data, device)


def main():
  args = parser.parse_args()
  print(args)

  if 'feat' in args.mode:
    diffuseNet = make_feat_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode)
    specularNet = make_feat_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode)
  else:
    diffuseNet = make_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode)
    specularNet = make_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode)

  test_model(diffuseNet, specularNet, args.device, args.test_noisy, args.test_clean)


if __name__ == '__main__':
  main()