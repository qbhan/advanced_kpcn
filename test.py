import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

import argparse
import os
from tqdm import tqdm

from utils import *
from model import *
from visualize import *
from generate_patch import crop
from preprocess import preprocess_input
from dataset import DenoiseDataset

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
parser.add_argument('--input_channels', default=34, type=int)
parser.add_argument('--hidden_channels', default=100, type=int)
parser.add_argument('--kernel_size', default=5, type=int)

parser.add_argument('--diffuse_model')
parser.add_argument('--specular_model')

parser.add_argument('--data_dir')
parser.add_argument('--save_dir')
parser.add_argument('--test_noisy', default='sample_data/eval1.exr')
parser.add_argument('--test_clean', default='sample_data/evalref1.exr')


def unsqueeze_all(d):
  for k, v in d.items():
    d[k] = torch.unsqueeze(v, dim=0)
  return d


def denoise(diffuseNet, specularNet, dataloader, device, mode, save_dir, debug=False):
  with torch.no_grad():
    # out_channels = diffuseNet[len(diffuseNet)-1].out_channels
    # mode = 'dpcn' if out_channels == 3 else 'kpcn'
    # criterion = nn.L1Loss()
    
    # if debug:
    #   print("Out channels:", out_channels)
    #   print("Detected mode", mode)
    # writer = SummaryWriter('runs/test_' + mode + '_2')
    criterion = nn.L1Loss()
    lossDiff, lossSpec, lossFinal = 0,0,0
    # for image_idx, data in enumerate(dataloader):
    image_idx = 0
    # input_image = torch.zeros((3, 1024, 1024))
    input_image = torch.zeros((3, 960, 960)).to(device)
    gt_image = torch.zeros((3, 960, 960)).to(device)
    output_image = torch.zeros((3, 960, 960)).to(device)

    x, y = 0, 0
    for data in tqdm(dataloader, leave=False, ncols=70):
      # print(x, y)
      X_diff = data['kpcn_diffuse_in'].to(device)
      Y_diff = data['target_diffuse'].to(device)

      outputDiff = diffuseNet(X_diff)
      # if mode == 'KPCN':
      if 'kpcn' in mode:
        X_input = crop_like(X_diff, outputDiff)
        outputDiff = apply_kernel(outputDiff, X_input, device)

      Y_diff = crop_like(Y_diff, outputDiff)
      lossDiff += criterion(outputDiff, Y_diff).item()

      X_spec = data['kpcn_specular_in'].to(device)
      Y_spec = data['target_specular'].to(device)
      
      outputSpec = specularNet(X_spec)
      # if mode == 'KPCN':
      if 'kpcn' in mode:
        X_input = crop_like(X_spec, outputSpec)
        outputSpec = apply_kernel(outputSpec, X_input, device)

      Y_spec = crop_like(Y_spec, outputSpec)
      lossSpec += criterion(outputSpec, Y_spec).item()

      # calculate final ground truth error
      albedo = data['kpcn_albedo'].to(device)
      albedo = crop_like(albedo, outputDiff)
      outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

      Y_final = data['target_total'].to(device)
      Y_final = crop_like(Y_final, outputFinal)
      lossFinal += criterion(outputFinal, Y_final).item()


      # visualize
      
      # inputFinal = data['kpcn_diffuse_buffer'] * (albedo + eps) + torch.exp(data['kpcn_specular_buffer']) - 1.0
      inputFinal = data['kpcn_diffuse_buffer'] * (data['kpcn_albedo'] + eps) + torch.exp(data['kpcn_specular_buffer']) - 1.0
      # inputGrid = torchvision.utils.make_grid(inputFinal)
      # writer.add_image('noisy patches e{}'.format(image_idx+1), inputGrid)
      # save_image(inputFinal, save_dir + '/test{}/noisy.png'.format(image_idx+1))
      # print(np.shape(inputFinal))
      # print(np.shape(outputFinal))
      # print(np.shape(Y_final))
      input_image[:, x*64:x*64+64, y*64:y*64+64] = inputFinal[0, :, 32:96, 32:96]
      output_image[:, x*64:x*64+64, y*64:y*64+64] = outputFinal[0, :, 16:80, 16:80]
      gt_image[:, x*64:x*64+64, y*64:y*64+64] = Y_final[0, :, 16:80, 16:80]
      y += 1
      if x < 15 and y>=15:
        x += 1
        y = 0

      if x >= 15:
        if not os.path.exists(save_dir + '/test{}'.format(image_idx)):
          os.makedirs(save_dir + '/test{}'.format(image_idx))
        # inputGrid = torchvision.utils.make_grid(input_image)
        # outputGrid = torchvision.utils.make_grid(output_image)
        # cleanGrid = torchvision.utils.make_grid(gt_image)
        save_image(ToneMapTest(input_image), save_dir + '/test{}/noisy.png'.format(image_idx))
        save_image(ToneMapTest(output_image), save_dir + '/test{}/denoise.png'.format(image_idx))
        save_image(ToneMapTest(gt_image), save_dir + '/test{}/clean.png'.format(image_idx))
        # print('SAVED IMAGES')
        x, y = 0, 0
        image_idx += 1
      # outputGrid = torchvision.utils.make_grid(outputFinal)
      # writer.add_image('denoised patches e{}'.format(image_idx+1), outputGrid)
      # save_image(outputFinal, save_dir + '/test{}/denoise.png'.format(image_idx+1))

      # cleanGrid = torchvision.utils.make_grid(Y_final)
      # writer.add_image('clean patches e{}'.format(image_idx+1), cleanGrid)
      # save_image(Y_final, save_dir + '/test{}/clean.png'.format(image_idx+1))


  return lossDiff/len(dataloader), lossSpec/len(dataloader), lossFinal/len(dataloader)

def test_model(diffuseNet, specularNet, device, data_dir, mode, args):
  pass
  diffuseNet.to(device)
  specularNet.to(device)
  # test_input_dir = os.path.join(test_dir, 'input')
  # test_gt_dir = os.path.join(test_dir, 'gt')
  # for subdir, dirs, files in os.walk(test_gt_dir):
  #   for file in files:
  #     test_clean = np.load(os.path.join(subdir, file))
  #     test_noisy = None
  #     for _, _, noisy_files in os.walk(test_input_dir):
  #       for noisy_file in noisy_files:
  #         if file.split('.')[0] in noisy_file and '8' in noisy_file:
  #           test_noisy = np.load(os.path.join(subdir, noisy_file))
  #     assert(test_noisy)
  #     eval_data = (test_clean, test_noisy)
  #     # eval_data = preprocess_input(test_noisy, test_clean)
  #     # eval_data = crop(eval_data, (1280//2, 720//2), 300)
  #     denoise(diffuseNet, specularNet, eval_data, device)
  dataset = DenoiseDataset(data_dir, 8, 'kpcn', 'test', 1, 'recon',
        use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, pnet_out_size=3)
  dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=False
    )
  denoise(diffuseNet, specularNet, dataloader, device, mode, args.save_dir)


def main():
  args = parser.parse_args()
  print(args)

  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

  if 'simple_feat' in args.mode:
    diffuseNet = make_simple_feat_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode)
    specularNet = make_simple_feat_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode)
  else:
    diffuseNet = make_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode)
    specularNet = make_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode)

  diffuseNet.load_state_dict(torch.load(args.diffuse_model))
  specularNet.load_state_dict(torch.load(args.specular_model))
  test_model(diffuseNet, specularNet, args.device, args.data_dir, args.mode, args)



if __name__ == '__main__':
  main()