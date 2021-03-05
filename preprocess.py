import argparse
import glob

import numpy as np
import pyexr
import torch

from generate_patch import *
from utils import *

# patch_size = 64 # patches are 64x64
# n_patches = 400
eps = 0.00316

def build_data(img):
  data = img.get()


def preprocess_diffuse(diffuse, albedo):
  return diffuse / (albedo + eps)


def postprocess_diffuse(diffuse, albedo):
  return diffuse * (albedo + eps)


def preprocess_specular(specular):
  assert(np.sum(specular < 0) == 0)
  return np.log(specular + 1)


def postprocess_specular(specular):
  return np.exp(specular) - 1


def preprocess_diff_var(variance, albedo):
  return variance / (albedo + eps)**2


def preprocess_spec_var(variance, specular):
  return variance / (specular+1e-5)**2


def gradients(data):
  h, w, c = data.shape
  dX = data[:, 1:, :] - data[:, :w - 1, :]
  dY = data[1:, :, :] - data[:h - 1, :, :]
  # padding with zeros
  dX = np.concatenate((np.zeros([h,1,c], dtype=np.float32),dX), axis=1)
  dY = np.concatenate((np.zeros([1,w,c], dtype=np.float32),dY), axis=0)
  
  return np.concatenate((dX, dY), axis=2)


def remove_channels(data, channels):
  for c in channels:
    if c in data:
      del data[c]
    else:
      print("Channel {} not found in data!".format(c))

      
# returns network input data from noisy .exr file
def preprocess_input(filename, gt, debug=False):
  
  file = pyexr.open(filename)
  # file = OpenEXR.InputFile(filename)
  data = file.get_all()
  
  if debug:
    for k, v in data.items():
      print(k, v.dtype)

  # just in case
  for k, v in data.items():
    data[k] = np.nan_to_num(v)

    
    
  file_gt = pyexr.open(gt)
  gt_data = file_gt.get_all()
  
  # just in case
  for k, v in gt_data.items():
    gt_data[k] = np.nan_to_num(v)
    
    
  # clip specular data so we don't have negative values in logarithm
  data['specular'] = np.clip(data['specular'], 0, np.max(data['specular']))
  data['specularVariance'] = np.clip(data['specularVariance'], 0, np.max(data['specularVariance']))
  gt_data['specular'] = np.clip(data['specular'], 0, np.max(data['specular']))
  gt_data['specularVariance'] = np.clip(gt_data['specularVariance'], 0, np.max(gt_data['specularVariance']))
    
    
  # save albedo
  data['origAlbedo'] = data['albedo'].copy()
    
  # save reference data (diffuse and specular)
  diff_ref = preprocess_diffuse(gt_data['diffuse'], gt_data['albedo'])
  spec_ref = preprocess_specular(gt_data['specular'])
  diff_sample = preprocess_diffuse(data['diffuse'], data['albedo'])
  
  data['Reference'] = np.concatenate((diff_ref[:,:,:3].copy(), spec_ref[:,:,:3].copy()), axis=2)
  data['Sample'] = np.concatenate((diff_sample, data['specular']), axis=2)
  
  # save final input and reference for error calculation
  # apply albedo and add specular component to get final color
  data['finalGt'] = gt_data['default']#postprocess_diffuse(data['Reference'][:,:,:3], data['albedo']) + data['Reference'][:,:,3:]
  data['finalInput'] = data['default']#postprocess_diffuse(data['diffuse'][:,:,:3], data['albedo']) + data['specular'][:,:,3:]
    
    
    
    
  # preprocess diffuse
  data['diffuse'] = preprocess_diffuse(data['diffuse'], data['albedo'])

  # preprocess diffuse variance
  data['diffuseVariance'] = preprocess_diff_var(data['diffuseVariance'], data['albedo'])

  # preprocess specular
  data['specular'] = preprocess_specular(data['specular'])

  # preprocess specular variance
  data['specularVariance'] = preprocess_spec_var(data['specularVariance'], data['specular'])

  # just in case
  data['depth'] = np.clip(data['depth'], 0, np.max(data['depth']))

  # normalize depth
  max_depth = np.max(data['depth'])
  if (max_depth != 0):
    data['depth'] /= max_depth
    # also have to transform the variance
    data['depthVariance'] /= max_depth * max_depth

  # Calculate gradients of features (not including variances)
  data['gradNormal'] = gradients(data['normal'][:, :, :3].copy())
  data['gradDepth'] = gradients(data['depth'][:, :, :1].copy())
  data['gradAlbedo'] = gradients(data['albedo'][:, :, :3].copy())
  data['gradSpecular'] = gradients(data['specular'][:, :, :3].copy())
  data['gradDiffuse'] = gradients(data['diffuse'][:, :, :3].copy())
  data['gradIrrad'] = gradients(data['default'][:, :, :3].copy())

  # append variances and gradients to data tensors
  data['diffuse'] = np.concatenate((data['diffuse'], data['diffuseVariance'], data['gradDiffuse']), axis=2)
  data['specular'] = np.concatenate((data['specular'], data['specularVariance'], data['gradSpecular']), axis=2)
  data['normal'] = np.concatenate((data['normalVariance'], data['gradNormal']), axis=2)
  data['depth'] = np.concatenate((data['depthVariance'], data['gradDepth']), axis=2)

  if debug:
    for k, v in data.items():
      print(k, v.shape, v.dtype)

  X_diff = np.concatenate((data['diffuse'],
                           data['normal'],
                           data['depth'],
                           data['gradAlbedo']), axis=2)

  X_spec = np.concatenate((data['specular'],
                           data['normal'],
                           data['depth'],
                           data['gradAlbedo']), axis=2)
  
  assert not np.isnan(X_diff).any()
  assert not np.isnan(X_spec).any()

  print("X_diff shape:", X_diff.shape)
  print(X_diff.dtype, X_spec.dtype)
  
  data['X_diff'] = X_diff
  data['X_spec'] = X_spec
  
  remove_channels(data, ('diffuseA', 'specularA', 'normalA', 'albedoA', 'depthA',
                         'visibilityA', 'colorA', 'gradNormal', 'gradDepth', 'gradAlbedo',
                        'gradSpecular', 'gradDiffuse', 'gradIrrad', 'albedo', 'diffuse', 
                         'depth', 'specular', 'diffuseVariance', 'specularVariance',
                        'depthVariance', 'visibilityVariance', 'colorVariance',
                        'normalVariance', 'depth', 'visibility'))
  
  return data


def image_preprocess(patch_size, n_patches, val=False):
  cropped = []
  if not val:
    cropped += get_cropped_patches("sample_data/sample.exr", "sample_data/gt.exr", patch_size, n_patches)
    for f in glob.glob('sample_data/sample*.exr'):

      if f == 'sample_data\sample.exr': continue

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
  else:
    for f in glob.glob('sample_data/evalref*.exr'):
      print(f)
      if f == 'sample_data\evalref1.exr': 
        print('RESERVE FOR TEST')
        continue

      num = f[len('sample_data/evalref'):f.index('.')]
      sample_name = 'sample_data/eval{}.exr'.format(num)
      gt_name = 'sample_data/evalref{}.exr'.format(num)
      print(sample_name, gt_name)
      cropped += get_cropped_patches(sample_name, gt_name, patch_size, n_patches)

    print('Patches cropped : ' + str(len(cropped)))
    print('Saving patches')
    # save the training data
    for i, v in enumerate(cropped):
      torch.save(v, 'val/eval'+str(i+1)+'.pt')
      # print('SAVED data/sample'+str(i+1)+'.pt')

  # Check sizes of data
  for k, v in cropped[0].items():
    print(k, getsize(v))
    
  print(getsize(cropped) / 1024 / 1024, "MiB")

parser = argparse.ArgumentParser(description='Preprocess Scenes')

parser.add_argument('--patch_size', default=64, type=int)
parser.add_argument('--n_patches', default=400, type=int)
parser.add_argument('--val', dest='val', action='store_true')
parser.set_defaults(val=False)

def main():
  pass
  args = parser.parse_args()
  print(args.patch_size, args.n_patches, args.val)
  image_preprocess(args.patch_size, args.n_patches, args.val)


if __name__ == '__main__':
  main()