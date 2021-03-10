import torch
import sys
import gc
import matplotlib.pyplot as plt
import numpy as np


def to_torch_tensors(data):
  if isinstance(data, dict):
    for k, v in data.items():
      if not isinstance(v, torch.Tensor):
        data[k] = torch.from_numpy(v)
  elif isinstance(data, list):
    for i, v in enumerate(data):
      if not isinstance(v, torch.Tensor):
        data[i] = to_torch_tensors(v)
    
  return data


def send_to_device(data, device):
  if isinstance(data, dict):
    for k, v in data.items():
      if isinstance(v, torch.Tensor):
        data[k] = v.to(device)
  elif isinstance(data, list):
    for i, v in enumerate(data):
      if isinstance(v, torch.Tensor):
        data[i] = v.to(device)
    
  return data


def getsize(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object reffered to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz

def plot_training(diff, spec, filename):
  pass
  plt.plot(diff, 'r', label="Diffuse")
  plt.plot(spec, 'b', label="Specular")
  plt.title(filename)
  plt.xlim([0, 40])
  plt.xlabel('Loss')
  plt.ylabel('Epoch')
  plt.legend()
  plt.savefig(filename + '.jpg')
  # plt.show()


def ToneMap(c, limit=1.5):
    # c: (W, H, C=3)
    luminance = 0.2126 * c[:,:,0] + 0.7152 * c[:,:,1] + 0.0722 * c[:,:,2]
    col = c.copy()
    col[:,:,0] /=  (1.0 + luminance / limit)
    col[:,:,1] /=  (1.0 + luminance / limit)
    col[:,:,2] /=  (1.0 + luminance / limit)
    return col

def LinearToSrgb(c):
    # c: (W, H, C=3)
    kInvGamma = 1.0 / 2.2
    return np.clip(c ** kInvGamma, 0.0, 1.0)

def ToneMapBatch(c):
    # c: (B, C=3, W, H)
    luminance = 0.2126 * c[:,0,:,:] + 0.7152 * c[:,1,:,:] + 0.0722 * c[:,2,:,:]
    col = c.copy()
    col[:,0,:,:] /= (1 + luminance / 1.5)
    col[:,1,:,:] /= (1 + luminance / 1.5)
    col[:,2,:,:] /= (1 + luminance / 1.5)
    col = np.clip(col, 0, None)
    kInvGamma = 1.0 / 2.2
    return np.clip(col ** kInvGamma, 0.0, 1.0)


def crop_like(src, tgt):
    src_sz = np.array(src.shape)
    tgt_sz = np.array(tgt.shape)

    # Assumes the spatial dimensions are the last two
    # delta = (src_sz[2:4]-tgt_sz[2:4])
    delta = (src_sz[-2:]-tgt_sz[-2:])
    crop = np.maximum(delta // 2, 0)  # no negative crop
    crop2 = delta - crop

    if (crop > 0).any() or (crop2 > 0).any():
        # NOTE: convert to ints to enable static slicing in ONNX conversion
        src_sz = [int(x) for x in src_sz]
        crop = [int(x) for x in crop]
        crop2 = [int(x) for x in crop2]
        return src[..., crop[0]:src_sz[-2]-crop2[0],
                   crop[1]:src_sz[-1]-crop2[1]]
    else:
        return src
