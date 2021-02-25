import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class KPCNDataset(torch.utils.data.Dataset):
  def __init__(self, cropped):
    self.samples = cropped

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]