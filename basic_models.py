import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class featBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(featBasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        # init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out


class featBasicBlockSig(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(featBasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )

        # init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out


class featAttnBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(featAttnBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = featBasicBlock(channel , channel // reduction, 1, 1, 0)
        self.c2 = featBasicBlockSig(channel // reduction, channel , 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2 + x
        

# class BasicBlock(nn.Module):
#     def __init__(self, input_channels, hidden_channels, kernel_size):
#         super(BasicBlock, self).__init__()

#         self.conv = nn.Conv2d(input_channels, hidden_channels, kernel_size)
#         self.act = nn.ReLU()
