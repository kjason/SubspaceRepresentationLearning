"""
Created on Mon Aug 14 2023

@author: Kuan-Lin Chen

Modified from https://github.com/kjason/DnnNormTimeFreq4DoA/tree/main/SpeechEnhancement

https://arxiv.org/abs/2408.16605
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ComplexMat2RealImagMat, RealImagMat2ComplexMat, RealImagMat2GramComplexMat, RealVec2HermitianMat, RealVec2HermitianToeplitzMat

class BasicBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, stride=1, bias=False, bn=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)
        if bn is True:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(mid_planes)
        if stride != 1 or in_planes != out_planes:
            self.projection = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x):
        y = self.bn1(x) if hasattr(self,'bn1') else x
        y = F.relu(y)
        shortcut = self.projection(y) if hasattr(self, 'projection') else x
        y = self.conv1(y)
        y = self.bn2(y) if hasattr(self,'bn2') else y
        v = F.relu(y)
        out = self.conv2(v) + shortcut if hasattr(self,'conv2') else 0
        return out,v

# see Fig. 2 in the paper
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_out_channels, num_mid_channels, M_sensor_ULA, bias, bn, out_type):
        super(ResNet, self).__init__()
        assert(len(num_blocks)==len(num_out_channels)), "size does not match between num_blocks and num_out_channels"
        assert(len(num_blocks)==len(num_mid_channels)), "size does not match between num_blocks and num_mid_channels"
        self.bias = bias
        self.bn = bn
        self.in_planes = num_out_channels[0]
        self.num_blocks = num_blocks
        self.M_sensor_ULA = M_sensor_ULA
        self.out_type = out_type
        self.expansion = nn.Conv2d(2, num_out_channels[0], kernel_size=3, stride=1, padding=1, bias=bias)
        self.stage = nn.ModuleList()
        self.stage.append(self._creat_block_seq(block, num_mid_channels[0], num_out_channels[0], num_blocks[0], stride=1))
        for j in range(1,len(num_blocks)):
            self.stage.append(self._creat_block_seq(block, num_mid_channels[j], num_out_channels[j], num_blocks[j], stride=2))
        if bn is True:
            self.final_bn = nn.BatchNorm2d(num_out_channels[-1])
        if out_type == 'direct':
            self.linear = nn.Linear(num_out_channels[-1],M_sensor_ULA - 1)
        elif out_type == 'branch':
            self.linear = nn.ModuleList()
            for j in range(1,M_sensor_ULA):
                self.linear.append(nn.Linear(num_out_channels[-1],j))
        elif out_type == 'hermitian':
            self.linear = nn.Linear(num_out_channels[-1],M_sensor_ULA**2)
        elif out_type == 'toep':
            self.linear = nn.Linear(num_out_channels[-1],2*M_sensor_ULA-1)
        elif out_type == 'gram':
            self.linear = nn.Linear(num_out_channels[-1],2*M_sensor_ULA**2)
        else:
            self.linear = nn.Linear(num_out_channels[-1],2*M_sensor_ULA**2)

    def _creat_block_seq(self, block, mid_planes, out_planes, num_blocks, stride):
        stride_seq = [stride] + [1]*(num_blocks-1)
        block_seq = nn.ModuleList()
        for stride in stride_seq:
            block_seq.append(block(self.in_planes, mid_planes, out_planes, stride, self.bias, self.bn))
            self.in_planes = out_planes
        return block_seq

    def forward(self, x):
        x = ComplexMat2RealImagMat(x)
        out = self.expansion(x)
        for j in range(len(self.num_blocks)):
            for i in range(self.num_blocks[j]):
                out,_ = self.stage[j][i](out)
        out = self.final_bn(out) if hasattr(self,'final_bn') else out
        out = F.relu(out)
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        if self.out_type == 'direct':
            out = self.linear(out)
            return out
        elif self.out_type == 'branch':
            return [self.linear[j](out) for j in range(self.M_sensor_ULA-1)]
        elif self.out_type == 'hermitian':
            out = self.linear(out)
            return RealVec2HermitianMat(out)
        elif self.out_type == 'toep':
            out = self.linear(out)
            return RealVec2HermitianToeplitzMat(out)
        elif self.out_type == 'gram':
            out = self.linear(out)
            return RealImagMat2GramComplexMat(torch.reshape(out,(-1,2,self.M_sensor_ULA,self.M_sensor_ULA)))
        else:
            out = self.linear(out)
            return RealImagMat2ComplexMat(torch.reshape(out,(-1,2,self.M_sensor_ULA,self.M_sensor_ULA)))

# models

def N4_M7_ResNet_20(): return ResNet(BasicBlock,[3,3,3],[16,32,64],[16,32,64],7,True,False,out_type='gram')
def N5_M10_ResNet_20(): return ResNet(BasicBlock,[3,3,3],[16,32,64],[16,32,64],10,True,False,out_type='gram')
def N6_M14_ResNet_20(): return ResNet(BasicBlock,[3,3,3],[16,32,64],[16,32,64],14,True,False,out_type='gram')

def N4_M7_toep_ResNet_20(): return ResNet(BasicBlock,[3,3,3],[16,32,64],[16,32,64],7,True,False,out_type='toep')
def N5_M10_toep_ResNet_20(): return ResNet(BasicBlock,[3,3,3],[16,32,64],[16,32,64],10,True,False,out_type='toep')
def N6_M14_toep_ResNet_20(): return ResNet(BasicBlock,[3,3,3],[16,32,64],[16,32,64],14,True,False,out_type='toep')

def N4_M7_WRN_16_8(): return ResNet(BasicBlock,[2,2,2],[128,256,512],[128,256,512],7,True,False,out_type='gram')
def N5_M10_WRN_16_8(): return ResNet(BasicBlock,[2,2,2],[128,256,512],[128,256,512],10,True,False,out_type='gram')
def N6_M14_WRN_16_8(): return ResNet(BasicBlock,[2,2,2],[128,256,512],[128,256,512],14,True,False,out_type='gram')

def N4_M7_toep_WRN_16_8(): return ResNet(BasicBlock,[2,2,2],[128,256,512],[128,256,512],7,True,False,out_type='toep')
def N5_M10_toep_WRN_16_8(): return ResNet(BasicBlock,[2,2,2],[128,256,512],[128,256,512],10,True,False,out_type='toep')
def N6_M14_toep_WRN_16_8(): return ResNet(BasicBlock,[2,2,2],[128,256,512],[128,256,512],14,True,False,out_type='toep')

def N5_M10_WRN_40_4(): return ResNet(BasicBlock,[6,6,6],[64,128,256],[64,128,256],10,True,False,out_type='gram')
def N5_M10_WRN_28_10(): return ResNet(BasicBlock,[4,4,4],[160,320,640],[160,320,640],10,True,False,out_type='gram')

def N5_M10_hermitian_WRN_16_8(): return ResNet(BasicBlock,[2,2,2],[128,256,512],[128,256,512],10,True,False,out_type='hermitian')

def Direct_N5_M10_ResNet_20(): return ResNet(BasicBlock,[3,3,3],[16,32,64],[16,32,64],10,True,False,out_type='direct')
def Direct_N5_M10_WRN_16_8(): return ResNet(BasicBlock,[2,2,2],[128,256,512],[128,256,512],10,True,False,out_type='direct')

def Branch_N5_M10_ResNet_20(): return ResNet(BasicBlock,[3,3,3],[16,32,64],[16,32,64],10,True,False,out_type='branch')
def Branch_N5_M10_WRN_16_8(): return ResNet(BasicBlock,[2,2,2],[128,256,512],[128,256,512],10,True,False,out_type='branch')

model_dict = {
    'N4_M7_ResNet_20': N4_M7_ResNet_20,
    'N5_M10_ResNet_20': N5_M10_ResNet_20,
    'N6_M14_ResNet_20': N6_M14_ResNet_20,
    'N4_M7_toep_ResNet_20': N4_M7_toep_ResNet_20,
    'N5_M10_toep_ResNet_20': N5_M10_toep_ResNet_20,
    'N6_M14_toep_ResNet_20': N6_M14_toep_ResNet_20,
    'N4_M7_WRN_16_8': N4_M7_WRN_16_8,
    'N5_M10_WRN_16_8': N5_M10_WRN_16_8,
    'N6_M14_WRN_16_8': N6_M14_WRN_16_8,
    'N4_M7_toep_WRN_16_8': N4_M7_toep_WRN_16_8,
    'N5_M10_toep_WRN_16_8': N5_M10_toep_WRN_16_8,
    'N6_M14_toep_WRN_16_8': N6_M14_toep_WRN_16_8,
    'N5_M10_WRN_40_4': N5_M10_WRN_40_4,
    'N5_M10_WRN_28_10': N5_M10_WRN_28_10,
    'N5_M10_hermitian_WRN_16_8' : N5_M10_hermitian_WRN_16_8,
    'Direct_N5_M10_ResNet_20': Direct_N5_M10_ResNet_20,
    'Direct_N5_M10_WRN_16_8': Direct_N5_M10_WRN_16_8,
    'Branch_N5_M10_ResNet_20': Branch_N5_M10_ResNet_20,
    'Branch_N5_M10_WRN_16_8': Branch_N5_M10_WRN_16_8
    }