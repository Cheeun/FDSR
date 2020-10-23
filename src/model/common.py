import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time
total_FLOPs = 0
i=0 
def prune(a, index):
    # a (1,4,2,2)
    # b (4)
    return a[:,np.nonzero(index).squeeze(1),:,:]

def prune_n(a, len):
    return a[:,:len, :,:]

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class act_PR(nn.Module):
    def __init__(self, affine=True):
        super(act_PR, self).__init__()
        self.prelu = nn.PReLU(num_parameters=1)
        self.relu = nn.ReLU(inplace=False)
        # print("PR called")
    def forward(self, x):
        out = (self.relu(x)+self.prelu(x))/2
        return out
class act_PT(nn.Module):
    def __init__(self, affine=True):
        super(act_PT, self).__init__()
        self.prelu = nn.PReLU(num_parameters=1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        out = (self.prelu(x)+self.tanh(x))/2
        return out

class act_RT(nn.Module):
    def __init__(self, affine=True):
        super(act_RT, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()
    def forward(self, x):
        out = (self.relu(x)+self.tanh(x))/2
        return out
class act_PRT(nn.Module):
    def __init__(self, affine=True):
        super(act_PRT, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.prelu = nn.PReLU(num_parameters=1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        out = (self.relu(x)+self.prelu(x)+self.tanh(x))/3
        return out
        
class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()
    def forward(self,x):
        return torch.tensor([])
        
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x

class Activation(nn.Module): 
    def __init__(self, act_index, act_type):
        super(Activation, self).__init__()
        self.act = nn.Tanh() # default : erase
        if act_type == 'p':
            self.act = nn.PReLU(num_parameters=1)
        elif act_type == 'r':
            self.act = nn.ReLU(inplace=False)
        elif act_type == 't':
            self.act = nn.Tanh()
        elif act_type == 'pr':
            self.act == act_PR(affine=True)
        elif act_type == 'pt':
            self.act == act_PT()
        elif act_type == 'rt':
            self.act == act_RT()
        elif act_type == 'prt':
            self.act == act_PRT()
        elif act_type == 'nop':
            self.act == Identity()
            
        self.index= act_index

    def forward(self, x):
        Act_input = []
        if len(self.index)!=0:
            for idx in range(len(self.index)):
                Act_index = int((self.index[idx]))
                Act_input_data = x[:,Act_index,:,:]
                Act_input.append(Act_input_data)

            Act_input = torch.stack(Act_input,1) 
            x = self.act(Act_input)
        else:
            x= torch.zeros(x.size(0), 0, x.size(2),x.size(3))

        return x

def reshape(a,b):
    return a[np.nonzero(b)].squeeze()


class ActivatedOp(nn.Module): 
    def __init__(self, skip_only, conv_only,skip_conv, op2_gate, input_gate, op_type, scale):
        super(ActivatedOp, self).__init__()
        self.op_type = op_type
        self.res_scale=scale
        self.op2_gate=op2_gate
        self.input_gate=input_gate
        if op_type == 'skip_only':
            self.gate = skip_only        
        elif op_type == 'conv_tot':
            self.gate = conv_only+skip_conv
            self.skip_gate = skip_conv
            if int(sum(self.gate))==0:
                self.conv3= Identity()
            else:
                self.conv3= nn.Conv2d(int(sum(self.gate)), int(sum(self.gate)), kernel_size=3, padding=1, bias=False)
            

    def forward(self, op2, x):
        op2 = prune(op2, self.gate)
        x= prune(x, self.gate)
        if self.op_type == 'skip_only':
            return op2+x
        
        elif self.op_type == 'conv_tot':
            op3 = self.conv3(op2) # 54
            op3 = op3.mul(self.res_scale)
            x = (x.clone().permute(0,2,3,1)*reshape(self.skip_gate,self.gate)).permute(0,3,1,2) # 54 (10 activated)
            op3 += x
             
            return op3
        


    


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,idx, input,op1,op2,P,R,T,PR,PT,RT,PRT, skip_gate_bef, resconv3_gate_bef,skip_gate,resconv3_gate, resconv2_gate,bias=True, bn=False, res_scale=1):
        super(ResBlock, self).__init__()
        self.idx = idx
        self.input_gate = input
        self.op1 = op1
        self.resconv2_gate =resconv2_gate
        self.resconv3_gate = resconv3_gate
        self.op2 = op2
        self.skip_gate_bef =skip_gate_bef
        self.resconv3_gate_bef = resconv3_gate_bef
        self.skip_gate = skip_gate
        self.ch_length = len(P)+len(R)+len(T)+len(PR)+len(PT)+len(RT)+len(PRT)

        self.n_feats_in =int(sum(resconv3_gate_bef + skip_gate_bef - resconv3_gate_bef*skip_gate_bef.detach()))
        self.conv2 = conv(self.n_feats_in, self.ch_length, kernel_size=3, bias=False)

        self.res_scale=res_scale
        self.actP   = Activation(act_index=P,   act_type='p') 
        self.actR   = Activation(act_index=R,   act_type='r')
        self.actT   = Activation(act_index=T,   act_type='t')
        self.actPR  = Activation(act_index=PR,  act_type='pr')
        self.actPT  = Activation(act_index=PT,  act_type='pt')
        self.actRT  = Activation(act_index=RT,  act_type='rt')
        self.actPRT = Activation(act_index=PRT, act_type='prt')



        self.n_feats_out = int(sum(resconv3_gate + skip_gate - resconv3_gate*skip_gate.detach()))

        self.skip_only_gate =  skip_gate-skip_gate*resconv3_gate.detach()
        self.conv_only_gate =  resconv3_gate-skip_gate*resconv3_gate.detach()
        self.skip_conv_gate =  skip_gate*resconv3_gate.detach()

        
        self.skip_only = ActivatedOp(skip_only=self.skip_only_gate,conv_only=self.conv_only_gate,skip_conv=self.skip_conv_gate, op2_gate=op2, input_gate=input,op_type='skip_only',scale=res_scale)
        self.conv_tot  = ActivatedOp(skip_only=self.skip_only_gate,conv_only=self.conv_only_gate,skip_conv=self.skip_conv_gate, op2_gate=op2, input_gate=input,op_type='conv_tot',scale=res_scale)
        
    def forward(self, x):
        x= x.cuda()

        if self.idx == 0:
            x_conv2 = prune(x, self.skip_gate) 
        else:
            x_conv2 = prune_n(x, int(self.skip_gate.sum())) 

        op2 = self.conv2(x_conv2) 
        op2 = torch.cat([self.actP(op2).cuda(), self.actR(op2).cuda(), self.actT(op2).cuda(), self.actPR(op2).cuda(), self.actPT(op2).cuda(), self.actRT(op2).cuda(), self.actPRT(op2).cuda(),prune(x,1-self.resconv2_gate)], dim=1)


        op3_gate = self.skip_only_gate + self.conv_only_gate+self.skip_conv_gate 

        op3 = torch.cat([self.skip_only(op2,x).cuda(), self.conv_tot(op2,x).cuda(),prune(x,1-op3_gate)],dim=1)
        return op3


def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)
    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


class PixelShuffle(nn.Module):
   def __init__(self, scale_factor):
       super(PixelShuffle, self).__init__()
       self.scale_factor = scale_factor

   def forward(self, x):
       return pixel_shuffle(x, self.scale_factor)
   def extra_repr(self):
       return 'scale_factor={}'.format(self.scale_factor)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        self.mode = "pixelshuffle"
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias)) # added this again to fix error for x4
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


