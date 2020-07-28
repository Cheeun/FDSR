import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

total_FLOPs = 0
i=0 
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def flops_counter(conv_module, in_ch, out_ch, output):
    kernel_flops = 2*(conv_module.kernel_size)*(conv_module.kernel_size)*in_ch*out_ch
    active_element = np.prod(np.array(output.shape))/out_ch
    
    if conv_module.bias is not None:
        bias_flops = out_ch * active_element
    else:
        bias_flops =0
    
    flops = kernel_flops * active_element + bias_flops
    return flops / output.shape[0]

def conv_flops_counter(conv_module, output, in_channels, out_channels):
    # Can have multiple inputs, getting the first one
    batch_size = output.shape[0]
    output_height, output_width = output.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = in_channels
    out_channels = out_channels
    
    # We count multiply-add as 2 flops
    conv_per_position_flops = 2 * kernel_height * kernel_width * in_channels * out_channels
    active_elements_count = batch_size * output_height * output_width

    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0

    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count
    
    overall_flops = overall_conv_flops + bias_flops

    return overall_flops/batch_size

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

class Activation(nn.Module): 
    def __init__(self, act_index, act_type):
        super(Activation, self).__init__()
        # p r t pr pt rt prt zero
        # self.act = None
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
        # print(self.act)
        self.index= act_index

    def forward(self, x):
        Act_input = []
        # print(self.index)
        # if self.index.dim()!=0:
        if len(self.index)!=0:
            for idx in range(len(self.index)):
                # print(self.index[idx])
                Act_index = int((self.index[idx]))
                Act_input_data = x[:,Act_index,:,:]
                Act_input.append(Act_input_data)

            Act_input = torch.stack(Act_input,1) 
            # print(Act_input.shape)
            # Act_input =Act_input.transpose(0,1)
            x = self.act(Act_input)
        else:
            # x=self.act(x)
            x= torch.zeros(x.size(0), 0, x.size(2),x.size(3)).cuda()
            # print(x)
            # print(x.size())         
        return x

class Truncate(nn.Module): 
    # returns x that corresponds to activated operation gate's indices 
    def __init__(self, index):
        super(Truncate, self).__init__()
        self.index= index
    def forward(self, x):
        truncated_x = []
        if len(self.index)!=0:
            for idx in range(len(self.index)):
                conv_index = int((self.index[idx]))
                truncated_x.append(x[:,conv_index,:,:])
            truncated_x = torch.stack(truncated_x) 
            truncated_x =truncated_x.transpose(0,1)
            return truncated_x
        else:
            return x

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, P,R,T,PR,PT,RT,PRT, op3_gate, skip_gate, bias=True, bn=False, res_scale=1):
        # conv 1x1
        # norm
        # act_ops
        super(ResBlock, self).__init__()
        self.conv = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size=3, bias=False)

        self.res_scale=res_scale
        self.actP = Activation(act_index=P, act_type='p') 
        self.actR = Activation(act_index=R, act_type='r')
        self.actT = Activation(act_index=T, act_type='t')
        self.actPR = Activation(act_index=PR, act_type='pr')
        self.actPT = Activation(act_index=PT, act_type='pt')
        self.actRT = Activation(act_index=RT, act_type='rt')
        self.actPRT = Activation(act_index=PRT, act_type='prt')
        self.truncate_op3 = Truncate(op3_gate)
        self.skip_gate = skip_gate
        self.skip_length = int(skip_gate.sum())
        # self.truncate_skip = Truncate(skip_gate)
        
        self.ch_length = len(P)+len(R)+len(T)+len(PR)+len(PT)+len(RT)+len(PRT)
        # self.ch_length = len(P)+len(R)+len(T)+len(PR)+len(PT)+len(RT)+len(PRT)+len(op3_gate)
        self.conv3 = nn.Conv2d(self.ch_length, 64, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x=x.cuda()
        op1 = self.conv(x)
        op2 = self.conv2(op1)
        # print("P",self.actP(op2).shape)
        # print("r",self.actR(op2).shape)
        # print("t",self.actT(op2).shape)

        op2 = torch.cat([self.actP(op2), self.actR(op2), self.actT(op2), self.actPR(op2), self.actPT(op2), 
                        self.actRT(op2), self.actPRT(op2)], dim=1)
        # op2 = torch.cat([self.actP(op2), self.actR(op2), self.actT(op2), self.actPR(op2), self.actPT(op2), 
        #                 self.actRT(op2), self.actPRT(op2),self.truncate_op3(op2)], dim=1)
        # this was done in ver 1. > revised to fit 2 integrated
        # TODO
        # out_gate_conv = act_gates + op3_gate
        
        op3= self.conv3(op2)
        op3= op3.mul(self.res_scale).cuda()
        
        op3 += torch.mul(x,self.skip_gate)
        return op3
        
        
        # conv3_input = conv3_gate의 index에 해당하는 op1 가져오기
        # op2 = torch.cat([op2, conv3_input], dim=1)
        # op3 = conv3(op2)
        # op3= op3.mul(self.res_scale).cuda()
        # op3 += x*skip_gate
        # return op3
        """
        padded_res= torch.zeros(x.size())
        padded_res[:,:res.size(1),:,:]=res
        self.conv3= default_conv(x.size(1),x.size(1),kernel_size=3)
        padded_res = self.conv3(padded_res)
        """
        # res =self.prune(res)
        # padded_res= torch.zeros(x.size())
        # padded_res[:,:res.size(1),:,:]=res
        
        # res2 = self.conv3(res)
        # res = res2.mul(self.res_scale).cuda()
        
        # res += x.cuda()
        # res += self.prune(x)
        # res += x
        # return res


# class ResBlock(nn.Module):
#     def __init__(
#         self, conv, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

#         super(ResBlock, self).__init__()
#         m = []
#         for i in range(2):
#             m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
#             if bn:
#                 m.append(nn.BatchNorm2d(n_feats))
#             if i == 0:
#                 m.append(act)

#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale

#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res += x

#         return res
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
                # m.append(conv(n_feats, n_feats, 64, bias)) # fixed at 7/10
                m.append(conv(n_feats, 4 * n_feats, 3, bias)) # added this again to fix error for x4
                # m.append(conv(n_feats, 4 * n_feats, 3, bias)) # updated : now in edsr tail
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

