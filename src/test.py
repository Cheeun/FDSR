
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import numpy as np
from PIL import Image 

import time
import argparse
import math
import matplotlib.pyplot as plt
import glob

parser = argparse.ArgumentParser(description='FDSR')
parser.add_argument('--scale', type=int, default='4',help='super resolution scale')
# parser.add_argument('--pre_train', type=str, default='searched_small_edsr_x4/model_best.pt', help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16, help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
# parser.add_argument('--import_dir', type=str, default='searched_small_edsr_x4', help='file dir to import from')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
args = parser.parse_args()

args.pre_train = 'network_architectures/fdsr_small_x'+str(args.scale)+'/model_best.pt'
args.import_dir = 'network_architectures/fdsr_small_x'+str(args.scale)
##################### Exported Architecture ##########################
args.op1 = torch.load(args.import_dir+'/NoAct1.pt') 
args.op2 = torch.load(args.import_dir+'/NoAct2.pt')
args.op_last = torch.load(args.import_dir+'/NoAct3.pt')
args.skip = torch.load(args.import_dir+'/skip.pt')
args.skip_num=0
for i in range(args.n_resblocks): 
    args.skip_num+=args.skip[i].sum()
args.P  = np.load(args.import_dir+'/p.npy' ,allow_pickle=True)
args.R  = np.load(args.import_dir+'/r.npy' ,allow_pickle=True)
args.T  = np.load(args.import_dir+'/t.npy' ,allow_pickle=True)
args.PR = np.load(args.import_dir+'/pr.npy',allow_pickle=True)
args.PT = np.load(args.import_dir+'/pt.npy',allow_pickle=True)
args.RT = np.load(args.import_dir+'/rt.npy',allow_pickle=True)
args.PRT= np.load(args.import_dir+'/prt.npy', allow_pickle=True)
args.op3= np.load(args.import_dir+'/conv3.npy',allow_pickle=True)

###################FDSR Architecture #################################
total_FLOPs = 0
i=0 
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
            x = self.act(Act_input)
        else:
            x= torch.zeros(x.size(0), 0, x.size(2),x.size(3)).cuda()
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

        op2 = torch.cat([self.actP(op2), self.actR(op2), self.actT(op2), self.actPR(op2), self.actPT(op2), 
                        self.actRT(op2), self.actPRT(op2)], dim=1)

        
        op3= self.conv3(op2)
        op3= op3.mul(self.res_scale).cuda()
        
        op3 += torch.mul(x,self.skip_gate)
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




class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)
        
        p=args.P
        r=args.R
        t=args.T
        pr=args.PR
        pt=args.PT
        rt=args.RT
        prt=args.PRT
        op3_gate = args.op3
        skip_gate = args.skip

        self.head = conv(args.n_colors, n_feats, kernel_size)
        self.op1 = args.op1
        self.op1_gate = len(args.op1)
        
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, p[i],r[i], t[i],pr[i],pt[i],rt[i],prt[i],op3_gate[i],skip_gate[i], res_scale=args.res_scale
            ) for i in range(n_resblocks)
        ]
        
        self.body = nn.Sequential(*m_body)
        self.body2 = conv(len(args.op2), n_feats, kernel_size)

        self.op2 = args.op2
        
        self.op_last = args.op_last
        self.tail1 = conv(len(args.op_last), args.n_colors*(4**int(math.log(scale, 2))), kernel_size)# only covers scale 2, 4
        tail2 = [nn.PixelShuffle(2) for i in range(int(math.log(scale, 2)))] # only covers scale 2, 4
        self.tail2 = nn.Sequential(*tail2)



    def forward(self, x):
        x = x.cuda()
        x = self.sub_mean(x)

        x1 = self.head(x)
        res = self.body(x1)
        res_=torch.zeros(res.size(0),len(self.op2),res.size(2),res.size(3))

        for i in range(len(self.op2)):
            res_[:,i,:,:]=res[:,self.op2[i],:,:]
        res_=res_.cuda()
        res2 = self.body2(res_)

        res2 += x1

        res2_=torch.zeros(res2.size(0),len(self.op_last),res2.size(2),res2.size(3))
        for i in range(len(self.op_last)):
            res2_[:,i,:,:]=res2[:,self.op_last[i],:,:] 
        res2_=res2_.cuda()
        x = self.tail1(res2_)
        x=self.tail2(x)
        x=self.add_mean(x)
    
        return x


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail2') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail2') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))




    

    

def main():
    ######### Load Model #######
    device=torch.device("cuda")
    model = EDSR(args)
    kwargs = {}
    print('Scale'+str(args.scale))
    print('Load the model from {}'.format(args.pre_train))
    load_from = torch.load(args.pre_train, **kwargs)
    if load_from:
        model.load_state_dict(load_from, strict=False)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    ###### Calculate Parameters #####
    n_params = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        n_params += nn
    print('Parameters: {:.1f}K \n'.format(n_params/(10**3)))
    idx_scale =0

    lr_list =[]
    for filename in glob.glob('images/*.png'):
        image = Image.open(filename)
        pix = np.array(image)
        lr = torch.Tensor(pix).permute(2,0,1).unsqueeze(0)

        ##### Input, Output Size #####
        print('Input Image Size:  {} x {}'.format(lr.size(2),lr.size(3)))
        print('Output Image Size: {} x {}'.format(lr.size(2)*args.scale,lr.size(3)*args.scale))

        ###### Calculate Operation Time #####
        time_list= []
            
        for i in range(10):
            # lr = torch.randn(input_size[0])
            hr = torch.randn(lr.size(0),lr.size(1),lr.size(2)*args.scale,lr.size(3)*args.scale)
            torch.cuda.synchronize()
            tic = time.time()
            sr_test = model(lr)
            torch.cuda.synchronize()
            toc = time.time() - tic
            if i>=1:
                time_list.append(toc)

        print('Operation time: {:.3f}s'.format(np.mean(np.array(time_list))))

        ###### Calculate FLOPs #####
        _output = (lr.size(2)*args.scale,lr.size(3)*args.scale)
        _input= (lr.size(2),lr.size(3))
        
        kernel_size = 3
        flops= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*args.n_colors*len(args.op1)
        ch_length=0
        for i in range (args.n_resblocks):
            ch_length += len(args.P[i])+len(args.R[i])+len(args.T[i])+len(args.PR[i])+len(args.PT[i])+len(args.RT[i])+len(args.PRT[i])
        # Resblocks
        flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*args.n_feats*2*(ch_length)
        # Global Skip
        flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*args.n_feats*len(args.op2)
        # Tail
        flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*len(args.op_last)*args.n_colors*(4**(args.scale//2))
        print('FLOPs: {:.1f}G\n'.format(flops/(10**9)))         

        ##### Get SR #######
        sr = model(lr)
        sr = sr.clamp(0, 255).round().div(255)

        save_image(sr[0], filename[:-4]+'_SR_'+str(args.scale)+'.png')


    torch.set_grad_enabled(True) 

if __name__ == '__main__':
    main()
