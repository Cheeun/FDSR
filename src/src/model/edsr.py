#### EDSR.FINAL############
from model import common
import math
from option import args
import torch
import torch.nn as nn

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]

        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
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
            common.ResBlock(
                conv, n_feats, kernel_size, p[i],r[i], t[i],pr[i],pt[i],rt[i],prt[i],op3_gate[i],skip_gate[i], res_scale=args.res_scale
            ) for i in range(n_resblocks)
        ]
        
        self.body = nn.Sequential(*m_body)
        self.body2 = conv(len(args.op2), n_feats, kernel_size)
        # self.body2 = conv(n_feats, n_feats, kernel_size)

        self.op2 = args.op2
        # self.body2_pruned = conv(n_feats, self.op2.size(1), kernel_size) # input_ch n_feats doesn't change
        # self.body2_pruned = conv(self.op1.size(1), self.op2.size(1), kernel_size) 
        
        self.op_last = args.op_last
        self.tail1 = conv(len(args.op_last), args.n_colors*(4**int(math.log(scale, 2))), kernel_size)
        # self.tail1 = conv(n_feats, args.n_colors*(4**int(math.log(scale, 2))), kernel_size)


        tail2 = [nn.PixelShuffle(2) for i in range(int(math.log(scale, 2)))] # only covers scale 2, 4
        self.tail2 = nn.Sequential(*tail2)

        ################ Second EDSR
        self.downshuffle = nn.PixelShuffle(0.5)

    def forward(self, x):
        x = x.cuda()
        x = self.sub_mean(x)

        x1 = self.head(x)

        res = self.body(x1)
        # print(res.shape)
        res_=torch.zeros(res.size(0),len(self.op2),res.size(2),res.size(3))
        # print(res_.shape)
        for i in range(len(self.op2)):
            res_[:,i,:,:]=res[:,self.op2[i],:,:]
        res_=res_.cuda()
        res2 = self.body2(res_)
        # res2 = self.body2(res)

        res2 += x1

        res2_=torch.zeros(res2.size(0),len(self.op_last),res2.size(2),res2.size(3))
        for i in range(len(self.op_last)):
            res2_[:,i,:,:]=res2[:,self.op_last[i],:,:] 
        res2_=res2_.cuda()
        x = self.tail1(res2_)
        # x = self.tail1(res2)


        x=self.tail2(x)

        x=self.add_mean(x)

        '''
        x= self.downshuffle(x)
        x = self.sub_mean(x)
        x1 = self.head(x) 
        res = self.body(x1)
        res2 = self.body2(res)
        res2 += x1
        x = self.tail1(res2)
        x = self.tail2(x)
        x = self.add_mean(x)
        '''

    
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
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))




    

    