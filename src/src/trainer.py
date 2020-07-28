import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
import time

import numpy as np

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)

            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release(), 
                ))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()

        ################## get_#_params ####################
        n_params = 0
        for p in list(self.model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            n_params += nn
        self.ckp.write_log('Parameters: {:.1f}K'.format(n_params/(10**3)))
        
       
        if self.args.save_results:
            self.ckp.begin_background()
        ################### Test for Single Image ######################
        
        time_list= []
        for idx_scale, scale in enumerate(self.scale):
            
            for i in range(10):
                lr = torch.randn(1,3,1280//self.args.scale[0], 720//self.args.scale[0]) 
                hr = torch.randn(1,3,1280,720) # HD image
                torch.cuda.synchronize()
                tic = time.time()
                sr = self.model(lr, idx_scale)
                torch.cuda.synchronize()
                toc = time.time() - tic
                # print(toc)
                if i>=1:
                    time_list.append(toc)
                sr = utility.quantize(sr, self.args.rgb_range)
                save_list = [sr]
                if self.args.save_gt:
                    save_list.extend([lr, hr])
                best = self.ckp.log.max(0)
        
            # print("average",np.mean(np.array(time_list)))
            self.ckp.write_log('Per 1 HD image (1280x720):')
            self.ckp.write_log('Average operation time: {:.3f}s'.format(np.mean(np.array(time_list))))

            _output = (1280,720)
            _input= (_output[0]//self.args.scale[0],_output[1]//self.args.scale[0])

            kernel_size = 3
            flops= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*self.args.n_colors*len(self.args.op1)
            # flops= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*self.args.n_colors*(self.args.n_feats)

            ch_length=0
            for i in range (self.args.n_resblocks):
                # ch_length += 64 # for backbone
                ch_length += len(self.args.P[i])+len(self.args.R[i])+len(self.args.T[i])+len(self.args.PR[i])+len(self.args.PT[i])+len(self.args.RT[i])+len(self.args.PRT[i])
            
            # Resblocks
            flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*self.args.n_feats*2*(ch_length)
            # flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*self.args.n_feats*self.args.n_feats*2*self.args.n_resblocks # baseline

            # Global Skip
            flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*self.args.n_feats*len(self.args.op2)
            # flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*self.args.n_feats*self.args.n_feats # baseline

            # Tail
            flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*len(self.args.op_last)*self.args.n_colors*(4**(self.args.scale[0]//2))
            # flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*(self.args.n_feats)*self.args.n_colors*(4**(self.args.scale[0]//2))
            # if there is tail conv, instead of above,
            # flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*self.args.n_feats*self.args.n_feats*4*(self.args.scale[0]//2) # do conv twice if scale is 4
            # flops+= (_input[0]*_input[1])*2*(kernel_size*kernel_size)*self.args.n_feats*self.args.n_colors

            self.ckp.write_log('FLOPs: {:.1f}G'.format(flops/(10**9)))
            
        
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):

                d.dataset.set_scale(idx_scale)
                
                i=0
                dataset_size =0
                for lr, hr, filename in tqdm(d, ncols=80):
                    i+=1
                    lr, hr = self.prepare(lr, hr)
                    dataset_size += lr.size(2)*lr.size(3)

                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr( sr, hr, scale, self.args.rgb_range, dataset=d)
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} \t(Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
                
        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        torch.set_grad_enabled(True) 

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

