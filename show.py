import os
import sys
import cv2
import argparse
import math
import numpy as np
import itertools

import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.autograd import Variable 
from torch.utils.data import DataLoader

import settings
from dataset import ShowDataset
from model import ODE_DerainNet 
from cal_ssim import SSIM

os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
#torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        
def PSNR(img1, img2):
    b,_,_,_=img1.shape
    #mse=0
    #for i in range(b):
    img1=np.clip(img1,0,255)
    img2=np.clip(img2,0,255)
    mse = np.mean((img1/ 255. - img2/ 255.) ** 2)#+mse
    if mse == 0:
        return 100
    #mse=mse/b
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) 

class Session:
    def __init__(self):
        self.show_dir = settings.show_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.show_dir)
        ensure_dir(settings.model_dir)
        logger.info('set show dir as %s' % settings.show_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        if len(settings.device_id) >1:
            self.net = nn.DataParallel(ODE_DerainNet()).cuda()
            #self.l2 = nn.DataParallel(MSELoss(),settings.device_id)
            #self.l1 = nn.DataParallel(nn.L1Loss(),settings.device_id)
            #self.ssim = nn.DataParallel(SSIM(),settings.device_id)
            #self.vgg = nn.DataParallel(VGG(),settings.device_id)
        else:
            self.net = ODE_DerainNet().cuda()
        self.ssim = SSIM().cuda()
        self.dataloaders = {}
        self.ssim=SSIM().cuda()
        self.a=0
        self.t=0
    def get_dataloader(self, dataset_name):
        dataset = ShowDataset(dataset_name)
        self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=1, 
                            shuffle=False, num_workers=1)
        return self.dataloaders[dataset_name]

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))
    def inf_batch(self, name, batch,i):
         # self.print_network(self.net)
        file_name= batch['file_name']
        print('file-name',file_name)
        file_name = str(file_name[0])
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O_2, B_2 = batch['O_2'].cuda(), batch['B_2'].cuda()
        O_4, B_4 = batch['O_4'].cuda(), batch['B_4'].cuda()
        O_8, B_8 = batch['O_8'].cuda(), batch['B_8'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        O_2, B_2 = Variable(O_2, requires_grad=False), Variable(B_2, requires_grad=False)
        O_4, B_4 = Variable(O_4, requires_grad=False), Variable(B_4, requires_grad=False)
        O_8, B_8 = Variable(O_8, requires_grad=False), Variable(B_8, requires_grad=False)
        with torch.no_grad():
            import time
            t0 = time.time()
            out1, out2, out3, multiexit2, multiexit4 = self.net(O, O_2, O_4)
            t1 = time.time()
            comput_time = t1 - t0
            print(comput_time)
            ssim = self.ssim(out1, B).data.cpu().numpy()
            psnr = PSNR(out1.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
            print('psnr:%4f-------------ssim:%4f'%(psnr, ssim))
            return out1, psnr, ssim, file_name

    def save_image(self, No, imgs, name, psnr, ssim, file_name):
        for i, img in enumerate(imgs):
            img = (img.cpu().data * 255).numpy()
            img = np.clip(img, 0, 255)
            img = np.transpose(img, (1, 2, 0))
            h, w, c = img.shape

            # img_file = os.path.join(self.show_dir, 'real_world_%s.png' % (file_name))
            img_file = os.path.join(self.show_dir, '%s.png' % (file_name))
            print(img_file)
            cv2.imwrite(img_file, img)


def run_show(ckp_name):
    sess = Session()
    sess.load_checkpoints(ckp_name)
    sess.net.eval()
    dataset = 'test'
    if settings.pic_is_pair is False:
        dataset = 'real-input-middle'
    dt = sess.get_dataloader(dataset)

    for i, batch in enumerate(dt):
        logger.info(i)
        if i>-1:
            imgs,psnr,ssim, file_name= sess.inf_batch('test', batch,i)
            sess.save_image(i, imgs, dataset, psnr, ssim, file_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='latest_net')

    args = parser.parse_args(sys.argv[1:])
    
    run_show(args.model)

