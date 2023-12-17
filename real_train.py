import os
import sys
import cv2
import argparse
import math

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import settings
from dataset import TrainValDataset, TestDataset
from model import ODE_DerainNet
from cal_ssim import SSIM

logger = settings.logger
os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
import numpy as np
from model import VGG


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def PSNR(img1, img2):
    b, _, _, _ = img1.shape
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100

    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.training_real_dir = settings.training_real_dir
        self.model_dir = settings.model_dir
        self.ssim_loss = settings.ssim_loss
        ensure_dir(settings.log_dir)
        ensure_dir(settings.training_real_dir)
        ensure_dir(settings.model_dir)
        ensure_dir('../log_test')
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)
        if len(settings.device_id) > 1:
            self.net = nn.DataParallel(ODE_DerainNet()).cuda()
        else:
            # torch.cuda.set_device(settings.device_id[0])
            self.net = ODE_DerainNet().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.celoss = nn.CrossEntropyLoss().cuda()
        self.bceloss = nn.BCEWithLogitsLoss().cuda()
        self.cosine = nn.CosineSimilarity().cuda()
        self.ssim = SSIM().cuda()
        self.kl = nn.KLDivLoss(reduce=False)
        self.vgg = VGG().cuda()
        self.step = 0
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.writers = {}
        self.dataloaders = {}
        self.opt_net = Adam(self.net.parameters(), lr=settings.lr)
        self.sche_net = MultiStepLR(self.opt_net, milestones=[settings.l1, settings.l2, settings.l3], gamma=0.1)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)
        out['lr'] = self.opt_net.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v)
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))
    def update_lr(self, lr):
        self.opt_net.param_groups[0]['lr'] = lr

    def get_dataloader(self, dataset_name):
        dataset = TrainValDataset(dataset_name)
        self.dataloaders = DataLoader(dataset, batch_size=self.batch_size,shuffle=True, num_workers=self.num_workers, drop_last=True)
        return iter(self.dataloaders)

    def get_test_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        self.dataloaders = DataLoader(dataset, batch_size=1,shuffle=False, num_workers=1, drop_last=False)
        return self.dataloaders

    def save_checkpoints_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock_net': self.step,
            'opt_net': self.opt_net.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt_net.load_state_dict(obj['opt_net'])
        self.step = obj['clock_net']
        self.sche_net.last_epoch = self.step

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))
    def pyramid_cl(self, clear, dehaze, haze):
        vgg_clear = self.vgg.forward(clear)
        vgg_dehaze = self.vgg.forward(dehaze)
        vgg_haze = self.vgg.forward(haze)
        loss_clear_dehaze_1 = self.l1(vgg_clear[0], vgg_dehaze[0])
        loss_dehaze_haze_1 = self.l1(vgg_haze[0], vgg_dehaze[0])
        loss_clear_dehaze_3 = self.l1(vgg_clear[1], vgg_dehaze[1])
        loss_dehaze_haze_3 = self.l1(vgg_haze[1], vgg_dehaze[1])
        loss_clear_dehaze_5 = self.l1(vgg_clear[2], vgg_dehaze[2])
        loss_dehaze_haze_5 = self.l1(vgg_haze[2], vgg_dehaze[2])
        loss_clear_dehaze_9 = self.l1(vgg_clear[3], vgg_dehaze[3])
        loss_dehaze_haze_9 = self.l1(vgg_haze[3], vgg_dehaze[3])
        loss_clear_dehaze_13 = self.l1(vgg_clear[4], vgg_dehaze[4])
        loss_dehaze_haze_13 = self.l1(vgg_haze[4], vgg_dehaze[4])
        loss1 = loss_clear_dehaze_1 / loss_dehaze_haze_1
        loss3 = loss_clear_dehaze_3 / loss_dehaze_haze_3
        loss5 = loss_clear_dehaze_5 / loss_dehaze_haze_5
        loss9 = loss_clear_dehaze_9 / loss_dehaze_haze_9
        loss13 = loss_clear_dehaze_13 / loss_dehaze_haze_13
        loss_total = 1/32 * loss1 + 1/16 * loss3 + 1/8 * loss5 + 1/4 * loss9 + 1 * loss13
        return loss_total

    def inf_batch(self, name, batch):
        if name == 'train':
            self.net.zero_grad()
        if self.step == 0:
            self.print_network(self.net)

        # sample = {'O': O, 'B': B, 'O_2': O_2, 'O_4': O_4, 'O_8': O_8, 'B_2': B_2, 'B_4': B_4, 'B_8': B_8}

        O, B = batch['O'].cuda(), batch['B'].cuda()
        O_2, B_2 = batch['O_2'].cuda(), batch['B_2'].cuda()
        O_4, B_4 = batch['O_4'].cuda(), batch['B_4'].cuda()
        O_8, B_8 = batch['O_8'].cuda(), batch['B_8'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        O_2, B_2 = Variable(O_2, requires_grad=False), Variable(B_2, requires_grad=False)
        O_4, B_4 = Variable(O_4, requires_grad=False), Variable(B_4, requires_grad=False)
        O_8, B_8 = Variable(O_8, requires_grad=False), Variable(B_8, requires_grad=False)

        O_random, B_random = batch['O_random'].cuda(), batch['B_random'].cuda()
        O_2_random, B_2_random = batch['O_2_random'].cuda(), batch['B_2_random'].cuda()
        O_4_random, B_4_random = batch['O_4_random'].cuda(), batch['B_4_random'].cuda()
        O_8_random, B_8_random = batch['O_8_random'].cuda(), batch['B_8_random'].cuda()
        O_random, B_random = Variable(O_random, requires_grad=False), Variable(B_random, requires_grad=False)
        O_2_random, B_2_random = Variable(O_2_random, requires_grad=False), Variable(B_2_random, requires_grad=False)
        O_4_random, B_4_random = Variable(O_4_random, requires_grad=False), Variable(B_4_random, requires_grad=False)
        O_8_random, B_8_random = Variable(O_8_random, requires_grad=False), Variable(B_8_random, requires_grad=False)

        print('file_name', batch['file_name'])
        out1, out2, out3, multiexit2, multiexit4 = self.net(O, O_2, O_4)

        # out1_random, out2_random, out3_random, multiexit2_random, multiexit4_random = self.net(O_random, O_2_random, O_4_random)

        # ssim1 = self.ssim(out1, B)
        # ssim2 = self.ssim(out2, B)
        # ssim3 = self.ssim(out3, B)
        b,c,h,w = O.size()
        # R = (O - out1)
        # R_random = (O_random-out1_random)
        # loss_cosine = torch.abs(self.cosine(R,R_random).sum().sum().sum()-1)
        # print('loss-cosine:',loss_cosine)
        # print('loss_cosine', loss_cosine)
        R = (O - out1)
        # print('R-size',R.size())
        R_random = (O_random-B_random)
        # print('R_random-size', R_random.size())

        import torch.nn.functional as F
        # celoss = torch.abs(self.cosine(R, R_random).sum()-1)
        # celoss = self.bceloss(R, R_random)

        loss_content = self.l1(out1,B)
        # print('loss_content', loss_content)
        loss_rain_kl = self.kl(R, R_random.detach()).sum()

        ssim1 = self.ssim(out1, B)

        loss = loss_content + settings.value_cl * loss_rain_kl
        if name == 'train':
            loss.backward()
            self.opt_net.step()
        losses = {'L1loss2': loss,'loss-ce': loss_rain_kl}
        ssimes = {'ssim1': ssim1}
        losses.update(ssimes)
        self.write(name, losses)

        return out1

    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)
        h, w = pred.shape[-2:]
        gen_num = (1, 1)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row + h, col: col + w] = tmp
        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)

    def save_image_truple(self, name, img_lists,epoch):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)
        h, w = pred.shape[-2:]
        gen_num = (1, 1)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row + h, col: col + w] = tmp
        img_file = os.path.join(self.log_dir, '%s_%d.png' % (name, epoch))
        cv2.imwrite(img_file, img)

    def inf_batch_test(self, name, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O_2, B_2 = batch['O_2'].cuda(), batch['B_2'].cuda()
        O_4, B_4 = batch['O_4'].cuda(), batch['B_4'].cuda()
        O_8, B_8 = batch['O_8'].cuda(), batch['B_8'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        O_2, B_2 = Variable(O_2, requires_grad=False), Variable(B_2, requires_grad=False)
        O_4, B_4 = Variable(O_4, requires_grad=False), Variable(B_4, requires_grad=False)
        O_8, B_8 = Variable(O_8, requires_grad=False), Variable(B_8, requires_grad=False)

        with torch.no_grad():
            out1, out2, out3, multiexit2, multiexit4 = self.net(O, O_2, O_4)

        l1_loss = self.l1(out1, B)
        ssim1 = self.ssim(out1, B)
        psnr1 = PSNR(out1.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        losses = {'L1 loss': l1_loss}
        ssimes = {'ssim1': ssim1}
        losses.update(ssimes)

        return l1_loss.data.cpu().numpy(), ssim1.data.cpu().numpy(), psnr1
    def updating_dataset(self, batch, name, root_dir,epoch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O_2, B_2 = batch['O_2'].cuda(), batch['B_2'].cuda()
        O_4, B_4 = batch['O_4'].cuda(), batch['B_4'].cuda()
        O_8, B_8 = batch['O_8'].cuda(), batch['B_8'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        O_2, B_2 = Variable(O_2, requires_grad=False), Variable(B_2, requires_grad=False)
        O_4, B_4 = Variable(O_4, requires_grad=False), Variable(B_4, requires_grad=False)
        O_8, B_8 = Variable(O_8, requires_grad=False), Variable(B_8, requires_grad=False)

        with torch.no_grad():
            out1, out2, out3, multiexit2, multiexit4 = self.net(O, O_2, O_4)

        gt = out1.cpu().data
        rain = O.cpu().data

        gt = gt * 255
        rain = rain * 255
        rain = np.clip(rain, 0, 255)
        gt = np.clip(gt, 0, 255)
        h, w = gt.shape[-2:]
        img = np.zeros((1 * h, 2 * w, 3))
        gt = np.transpose(gt[0], (1, 2, 0))
        rain = np.transpose(rain[0], (1, 2, 0))
        img[:, :w, :] = gt
        img[:, w:, :] = rain
        img_file = os.path.join(root_dir, '%s.png' % (name))
        cv2.imwrite(img_file, img)

        save_root_dir = root_dir + 'saveimg/%d' % epoch
        ensure_dir(save_root_dir)
        img_file_save = os.path.join(save_root_dir, '%s.png' % (name))
        img_gt = np.zeros((1 * h, 1 * w, 3))
        img_gt[:, :, :] = gt
        cv2.imwrite(img_file_save, img_gt)


def run_train_val(ckp_name_net='latest_net'):
    sess = Session()
    sess.load_checkpoints_net(ckp_name_net)
    sess.tensorboard('train')
    while sess.step < settings.total_step + 1:
        dt_train = sess.get_dataloader(settings.real_data_dir)
        sess.sche_net.step()
        # sess.update_lr(0.000001)
        sess.net.train()
        try:
            batch_t = next(dt_train)
        except StopIteration:
            dt_train = sess.get_dataloader(settings.real_data_dir)
            batch_t = next(dt_train)
        pred_t = sess.inf_batch('train', batch_t)
        if sess.step % (settings.one_epoch_real) == 0:
            sess.save_checkpoints_net('net_finetune_%d_epoch' % int((sess.step - settings.total_step_initial) / settings.one_epoch_real))
        # if sess.step % (settings.one_epoch_real * 1) == 0:
            # sess.save_image('train', [batch_t['O'], pred_t, batch_t['B']])
        print(batch_t['file_name'][0],int((sess.step - settings.total_step_initial) / settings.one_epoch_real))
        sess.save_image_truple(batch_t['file_name'][0], [batch_t['O'], pred_t, batch_t['B']], int((sess.step - settings.total_step_initial) / settings.one_epoch_real))
        if int((sess.step - settings.total_step_initial) / settings.one_epoch_real)==6:
            sess.update_lr(0.000001)
        if int((sess.step - settings.total_step_initial) / settings.one_epoch_real)==8:
            sess.update_lr(0.0000005)
        if int((sess.step - settings.total_step_initial) / settings.one_epoch_real)==10:
            sess.update_lr(0.0000001)
        if int((sess.step - settings.total_step_initial) / settings.one_epoch_real)==12:
            sess.update_lr(0.00000005)
        if int((sess.step - settings.total_step_initial) / settings.one_epoch_real)==14:
            sess.update_lr(0.00000001)
        if int((sess.step - settings.total_step_initial) / settings.one_epoch_real)==16:
            sess.update_lr(0.000000005)

        if sess.step % (settings.updating_epoch*settings.one_epoch_real) == 0:
            dt_val = sess.get_test_dataloader(settings.real_data_dir)
            sess.net.eval()
            for i, batch_v in enumerate(dt_val):
                print('updating-dataset', i, batch_v['file_name'][0], batch_v['O'].size())
                sess.updating_dataset(batch_v, batch_v['file_name'][0], settings.real_data_dir, int((sess.step - settings.total_step_initial) / settings.one_epoch_real))
        sess.step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model_1', default='latest_net')

    args = parser.parse_args(sys.argv[1:])
    run_train_val(args.model_1)