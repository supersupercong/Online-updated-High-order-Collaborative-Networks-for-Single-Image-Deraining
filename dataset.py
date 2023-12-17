import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset

import settings 
import random

class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = name
        self.mat_files = sorted(os.listdir(self.root_dir))
        self.patch_size = settings.patch_size
        self.file_num = len(self.mat_files)

        self.random_files = os.listdir(self.root_dir)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255

        random.shuffle(self.random_files)
        file_name_random = self.random_files[idx % self.file_num]
        img_file_random = os.path.join(self.root_dir, file_name_random)
        img_pair_random = cv2.imread(img_file_random).astype(np.float32) / 255

        if settings.aug_data:
            O, B = self.crop(img_pair, aug=True)
            O, B = self.flip(O, B)
            O, B = self.rotate(O, B)
        else:
            O, B = self.crop(img_pair, aug=False)
            O_random, B_random = self.crop(img_pair_random, aug=False)

        O_2 = cv2.pyrDown(O)
        O_4 = cv2.pyrDown(O_2)
        O_8 = cv2.pyrDown(O_4)

        B_2 = cv2.pyrDown(B)
        B_4 = cv2.pyrDown(B_2)
        B_8 = cv2.pyrDown(B_4)

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))

        O_2 = np.transpose(O_2, (2, 0, 1))
        O_4 = np.transpose(O_4, (2, 0, 1))
        O_8 = np.transpose(O_8, (2, 0, 1))

        B_2 = np.transpose(B_2, (2, 0, 1))
        B_4 = np.transpose(B_4, (2, 0, 1))
        B_8 = np.transpose(B_8, (2, 0, 1))

        O_2_random = cv2.pyrDown(O_random)
        O_4_random = cv2.pyrDown(O_2_random)
        O_8_random = cv2.pyrDown(O_4_random)

        B_2_random = cv2.pyrDown(B_random)
        B_4_random = cv2.pyrDown(B_2_random)
        B_8_random = cv2.pyrDown(B_4_random)

        O_random = np.transpose(O_random, (2, 0, 1))
        B_random = np.transpose(B_random, (2, 0, 1))

        O_2_random = np.transpose(O_2_random, (2, 0, 1))
        O_4_random = np.transpose(O_4_random, (2, 0, 1))
        O_8_random = np.transpose(O_8_random, (2, 0, 1))

        B_2_random = np.transpose(B_2_random, (2, 0, 1))
        B_4_random = np.transpose(B_4_random, (2, 0, 1))
        B_8_random = np.transpose(B_8_random, (2, 0, 1))

        sample = {'O': O, 'B': B, 'O_2': O_2, 'O_4': O_4, 'O_8': O_8, 'B_2': B_2, 'B_4': B_4, 'B_8': B_8, 'file_name':file_name[:-4],
                  'O_random': O_random, 'B_random': B_random, 'O_2_random': O_2_random, 'O_4_random': O_4_random, 'O_8_random': O_8_random,
                  'B_2_random': B_2_random, 'B_4_random': B_4_random, 'B_8_random': B_8_random, 'file_name_random':file_name_random[:-4]}

        return sample

    def crop(self, img_pair, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r+p_h, c+w: c+p_w+w]
        B = img_pair[r: r+p_h, c: c+p_w]

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))

        return O, B

    def flip(self, O, B):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
        return O, B

    def rotate(self, O, B):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        return O, B


class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = name
        self.mat_files = os.listdir(self.root_dir)
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        O = img_pair[:, w:]
        B = img_pair[:, :w]

        O_2 = cv2.pyrDown(O)
        O_4 = cv2.pyrDown(O_2)
        O_8 = cv2.pyrDown(O_4)

        B_2 = cv2.pyrDown(B)
        B_4 = cv2.pyrDown(B_2)
        B_8 = cv2.pyrDown(B_4)

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))

        O_2 = np.transpose(O_2, (2, 0, 1))
        O_4 = np.transpose(O_4, (2, 0, 1))
        O_8 = np.transpose(O_8, (2, 0, 1))

        B_2 = np.transpose(B_2, (2, 0, 1))
        B_4 = np.transpose(B_4, (2, 0, 1))
        B_8 = np.transpose(B_8, (2, 0, 1))
        # print('file_name',file_name[:-4])

        sample = {'O': O, 'B': B, 'O_2': O_2, 'O_4': O_4, 'O_8': O_8, 'B_2': B_2, 'B_4': B_4, 'B_8': B_8,'file_name':file_name[:-4]}

        return sample


class ShowDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        self.img_files = sorted(os.listdir(self.root_dir))
        self.file_num = len(self.img_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        print(img_file)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255

        h, ww, c = img_pair.shape
        w = int(ww / 2)

        #h_8 = h % 8
        #w_8 = w % 8
        if settings.pic_is_pair:
            O = img_pair[:, w:]
            B = img_pair[:, :w]
        else:
            O = img_pair[:, :]
            B = img_pair[:, :]

        O_2 = cv2.pyrDown(O)
        O_4 = cv2.pyrDown(O_2)
        O_8 = cv2.pyrDown(O_4)

        B_2 = cv2.pyrDown(B)
        B_4 = cv2.pyrDown(B_2)
        B_8 = cv2.pyrDown(B_4)

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))

        O_2 = np.transpose(O_2, (2, 0, 1))
        O_4 = np.transpose(O_4, (2, 0, 1))
        O_8 = np.transpose(O_8, (2, 0, 1))

        B_2 = np.transpose(B_2, (2, 0, 1))
        B_4 = np.transpose(B_4, (2, 0, 1))
        B_8 = np.transpose(B_8, (2, 0, 1))

        sample = {'O': O, 'B': B, 'O_2': O_2, 'O_4': O_4, 'O_8': O_8, 'B_2': B_2, 'B_4': B_4, 'B_8': B_8,'file_name':file_name[:-4]}

        return sample


if __name__ == '__main__':
    dt = TrainValDataset('val')
    print('TrainValDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    dt = TestDataset('test')
    print('TestDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    print('ShowDataset')
    dt = ShowDataset('test')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())
