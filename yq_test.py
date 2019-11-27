import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc as sm
from scipy.optimize import curve_fit

from os import mkdir

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=199, help='epoch to start training from')
parser.add_argument('--dataset_name', type=str, default="yq21-half", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=10000, help='interval between sampling of images from generators')
opt = parser.parse_args()
print(opt)
os.makedirs('/mnt/data/huifang/directional_result/%s' % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)


generator = GeneratorUNet()
discriminator = Discriminator()
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

generator.load_state_dict(torch.load('/mnt/data/huifang/directional_result/saved_models/%s/generator_%d.pth' % (opt.dataset_name, opt.epoch)))
discriminator.load_state_dict(torch.load('/mnt/data/huifang/directional_result/saved_models/%s/discriminator_%d.pth' % (opt.dataset_name, opt.epoch)))


# Configure dataloaders
transforms_ = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
transform = transforms.Compose(transforms_)

#datapath = "./image_lists/test_south_list_crossid.txt"
datapath = "./image_lists/test_yq21_half.txt"

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

with open(datapath, 'r') as f:
    lines = f.readlines()
num_test = len(lines)

IMAGE_SIZE = (314, 648)
DEPLOY_SIZE = (128, 256)
SCALE = (IMAGE_SIZE[0]/DEPLOY_SIZE[0], IMAGE_SIZE[1]/DEPLOY_SIZE[1])
flag_value = []
assign_flag = 0
# ----------
#  Testing
# ----------
for instance in range(0, len(lines)):
    start = time.time()
    line = lines[instance]
    imagepair = line.strip().split()
    #crossid = imagepair[3]
    if instance%100 == 0:
        print(instance)
    #if int(crossid) == 0:
    #    continue
    #print(instance)
    perception = imagepair[0]
    navigation = imagepair[1]
    real_mask = imagepair[2]
    real_A1 = Image.open(perception)
    real_A2 = Image.open(navigation)
    real_B = Image.open(real_mask)
    real_A1 = transform(real_A1)
    real_A2 = transform(real_A2)
    real_B = transform(real_B)
    real_A = torch.zeros([1, 6, 128, 256])
    real_A[0, :, :, :] = torch.cat((real_A1, real_A2), 0)
    real_A1 = real_A1.view((1, 3, 128, 256))
    real_A2 = real_A2.view((1, 3, 128, 256))

    fake_B = generator(real_A.cuda())

    fake_B = fake_B.cpu()
    fake_B = fake_B.detach().numpy()
    fake_B = fake_B[0, 0, :, :]
    np.savetxt('/mnt/data/huifang/directional_result/%s/' % opt.dataset_name + perception[-14:]
               + '.txt', fake_B, fmt='%.5f')
    end = time.time()
    '''
    print(torch.unique(real_B))
    print(type(fake_B))
    print(fake_B.shape)
    test = input()
    endnetwork = time.time()
    if assign_flag == 0:
        flag_value = torch.unique(real_B)
        flag_value = flag_value.sort()[0]
        assign_flag = 1
    intersection = 0
    union = 0
    fake_path = np.zeros([IMAGE_SIZE[0], IMAGE_SIZE[1]])
    fake_path_flag = np.zeros([IMAGE_SIZE[0], IMAGE_SIZE[1]])

    for i in range(2, IMAGE_SIZE[0]-2):
        for j in range(2, IMAGE_SIZE[1]-2):
            fake_path[i, j] = fake_path_flag[i, j]
            if fake_path_flag[i, j] == 1:
                cnt = 0
                cnt0 = 0
                cnt2 = 0
                for m in [-2, -1, 0, 1, 2]:
                    for n in [-2, -1, 0, 1, 2]:
                        if fake_path_flag[i+m, j+n] == 1:
                           cnt += 1
                        if fake_path_flag[i+m, j+n] == 0:
                            cnt0 += 1
                        if fake_path_flag[i+m, j+n] == 2:
                           cnt2 += 1
                if cnt < 18:
                    if cnt0 > cnt2:
                        fake_path[i, j] = 0
                    else:
                        fake_path[i, j] = 2
            
            for channel in [1]:
                if channel == real_path[i, j]:
                    gtimage[i, j, channel] = min(gtimage[i, j, channel], 155)
                    gtimage[i, j, channel] += 100
                    gtimage[i, j, channel-1] = max(gtimage[i, j, channel], 80)
                    gtimage[i, j, channel-1] -= 80
                    gtimage[i, j, channel+1] = max(gtimage[i, j, channel], 80)
                    gtimage[i, j, channel+1] -= 80
                #else:
                #    gtimage[i, j, channel] = max(gtimage[i, j, channel], 80)
                #    gtimage[i, j, channel] -= 60

                if channel == fake_path[i, j]:
                    preimage[i, j, channel] = min(preimage[i, j, channel], 155)
                    preimage[i, j, channel] += 100
                #else:
                #    preimage[i, j, channel] = max(preimage[i, j, channel], 80)
                #    preimage[i, j, channel] -= 60
            
            if round(mask[i, j]) == 1 and fake_path[i, j] == 1:
                intersection += 1
            if round(mask[i, j]) == 1 or fake_path[i, j] == 1:
                union += 1
    endiou = time.time()
    iou = intersection / union
    if int(crossid) == 0:
        straightiou += iou
        straightcnt += 1
    else:
        crossiou += iou
        crosscnt += 1
    #plt.imsave('/mnt/data/huifang/directional_result/%s/' % opt.dataset_name + perception[-14:]+'.png', fake_path_flag)

    plt.subplot(2, 2, 1)
    plt.title("perception")
    plt.imshow(perimage)
    plt.subplot(2, 2, 2)
    plt.title("navigation")
    plt.imshow(navimage)
    plt.subplot(2, 2, 3)
    plt.title("ground truth")
    plt.imshow(gtimage)
    plt.subplot(2, 2, 4)
    plt.title("prediction")
    plt.imshow(preimage)
    plt.show()
    plt.savefig('/mnt/data/huifang/directional_result/%s/' % opt.dataset_name + perception[-14:])
    print("current iou is %.2f" % iou)
    print(" time costs are : input %.2f; network %.2f; flag %.2f; iou %.2f" % ((endinput-start), (endnetwork-endinput),
                                                                              (endflag-endnetwork), (endiou-endflag)))
                                                                              '''
    #print(end-start)
    #test = input()
