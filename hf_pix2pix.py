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

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=10, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="graph-lstm", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between model checkpoints')
opt = parser.parse_args()
print(opt)

os.makedirs('/mnt/data/huifang/directional_result/images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('/mnt/data/huifang/directional_result/saved_models/%s' % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(1)
# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height//2**4, opt.img_width//2**4)
# Initialize generator and discriminator
orig_generator = GeneratorUNet()
generator = GeneratorLSTM()
discriminator = Discriminator()
#lstm = CLSTMblock()

if cuda:
    generator = generator.cuda()
    orig_generator = orig_generator.cuda()
    discriminator = discriminator.cuda()
    #lstm = lstm.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('/mnt/data/huifang/directional_result/saved_models/%s/g_%d.pth' % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load('/mnt/data/huifang/directional_result/saved_models/%s/d_%d.pth' % (opt.dataset_name, opt.epoch)))
else:
    generator.apply(weights_init_normal)
    path = '/mnt/data/huifang/directional_result/saved_models/yq21-graph/generator_199.pth'
    saved_model = torch.load(path)
    generator_dict = generator.state_dict()
    state_dict = {k: v for k, v in saved_model.items() if k in generator_dict.keys()}
    generator_dict.update(state_dict)
    generator.load_state_dict(generator_dict)
    discriminator.load_state_dict(torch.load('/mnt/data/huifang/directional_result/saved_models/yq21-graph/discriminator_199.pth'))

orig_generator.load_state_dict(torch.load('/mnt/data/huifang/directional_result/saved_models/yq21-graph/generator_199.pth'))
#unet encode
'''
for param in generator.down1.parameters():
    param.requires_grad = False
for param in generator.down2.parameters():
    param.requires_grad = False
for param in generator.down3.parameters():
    param.requires_grad = False
for param in generator.down4.parameters():
    param.requires_grad = False
for param in generator.down5.parameters():
    param.requires_grad = False
for param in generator.down6.parameters():
    param.requires_grad = False
for param in generator.down7.parameters():
    param.requires_grad = False
'''


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#optimizer_T = torch.optim.Adam(lstm.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# Configure dataloaders
transforms_ = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]


testdataloader = DataLoader(Yq21DatasetLSTM("./image_lists/graph_nav_train.txt", transforms_=transforms_),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
test_val_dataloader = DataLoader(Yq21LSTMTest("./image_lists/graph_nav_test.txt", transforms_=transforms_, ),
                            batch_size=1, shuffle=True, num_workers=opt.n_cpu)
test_samples = iter(test_val_dataloader)


# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def init_hidden(batch_size, hidden_c, shape):
    return (Variable(torch.zeros(batch_size,
                                 hidden_c,
                                 shape[0],
                                 shape[1])).cuda(),
            Variable(torch.zeros(batch_size,
                                 hidden_c,
                                 shape[0],
                                 shape[1])).cuda())


(h0, c0) = init_hidden(opt.batch_size, 512, (1, 2))
(hn, cn) = init_hidden(opt.batch_size, 512, (1, 2))
(hn_test, cn_test) = init_hidden(opt.batch_size, 512, (1, 2))

def sample_images(batches_done):
    '''
    batch = next(iter(test_val_dataloader))
    real_A = batch['A'][0]
    real_B = batch['B'][0]
    for step in range(1, len(batch['A'])):
        real_A = torch.cat((real_A, batch['A'][step]), 0)
        real_B = torch.cat((real_B, batch['B'][step]), 0)
    real_A1 = real_A[:, :3, :, :]
    real_A2 = real_A[:, -3:, :, :]

    # Model inputs
    real_A = Variable(real_A.type(Tensor))
    real_B = Variable(real_B.type(Tensor))
    fake_B = generator(real_A)
    fake_B = torch.unsqueeze(fake_B, dim=0)

    outputs = lstm(fake_B)
    real_A1 = real_A1[2:, :, :, :]
    real_A2 = real_A2[2:, :, :, :]
    real_B = real_B[2:, :, :, :]
    fake_B_T = outputs[2]
    for step in range(3, len(outputs)):
        fake_B_T = torch.cat((fake_B_T, outputs[step]), 0)
    img_sample = torch.cat((real_A1.data, real_A2.data), -2)
    pre_sample = torch.cat((real_B.data, fake_B_T.data), -2)

    save_image(img_sample, 'images/%s/%s_img.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)
    save_image(pre_sample, 'images/%s/%s_pre.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)
    '''
    """Saves a generated sample from the validation set"""
    test_batch = next(test_samples)
    test_a1 = test_batch['A1'][0]
    test_a2 = test_batch['A2'][0]
    test_real_b = test_batch['B'][0]
    for step in range(1, len(test_batch['A1'])):
        test_a1 = torch.cat((test_a1, test_batch['A1'][step]), 0)
        test_a2 = torch.cat((test_a2, test_batch['A2'][step]), 0)
        test_real_b = torch.cat((test_real_b, test_batch['B'][step]), 0)
    test_real_a = torch.cat((test_a1, test_a2), 1)
    test_real_a = Variable(test_real_a.type(Tensor))
    test_real_b = Variable(test_real_b.type(Tensor))
    test_a1 = Variable(test_a1.type(Tensor))
    test_a2 = Variable(test_a2.type(Tensor))
    #global hn_test, cn_test
    #(test_fake_b, hn_test, cn_test) = generator(test_real_a, hn_test, cn_test)
    test_fake_b = generator(test_real_a)
    comp_fake_b = orig_generator(test_real_a)
    #test_real_a1 = test_real_a1[2:, :, 5:-5, 5:-5]
    #test_real_a2 = test_real_a2[2:, :, 5:-5, 5:-5]
    #test_real_b = test_real_b[2:, :, 5:-5, 5:-5]
    #test_fake_b = test_fake_b[:, :, 5:-5, 5:-5]

    img_sample = torch.cat((test_a1.data, test_a2.data), -2)
    pre_sample = torch.cat((test_real_b.data, comp_fake_b.data), -2)
    pre_sample = torch.cat((pre_sample.data, test_fake_b.data), -2)
    #pre_sample = test_fake_b

    save_image(img_sample, '/mnt/data/huifang/directional_result/images/%s/%s_img.png' % (opt.dataset_name, batches_done), nrow=4, normalize=True)
    save_image(pre_sample, '/mnt/data/huifang/directional_result/images/%s/%s_pre.png' % (opt.dataset_name, batches_done), nrow=4, normalize=True)

# ----------
#  Training
# ----------


prev_time = time.time()

log = open('logs/%s_log.txt' % opt.dataset_name, mode='w')
log.write("batch-size : %d \n" % opt.batch_size)
log.write("epoch batch_id loss_D loss_G loss_pixel loss_GAN\n")
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(testdataloader):
        real_A = batch['A'][0]
        real_B = batch['B'][0]
        for step in range(1, len(batch['A'])):
            real_A = torch.cat((real_A, batch['A'][step]), 0)
            real_B = torch.cat((real_B, batch['B'][step]), 0)
        valid = Variable(Tensor(np.ones((real_B.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_B.size(0), *patch))), requires_grad=False)

        # Model inputs
        real_A = Variable(real_A.type(Tensor))
        real_B = Variable(real_B.type(Tensor))
        fake_B = generator(real_A)
        #h0.data = hn.data
        #c0.data = hn.data
        #print(h0.data)
        #test = input()
        #(fake_B, hn, cn) = generator(real_A, h0, c0)
        # ---------------------
        #  Train LSTM
        # --------------------
        optimizer_G.zero_grad()

        # GAN loss
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # Total loss
        loss_G = loss_GAN + lambda_pixel*loss_pixel
        #loss_G = loss_pixel
        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(testdataloader) + i
        batches_left = opt.n_epochs * len(testdataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        if i % 10 == 0:
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(testdataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_pixel.item(), loss_GAN.item(),
                                                        time_left))
        # Write log to file
        if i % 500 == 0:
            log.write("%d %d %f %f %f %f \n" % (epoch, i, loss_D.item(), loss_G.item(), loss_pixel.item(), loss_GAN.item()))
        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), '/mnt/data/huifang/directional_result/saved_models/%s/g_%d.pth' %
                   (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), '/mnt/data/huifang/directional_result/saved_models/%s/d_%d.pth'
                   % (opt.dataset_name, epoch))
        #hc = torch.cat((torch.squeeze(hn), torch.squeeze(cn)), 1)
        #hc = hc.cpu()
        #hc = hc.detach().numpy()
        #np.savetxt('/mnt/data/huifang/directional_result/saved_models/%s/hc_%d.txt' % (opt.dataset_name, epoch),
        #           hc, fmt='%.5f')
log.close()
