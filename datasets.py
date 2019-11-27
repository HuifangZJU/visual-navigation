import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files = glob.glob(os.path.join(root, mode) + '/*.*')
        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        if mode == 'train':
            self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)


#path = './image_lists/list.txt'


class Yq21Dataset(Dataset):
    def __init__(self, path, transforms_=None, mirror_=False):
        self.transform = transforms.Compose(transforms_)
        f = open(path, 'r')
        self.files = f.readlines()
        self.mirror = mirror_
        f.close()

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')

        img_A1 = Image.open(img_path[0])
        w, h = img_A1.size
        img_A1 = img_A1.crop((0, 0, w, h))

        img_A2 = Image.open(img_path[1])
        w, h = img_A2.size
        img_A2 = img_A2.crop((0, 0, w, h))

        img_B = Image.open(img_path[2])
        w, h = img_B.size
        img_B = img_B.crop((0, 0, w, h))

        # mirror the inputs
        if self.mirror:
            if np.random.random() < 0.5:
                img_A1 = Image.fromarray(np.array(img_A1)[:, ::-1, :], 'RGB')
                img_A2 = Image.fromarray(np.array(img_A2)[:, ::-1, :], 'RGB')
                img_B = Image.fromarray(np.array(img_B)[:, ::-1], 'L')
        # crop the navigation map

        img_A1 = self.transform(img_A1)
        img_A2 = self.transform(img_A2)
        img_B = self.transform(img_B)
        img_A = torch.cat((img_A1, img_A2), 0)
        #print(img_A1.size())
        #print(img_A2.size())
        return {'A': img_A, 'B': img_B, 'C': img_path[0][-14:]}

    def __len__(self):

        return len(self.files)


class Yq21DatasetLSTM(Dataset):
    def __init__(self, path, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        #f = open(path, 'r')
        #self.files = f.readlines()
        #f.close()

        f = open(path[:-4]+'_id'+path[-4:], 'r')
        self.files_id = f.readlines()
        f.close()

        f = open(path[:-4] + '_all' + path[-4:], 'r')
        self.files_all = f.readlines()
        f.close()

        f = open(path[:-4] + '_id_all' + path[-4:], 'r')
        self.files_all_id = f.readlines()
        f.close()

    def __getitem__(self, index):
        steps = random.choice((4, 5, 6, 7, 8, 9, 10))
        #steps = 4
        #steps = random.choice((10, 12, 14, 16, 18, 20))
        #steps = random.choice((2, 2))
        input_a = []
        input_b = []
        img_id = self.files_id[index % len(self.files_id)].strip()
        img_id = img_id.split(' ')
        img_id = int(img_id[0])

        img_index = self.files_all_id[img_id % len(self.files_all_id)].strip()
        img_index = img_index.split(' ')
        img_index = int(img_index[1])
        randomflag = False
        #if np.random.random() < 0.5:
        #    randomflag = True
        for step in range(0, steps):
            next_id = img_id + step*3
            next_index = self.files_all_id[next_id % len(self.files_all_id)].strip()
            next_index = next_index.split(' ')
            next_index = int(next_index[1])
            if abs(next_index - img_index) > 50:
                continue

            img_path = self.files_all[next_id % len(self.files_all)].strip()

            img_path = img_path.split(' ')
            img_a1 = Image.open(img_path[0])
            img_a2 = Image.open(img_path[1])
            img_b = Image.open(img_path[2])

            # mirror the inputs
            if randomflag:
                img_a1 = Image.fromarray(np.array(img_a1)[:, ::-1, :], 'RGB')
                img_a2 = Image.fromarray(np.array(img_a2)[:, ::-1, :], 'RGB')
                img_b = Image.fromarray(np.array(img_b)[:, ::-1], 'L')

            img_a1 = self.transform(img_a1)
            img_a2 = self.transform(img_a2)
            img_b = self.transform(img_b)

            img_a = torch.cat((img_a1, img_a2), 0)

            input_a.append(img_a)
            input_b.append(img_b)

        return {'A': input_a, 'B': input_b}

    def __len__(self):
        return len(self.files_id)


class Yq21LSTMTest(Dataset):
    def __init__(self, path, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()

    def __getitem__(self, index):
        steps = 4
        input_a1=[]
        input_a2=[]
        input_b = []
        input_f =[]
        for step in range(0, steps):
            id = index + step * 3
            img_path = self.files[id % len(self.files)].strip()
            img_path = img_path.split(' ')

            img_A1 = Image.open(img_path[0])
            w, h = img_A1.size
            img_A1 = img_A1.crop((0, 0, w, h))

            orig_name = img_path[1]
            #new_name = orig_name[:-14] + '/navi_offset/' + orig_name[-14:-4] + '_2.jpg'
            #if step<steps-1:
            img_A2 = Image.open(orig_name)
            #else:
            #    img_A2 = Image.open('/mnt/data/huifang/YQ_RAW_MASK/2017_05_09/2017_05_09_drive_0001_sync/label_01/0000006667.jpg')
            w, h = img_A2.size
            img_A2 = img_A2.crop((0, 0, w, h))


            img_A1 = self.transform(img_A1)
            img_A2 = self.transform(img_A2)

            img_B = Image.open(img_path[2])
            img_B = self.transform(img_B)

            input_a1.append(img_A1)
            input_a2.append(img_A2)
            input_b.append(img_B)
            input_f.append(img_path[0][-14:])

        return {'A1': input_a1, 'A2': input_a2, 'B': input_b, 'F': input_f}

    def __len__(self):
        return len(self.files)

class Yq21Dataset_test(Dataset):
    def __init__(self, path, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')

        img_A1 = Image.open(img_path[0])
        w, h = img_A1.size
        img_A1 = img_A1.crop((0, 0, w, h))

        img_A2 = Image.open(img_path[1])
        w, h = img_A2.size
        img_A2 = img_A2.crop((0, 0, w, h))

        img_B = Image.open(img_path[2])
        w, h = img_B.size
        img_B = img_B.crop((0, 0, w, h))

        img_A1 = self.transform(img_A1)
        img_A2 = self.transform(img_A2)
        img_B = self.transform(img_B)

        return {'A1': img_A1, 'A2': img_A2, 'B': img_B}

    def __len__(self):
        return len(self.files)


class Yq21DatasetE2E(Dataset):
    def __init__(self, path, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        f = open(path[:-4]+'_id'+path[-4:], 'r')
        self.files_id = f.readlines()
        f.close()

        f = open(path[:-4] + '_all' + path[-4:], 'r')
        self.files_all = f.readlines()
        f.close()

        f = open(path[:-4] + '_id_all' + path[-4:], 'r')
        self.files_all_id = f.readlines()
        f.close()

    def __getitem__(self, index):
        steps = 4
        #steps = random.choice((10, 12, 14, 16, 18, 20))
        #steps = random.choice((2, 2))
        input_a1 = []
        input_a2 = []
        input_b = []
        img_id = self.files_id[index % len(self.files_id)].strip()
        img_id = img_id.split(' ')
        img_id = int(img_id[0])

        img_index = self.files_all_id[img_id % len(self.files_all_id)].strip()
        img_index = img_index.split(' ')
        img_index = int(img_index[1])

        cropcnt = round(np.random.random() * 20)
        for step in range(0, steps):
            next_id = img_id + step*3
            next_index = self.files_all_id[next_id % len(self.files_all_id)].strip()
            next_index = next_index.split(' ')
            next_index = int(next_index[1])
            if abs(next_index - img_index) > 50:
                continue

            img_path = self.files_all[next_id % len(self.files_all)].strip()

            img_path = img_path.split(' ')
            img = Image.open(img_path[0])
            navi = Image.open(img_path[1])
            v = eval(img_path[2])
            w = eval(img_path[3])

            # crop current perception
            # img_in: 314 648
            # img_out: 314 518
            keepsize = 628
            img = img.crop([cropcnt, 0, cropcnt + keepsize, img.size[1]])

            img = self.transform(img)
            navi = self.transform(navi)

            input_a1.append(img)
            input_a2.append(navi)
            input_b.append([v, w])

        return {'A1': input_a1, 'A2': input_a2[-1], 'B': input_b[-1]}

    def __len__(self):
        return len(self.files_id)


class Yq21DatasetE2E_test(Dataset):
    def __init__(self, path, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')

        img_a1 = Image.open(img_path[0])
        w, h = img_a1.size
        img_a1 = img_a1.crop((0, 0, w, h))

        img_a2 = Image.open(img_path[1])
        w, h = img_a2.size
        img_a2 = img_a2.crop((0, 0, w, h))

        v = eval(img_path[2])
        w = eval(img_path[3])

        img_a1 = self.transform(img_a1)
        img_a2 = self.transform(img_a2)
        motion = [v, w]
        return {'A1': img_a1, 'A2': img_a2, 'B': motion}

    def __len__(self):
        return len(self.files)