import os 
import copy
import random
import numpy as np
from numpy.fft import *
from pywt import dwt2, wavedec2

import torch
from torch.utils import data
from torchvision import get_image_backend

AVAILABLE_EXTENSIONS = ['.npy', '.tif', '.tiff']
       
class NpyFolder(data.Dataset): 
    def __init__(self, root, transform=None, target_transform=None):
        classes, class_to_idx = self.find_classes(root)
        imgs = self.make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(AVAILABLE_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):
        path, target = self.imgs[index] 
        img = self.img_enhence(np.load(path))
        img = np.transpose(img, (2, 1, 0))

        # import cv2
        # img = np.transpose(img, (2, 1, 0))
        # cv2.imwrite("./image/" + str(target) + "_" + str(index) + ".jpg", img)
        # cv2.imwrite("./image/" + str(target) + "_" + str(index) + "_1" + ".jpg", img[:, :, 0])
        # cv2.imwrite("./image/" + str(target) + "_" + str(index) + "_2" + ".jpg", img[:, :, 1])
        # cv2.imwrite("./image/" + str(target) + "_" + str(index) + "_3" + ".jpg", img[:, :, 2])

        if '/' in path:
            target = int(path.split('/')[-1].split('_')[0])
        elif '\\' in path:
            target = int(path.split('\\')[-1].split('_')[0])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img.float(), torch.tensor(target)

    def __len__(self):
        return len(self.imgs)

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in AVAILABLE_EXTENSIONS)

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def make_dataset(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.is_image_file(fname):
                        path = os.path.join(root, fname)   
                        item = (path, class_to_idx[target])  
                        images.append(item)
        return images
    
    def img_enhence(self, img):
        signal = img
        signal = np.concatenate([signal, signal], axis=0)
        signal = np.expand_dims(signal, axis=0)        
        signal = signal.repeat(len(signal[0]), axis=0)
        signal = (signal-np.min(signal))/(np.max(signal)-np.min(signal)) * 255.0
        signal_backup_1 = copy.deepcopy(signal)
        signal_backup_2 = copy.deepcopy(signal)

        s = []
        s.append(signal)                                        # 数组中第一个，原数据(120, 120)
        s.append(abs(fft(signal)))                              # 数组中第二个，原数据的傅里叶变换(120, 120)

        s.append(signal.T)                                      # 数组中第三个，原数据的转置(120, 120)
        s.append(abs(fft(signal.T)))                            # 数组中第四个，原数据的转置的傅里叶变换(120, 120)
        
        for ii in range(len(signal_backup_1)):
            signal_backup_1[ii] = np.roll(signal_backup_1[ii], 1)/2 + np.roll(signal_backup_1.T[ii], 1)/2         
        s.append(signal_backup_1)                                 # 数组中第五个，原数据依次右移1位/2+原数据依次上移2幂次位/2 (120, 120)
        s.append(abs(fft(signal_backup_1)))                       # 数组中第六个，数组中第五个数据的傅里叶变换(120, 120)
        
        for ii in range(len(signal_backup_2)):
            signal_backup_2[ii] = np.roll(signal_backup_2[ii], -1)/2 + np.roll(signal_backup_2.T[ii], 1)/2
        s.append(signal_backup_2)                                 # 数组中第七个，原数据依次左移1位/2+原数据依次上移2幂次位/2 (120, 120) 
        s.append(abs(fft(signal_backup_2)))                       # 数组中第八个，数组中第八个数据的傅里叶变换(120, 120)

        cA1, (cH1, cV1, cD1) = dwt2(signal, 'haar')               # 单级小波分解，返回分别为低频分量，水平高频，竖直高频，对角高频
        cA2, (cH2, cV2, cD2) = dwt2(signal.T, 'haar')
        cA3, (cH3, cV3, cD3) = dwt2(signal_backup_1, 'haar')
        cA4, (cH4, cV4, cD4) = dwt2(signal_backup_2, 'haar')

        s.append(np.concatenate([np.concatenate([cA1, cA2], axis=1), 
                np.concatenate([cA3, cA4], axis=1)], axis=0))           # 数组中第九个，小波变换，原数据(120, 120)
        s.append(np.concatenate([np.concatenate([cH1, cH2], axis=1), 
                np.concatenate([cH3, cH4], axis=1)], axis=0))           # 数组中第十个，小波变换，原数据(120, 120)
        s.append(np.concatenate([np.concatenate([cV1, cV2], axis=1), 
                np.concatenate([cV3, cV4], axis=1)], axis=0))           # 数组中第十一个，小波变换，原数据(120, 120)
        s.append(np.concatenate([np.concatenate([cD1, cD2], axis=1), 
                np.concatenate([cD3, cD4], axis=1)], axis=0))           # 数组中第十二个，小波变换，原数据(120, 120)

        for ii in range(len(s)):
            s[ii] = (s[ii]-np.min(s[ii]))/(np.max(s[ii])-np.min(s[ii])) * 255.0
        s = np.array(s)

        channel1 = np.concatenate([np.concatenate([s[0], s[6]], axis=1),    # 合并时域，1 3 5 7通道
                np.concatenate([s[4], s[2]], axis=1)], axis=0)              
        channel2 = np.concatenate([np.concatenate([s[1], s[7]], axis=1),    # 合并频域，2 4 6 8通道
                np.concatenate([s[5], s[3]], axis=1)], axis=0)
        channel3 = np.concatenate([np.concatenate([s[8], s[9]], axis=1),    # 合并小波域，9 10 11 12通道
                np.concatenate([s[10], s[11]], axis=1)], axis=0)

        final_img = np.array([channel1, channel2, channel3])
        return final_img

class ComNpyFolder(data.Dataset): 
    def __init__(self, root, transform=None, target_transform=None):
        classes, class_to_idx = self.find_classes(root)
        imgs = self.make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(AVAILABLE_EXTENSIONS)))
 
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):
        path, target = self.imgs[index] 
        img = self.img_enhence(np.load(path))
        img = np.transpose(img, (2, 1, 0))

        # import cv2
        # img = np.transpose(img, (2, 1, 0))
        # cv2.imwrite("./image/" + str(target) + "_" + str(index) + ".jpg", img)
        # cv2.imwrite("./image/" + str(target) + "_" + str(index) + "_1" + ".jpg", img[:, :, 0])
        # cv2.imwrite("./image/" + str(target) + "_" + str(index) + "_2" + ".jpg", img[:, :, 1])
        # cv2.imwrite("./image/" + str(target) + "_" + str(index) + "_3" + ".jpg", img[:, :, 2])

        if '/' in path:
            target = int(path.split('/')[-1].split('_')[0])
        elif '\\' in path:
            target = int(path.split('\\')[-1].split('_')[0])
        if self.transform is not None:
            imgL = self.transform(img)
            imgR = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgL.float(), imgR.float(), torch.tensor(target)

    def __len__(self):
        return len(self.imgs)

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in AVAILABLE_EXTENSIONS)

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def make_dataset(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.is_image_file(fname):
                        path = os.path.join(root, fname)   
                        item = (path, class_to_idx[target])  
                        images.append(item)
        return images
    
    def img_enhence(self, img):
        signal = img
        signal = np.concatenate([signal, signal], axis=0)
        signal = np.expand_dims(signal, axis=0)        
        signal = signal.repeat(len(signal[0]), axis=0)
        signal = (signal-np.min(signal))/(np.max(signal)-np.min(signal)) * 255.0
        signal_backup_1 = copy.deepcopy(signal)
        signal_backup_2 = copy.deepcopy(signal)

        s = []
        s.append(signal)                                        # 数组中第一个，原数据(120, 120)
        s.append(abs(fft(signal)))                              # 数组中第二个，原数据的傅里叶变换(120, 120)

        s.append(signal.T)                                      # 数组中第三个，原数据的转置(120, 120)
        s.append(abs(fft(signal.T)))                            # 数组中第四个，原数据的转置的傅里叶变换(120, 120)
        
        for ii in range(len(signal_backup_1)):
            signal_backup_1[ii] = np.roll(signal_backup_1[ii], 1)/2 + np.roll(signal_backup_1.T[ii], 1)/2         
        s.append(signal_backup_1)                                 # 数组中第五个，原数据依次右移1位/2+原数据依次上移2幂次位/2 (120, 120)
        s.append(abs(fft(signal_backup_1)))                       # 数组中第六个，数组中第五个数据的傅里叶变换(120, 120)
        
        for ii in range(len(signal_backup_2)):
            signal_backup_2[ii] = np.roll(signal_backup_2[ii], -1)/2 + np.roll(signal_backup_2.T[ii], 1)/2
        s.append(signal_backup_2)                                 # 数组中第七个，原数据依次左移1位/2+原数据依次上移2幂次位/2 (120, 120) 
        s.append(abs(fft(signal_backup_2)))                       # 数组中第八个，数组中第八个数据的傅里叶变换(120, 120)

        cA1, (cH1, cV1, cD1) = dwt2(signal, 'haar')               # 单级小波分解，返回分别为低频分量，水平高频，竖直高频，对角高频
        cA2, (cH2, cV2, cD2) = dwt2(signal.T, 'haar')
        cA3, (cH3, cV3, cD3) = dwt2(signal_backup_1, 'haar')
        cA4, (cH4, cV4, cD4) = dwt2(signal_backup_2, 'haar')

        s.append(np.concatenate([np.concatenate([cA1, cA2], axis=1), 
                np.concatenate([cA3, cA4], axis=1)], axis=0))           # 数组中第九个，小波变换，原数据(120, 120)
        s.append(np.concatenate([np.concatenate([cH1, cH2], axis=1), 
                np.concatenate([cH3, cH4], axis=1)], axis=0))           # 数组中第十个，小波变换，原数据(120, 120)
        s.append(np.concatenate([np.concatenate([cV1, cV2], axis=1), 
                np.concatenate([cV3, cV4], axis=1)], axis=0))           # 数组中第十一个，小波变换，原数据(120, 120)
        s.append(np.concatenate([np.concatenate([cD1, cD2], axis=1), 
                np.concatenate([cD3, cD4], axis=1)], axis=0))           # 数组中第十二个，小波变换，原数据(120, 120)

        for ii in range(len(s)):
            s[ii] = (s[ii]-np.min(s[ii]))/(np.max(s[ii])-np.min(s[ii])) * 255.0
        s = np.array(s)

        channel1 = np.concatenate([np.concatenate([s[0], s[6]], axis=1),    # 合并时域，1 3 5 7通道
                np.concatenate([s[4], s[2]], axis=1)], axis=0)              
        channel2 = np.concatenate([np.concatenate([s[1], s[7]], axis=1),    # 合并频域，2 4 6 8通道
                np.concatenate([s[5], s[3]], axis=1)], axis=0)
        channel3 = np.concatenate([np.concatenate([s[8], s[9]], axis=1),    # 合并小波域，9 10 11 12通道
                np.concatenate([s[10], s[11]], axis=1)], axis=0)

        final_img = np.array([channel1, channel2, channel3])
        return final_img