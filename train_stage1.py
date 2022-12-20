import os
import tqdm
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50

from dataloader import ComNpyFolder
from argument import ArgumentParser
from data_processing import data_convert
from models.model import ModelStage1, ModelStage2
from models.utils import ComparedLoss
from utils import train_transform, test_transform
    
def train(args):
    # log and save model dir
    if (Path(args.log_path).is_dir()==False):
        os.mkdir(args.log_path)
    if (os.path.isfile(args.log_path+"stage1_log.txt")):
        os.remove(args.log_path+"stage1_log.txt")
    if (Path(args.save_model_path).is_dir()==False):
        os.mkdir(args.save_model_path)    

    # device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    train_dataset = ComNpyFolder(root=args.train_dataset, transform=train_transform)
    train_iters = DataLoader(train_dataset, batch_size=args.s1_batch_size, shuffle=True)
    test_dataset = ComNpyFolder(root=args.test_dataset, transform=test_transform)
    test_iters = DataLoader(test_dataset, batch_size=args.s1_batch_size, shuffle=True)

    # model
    net = ModelStage1(args).to(device)
    print(net)

    # iterator
    optimizer = optim.Adam(net.parameters(), lr=args.s1_lr)
    criteon = ComparedLoss().to(device)
    postfix = {'epoch': 0, 'loss': 0.0, 'learn_rate': 0.0}
    
    for epoch in range(args.s1_epochs):
        pbar = tqdm.tqdm(train_iters, desc='train stage1', ncols=100, postfix=postfix)
        for batch_id, (xL, xR, y) in enumerate(pbar):
            xL, xR, y = xL.to(device), xR.to(device), y.to(device)
            
            _, pre_L = net(xL)
            _, pre_R = net(xR)
            
            loss = criteon(pre_L, pre_R, args.s1_batch_size, temperature=args.s1_temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(epoch=epoch, loss=float(loss), learn_rate=args.s1_lr)

        with open(args.log_path+"stage1_log.txt", "a") as f:
            f.write('stage1: epoch:[{}/{}], loss:{:.4f} \n'.format(epoch, args.s1_epochs, loss))
            
        # save model
        torch.save(net.state_dict(), os.path.join(args.save_model_path, 'model_stage1_epoch' + str(epoch) + '_' + str(float(loss)) + '.pth'))

if __name__ == "__main__":
    args = ArgumentParser()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        # test(args)
        print("no test options!")
    elif args.mode == "data":
        data_convert("./dataset/csv/train/", "./dataset/npy/train/")
        data_convert("./dataset/csv/test/", "./dataset/npy/test/")
