import os
import tqdm
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50

from dataloader import NpyFolder
from argument import ArgumentParser
from data_processing import data_convert
from models.model import ModelStage1, ModelStage2
from models.utils import ComparedLoss
from utils import train_transform, test_transform
    
def train(args):
    # log and save model dir
    if (Path(args.log_path).is_dir()==False):
        os.mkdir(args.log_path)
    if (os.path.isfile(args.log_path+"stage2_log.txt")):
        os.remove(args.log_path+"stage2_log.txt")
    if (Path(args.save_model_path).is_dir()==False):
        os.mkdir(args.save_model_path)    

    # device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    train_dataset = NpyFolder(root=args.train_dataset, transform=train_transform)
    train_iters = DataLoader(train_dataset, batch_size=args.s2_batch_size, shuffle=True)
    test_dataset = NpyFolder(root=args.test_dataset, transform=test_transform)
    test_iters = DataLoader(test_dataset, batch_size=args.s2_batch_size, shuffle=True)

    # model reload_path
    net = ModelStage2(args).to(device)
    net.load_state_dict(torch.load(args.reload_path, map_location=device), strict=False)

    # iterator
    optimizer = optim.Adam(net.parameters(), lr=args.s2_lr)
    criteon = torch.nn.CrossEntropyLoss()
    postfix = {'epoch': 0, 'loss': 0.0, 'learn_rate': 0.0}
    
    for epoch in range(args.s2_epochs):
        pbar = tqdm.tqdm(train_iters, desc='train stage2', ncols=100, postfix=postfix)
        for batch_id, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            pre = net(x)
            
            loss = criteon(pre, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(epoch=epoch, loss=float(loss), learn_rate=args.s2_lr)

        with torch.no_grad():
            train_correct = 0
            train_total = 0
            pbar_train = tqdm.tqdm(train_iters, desc='train set accuracy', ncols=100)
            for batch_id, (x, y) in enumerate(pbar_train):
                x = x.to(device)
                y = y.to(device)

                logits = net(x)
                pred = logits.argmax(dim=1)
                train_total += len(y)
                train_correct += pred.eq(y).sum().float().item()
        print('epoch:[{}/{}], loss:{:.4f}, train accuracy:{:.4f}'.format(epoch, args.s2_epochs, loss, train_correct / train_total))
        
        with torch.no_grad():
            test_correct = 0
            test_total = 0

            pbar_test = tqdm.tqdm(test_iters, desc='test set accuracy', ncols=100)
            for batch_id, (x, y) in enumerate(pbar_test):
                x = x.to(device)
                y = y.to(device)

                logits = net(x)
                pred = logits.argmax(dim=1)
                test_total += len(y)
                test_correct += pred.eq(y).sum().float().item()

        print('test accuracy:{:.4f}'.format(test_correct / test_total))

        with open(args.log_path+"stage2_log.txt", "a") as f:
            f.write('stage2: epoch:[{}/{}], loss:{:.4f}, train accuracy:{:.4f}\n'.format(epoch, args.s2_epochs, loss, train_correct / train_total))
            f.write('test accuracy:{:.4f}\n'.format(test_correct / test_total))

        # save model
        torch.save(net.state_dict(), os.path.join(args.save_model_path, 'model_stage2_epoch' + str(epoch) + '_' + str(test_correct / test_total) + '.pth'))

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    test_dataset = NpyFolder(root=args.test_dataset, transform=test_transform)
    test_iters = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # model
    net = ModelStage2(args).to(device)

    net.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0

        for x, y in test_iters:
            x = x.to(device)
            y = y.to(device)

            logits = net(x)
            pred = logits.argmax(dim=1)
            test_total += len(y)
            test_correct += pred.eq(y).sum().float().item()
        print('test accuracy:{:.4f}'.format(test_correct / test_total))

if __name__ == "__main__":
    args = ArgumentParser()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "data":
        data_convert("./dataset/csv/train/", "./dataset/npy/train/")
        data_convert("./dataset/csv/test/", "./dataset/npy/test/")
