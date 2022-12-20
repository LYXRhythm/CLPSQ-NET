import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from dataloader import NpyFolder
from argument import ArgumentParser
from data_processing import data_convert
from models.model import ModelStage2
from utils import test_transform

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=15, va='center', ha='center')
    
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.imshow(cm, interpolation='nearest')
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes)
    plt.yticks(xlocations, classes, rotation=90)
    plt.ylabel('Actual label', fontsize=20)
    plt.xlabel('Predict label', fontsize=20)
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

def Metrics(true_label, pred_label):
    print("accuracy_score: ", accuracy_score(true_label, pred_label))
    print("precision_score: ", precision_score(true_label, pred_label, average=None))  
    print("recall_score: ", recall_score(true_label, pred_label, average=None))
    print("f1_score: ", f1_score(true_label, pred_label, average=None))
    cm = confusion_matrix(true_label, pred_label)
    plot_confusion_matrix(cm, './result/confusion_matrix.png', title='confusion matrix')

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    test_dataset = NpyFolder(root=args.test_dataset, transform=test_transform)
    test_iters = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=True)

    # model
    net = ModelStage2(args).to(device)
    net.load_state_dict(torch.load(args.eval_model_path, map_location=device), strict=True)

    with torch.no_grad():
        # test_correct = 0
        # test_total = 0

        true_label = torch.tensor([]).to(device)
        pred_label = torch.tensor([]).to(device)

        for x, y in test_iters:
            x = x.to(device)
            y = y.to(device)
            logits = net(x)
            pred = logits.argmax(dim=1)

            true_label = torch.cat((true_label, y))
            pred_label = torch.cat((pred_label, pred))

        true_label = true_label.to("cpu")
        pred_label = pred_label.to("cpu")
        Metrics(true_label, pred_label)

if __name__ == "__main__":
    args = ArgumentParser()
    evaluate(args)