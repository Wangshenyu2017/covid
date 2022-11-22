# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:22:56 2022

@author: LENOVO
"""
import os
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchxrayvision as xrv
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from visdom import Visdom
import time

import os
import numpy as np 
import pandas as pd 
import seaborn as sns
from PIL import Image 
from PIL import ImageEnhance
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2

import os
import shutil
import time

import pandas as pd

import csv

from dataload import CovidCTDataset

device = torch.device("cpu") # 使用cpu训练，或者GPU
# device = torch.device("GPU") # 使用cpu训练，或者GPU


def train(optimizer, epoch, model, train_loader, modelname, criteria):
    model.train()  # 训练模式
    bs = 10
    train_loss = 0
    train_correct = 0

    for batch_index, batch_samples in enumerate(train_loader):

        # move data to device
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
        # data形状，torch.Size([32, 3, 224, 224])
        # data = data[:, 0, :, :]  # 原作者只取了第一个通道的数据来训练，笔者改成了3个通道

        # data = data[:, None, :, :]
        # data形状，torch.Size([32, 1, 224, 224])

        optimizer.zero_grad()

        output = model(data)
        loss = criteria(output, target.long())
        train_loss += criteria(output, target.long())  # 后面求平均误差用的

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()  # 累加预测与标签吻合的次数，用于后面算准确率

        # 显示一个epoch的进度，425张图片，批大小是32，一个epoch需要14次迭代
        if batch_index % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item() / bs))
    # print(len(train_loader.dataset))   # 425
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))

    if os.path.exists('performance') == 0:
        os.makedirs('performance')
    f = open('performance/{}.txt'.format(modelname), 'a+')
    f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f.write('\n')
    f.close()

    return train_loss / len(train_loader.dataset)  # 返回一个epoch的平均误差，用于可视化损失


'''
model:训练好的模型

val_loader: 验证集

criteria: 

'''
def val(model, val_loader, criteria):
    model.eval()
    val_loss = 0
    
    # Don't update model
    with torch.no_grad():
    
        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            # data = data[:, 0, :, :]  # 原作者只取了第一个通道的数据，笔者改成了3个通道
    
            # data = data[:, None, :, :]
            # data形状，torch.Size([32, 1, 224, 224])
            output = model(data)
    
            val_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
    
            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)
    
    return targetlist, scorelist, predlist, val_loss / len(val_loader.dataset)





normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 依通道标准化

train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


if __name__ == '__main__':

    batchsize = 10  
    total_epoch = 1500  
    votenum = 10

    # ------------------------------- step 1/5 数据读取 ----------------------------

    # 实例化CovidCTDataset
    trainset = CovidCTDataset(root_dir='C:/Users/LENOVO/python_files/covid_ct2/data',
                            txt_COVID='C:/Users/LENOVO/python_files/covid_ct2/data/trainCT_COVID.txt',
                            txt_NonCOVID='C:/Users/LENOVO/python_files/covid_ct2/data/trainCT_NonCOVID.txt',
                            transform=train_transformer)
    valset = CovidCTDataset(root_dir='C:/Users/LENOVO/python_files/covid_ct2/data',
                            txt_COVID='C:/Users/LENOVO/python_files/covid_ct2/data/valCT_COVID.txt',
                            txt_NonCOVID='C:/Users/LENOVO/python_files/covid_ct2/data/valCT_NonCOVID.txt',
                            transform=val_transformer)
    print(trainset.__len__())
    print(valset.__len__())

    # 构建DataLoader
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)


    # ------------------------------ step 2/5 模型 --------------------------------

    model = xrv.models.DenseNet(num_classes=2, in_channels=3).cpu()  # 
    # model = xrv.models.DenseNet(num_classes=2, in_channels=3).cuda() 
    #DenseNet 模型，二分类
    modelname = 'DenseNet_medical'
    torch.cuda.empty_cache()

    # ----------------------------- step 3/5 损失函数 ----------------------------

    criteria = nn.CrossEntropyLoss()  # 二分类用交叉熵损失

    # ----------------------------- step 4/5 优化器 -----------------------------

    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)  # 动态调整学习率策略，初始学习率0.0001

    # ----------------------------  step 5/5 训练 ------------------------------
    
    # visiom可视化训练过程
    
    viz = Visdom(server='http://localhost/', port=8097)

    viz.line([[0., 0., 0., 0., 0.]], [0], win='train_performance', update='replace', opts=dict(title='train_performance', legend=['precision', 'recall', 'AUC', 'F1', 'acc']))
    viz.line([[0., 0.]], [0], win='train_Loss', update='replace', opts=dict(title='train_Loss', legend=['train_loss', 'val_loss']))

    warnings.filterwarnings('ignore')

    # 模型评价
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    r_list = []
    p_list = []
    acc_list = []
    AUC_list = []

    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())

    

    for epoch in range(1, total_epoch + 1):

        train_loss = train(optimizer, epoch, model, train_loader, modelname, criteria)  # 进行一个epoch训练的函数

        targetlist, scorelist, predlist, val_loss = val(model, val_loader, criteria)  # 用验证集验证
        print('target', targetlist)
        print('score', scorelist)
        print('predict', predlist)
        vote_pred = vote_pred + predlist
        vote_score = vote_score + scorelist
        if epoch % votenum == 0:  # 每10个epoch，计算一次准确率和召回率等

            # major vote
            vote_pred[vote_pred <= (votenum / 2)] = 0
            vote_pred[vote_pred > (votenum / 2)] = 1
            vote_score = vote_score / votenum

            print('vote_pred', vote_pred)
            print('targetlist', targetlist)


            TP = ((vote_pred == 1) & (targetlist == 1)).sum()
            TN = ((vote_pred == 0) & (targetlist == 0)).sum()
            FN = ((vote_pred == 0) & (targetlist == 1)).sum()
            FP = ((vote_pred == 1) & (targetlist == 0)).sum()

            print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
            print('TP+FP', TP + FP)
            p = TP / (TP + FP)
            print('precision', p)
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            print('recall', r)
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('F1', F1)
            print('acc', acc)
            AUC = roc_auc_score(targetlist, vote_score)
            print('AUCp', roc_auc_score(targetlist, vote_pred))
            print('AUC', AUC)

            # 训练过程可视化
            train_loss = train_loss.cpu().detach().numpy()
            val_loss = val_loss.cpu().detach().numpy()
            '''viz.line([[p, r, AUC, F1, acc]], [epoch], win='train_performance', update='append',
                    opts=dict(title='train_performance', legend=['precision', 'recall', 'AUC', 'F1', 'acc']))
            viz.line([[train_loss], [val_loss]], [epoch], win='train_Loss', update='append',
                    opts=dict(title='train_Loss', legend=['train_loss', 'val_loss']))
'''
            print(
                '\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, '
                'average accuracy: {:.4f}, average AUC: {:.4f}'.format(
                    epoch, r, p, F1, acc, AUC))

            # 更新模型

            if os.path.exists('backup') == 0:
                os.makedirs('backup')
            torch.save(model.state_dict(), "backup/{}.pt".format(modelname))

            vote_pred = np.zeros(valset.__len__())
            vote_score = np.zeros(valset.__len__())
            f = open('performance/{}.txt'.format(modelname), 'a+')
            f.write(
                '\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, '
                'average accuracy: {:.4f}, average AUC: {:.4f}'.format(
                    epoch, r, p, F1, acc, AUC))
            f.close()
        if epoch % (votenum*10) == 0:  # 每100个epoch，保存一次模型
            torch.save(model.state_dict(), "backup/{}_epoch{}.pt".format(modelname, epoch))



    def test(model, test_loader):
        model.eval()

        # Don't update model
        with torch.no_grad():

            predlist = []
            scorelist = []
            targetlist = []
            # Predict
            for batch_index, batch_samples in enumerate(test_loader):
                data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
                # data = data[:, 0, :, :]            #  只取了第一个通道的数据来训练，笔者改成了灰度图像

                # data = 0.299 * data[:, 0, :, :] + 0.587 * data[:, 1, :, :] + 0.114 * data[:, 2, :, :]
                # data形状，torch.Size([32, 224, 224])

                # data = data[:, None, :, :]
                # data形状，torch.Size([32, 1, 224, 224])
                #             print(target)
                output = model(data)
                score = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)

                targetcpu = target.long().cpu().numpy()
                predlist = np.append(predlist, pred.cpu().numpy())
                scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
                targetlist = np.append(targetlist, targetcpu)

        return targetlist, scorelist, predlist



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 依通道标准化

test_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

if __name__ == '__main__':

    batchsize = 10  # 原来用的10，这里改成32，根据个人GPU容量来定。

    # ------------------------------- step 1/3 数据 ----------------------------

    # 实例化CovidCTDataset
    testset = CovidCTDataset(root_dir='C:/Users/LENOVO/python_files/covid_ct2/data',
                            txt_COVID='C:/Users/LENOVO/python_files/covid_ct2/data/testCT_COVID.txt',
                            txt_NonCOVID='C:/Users/LENOVO/python_files/covid_ct2/data/testCT_NonCOVID.txt',
                            transform=test_transformer)
    print(testset.__len__())

    # 构建DataLoader

    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

    # ------------------------------ step 2/3 模型 --------------------------------

    model = xrv.models.DenseNet(num_classes=2, in_channels=3).cpu()  # 
    # model = xrv.models.DenseNet(num_classes=2, in_channels=3).cuda() 
#    DenseNet 模型，二分类
    modelname = 'DenseNet_medical'
    torch.cuda.empty_cache()

    # ----------------------------  step 3/3 测试 ------------------------------

    f = open(f'performance/test_model.csv', mode='w')
    csv_writer = csv.writer(f)
    flag = 1

    for modelname in os.listdir('backup'):
        model.load_state_dict(torch.load('backup/{}'.format(modelname)))
        torch.cuda.empty_cache()

        bs = 10

        warnings.filterwarnings('ignore')

        r_list = []
        p_list = []
        acc_list = []
        AUC_list = []
        TP = 0
        TN = 0
        FN = 0
        FP = 0

        vote_score = np.zeros(testset.__len__())

        targetlist, scorelist, predlist = test(model, test_loader)
        vote_score = vote_score + scorelist

        TP = ((predlist == 1) & (targetlist == 1)).sum()
        TN = ((predlist == 0) & (targetlist == 0)).sum()
        FN = ((predlist == 0) & (targetlist == 1)).sum()
        FP = ((predlist == 1) & (targetlist == 0)).sum()

        p = TP / (TP + FP)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)

        AUC = roc_auc_score(targetlist, vote_score)

        print(
            '\n{}, recall: {:.4f}, precision: {:.4f},F1: {:.4f}, accuracy: {:.4f}, AUC: {:.4f}'.format(
                modelname, r, p, F1, acc, AUC))
        if flag:
            header = ['modelname', 'recall', 'precision', 'F1', 'accuracy', 'AUC']
            csv_writer.writerow(header)
            flag = 0
        row = [modelname, str(r), str(p), str(F1), str(acc), str(AUC)]
        csv_writer.writerow(row)

    f.close()


