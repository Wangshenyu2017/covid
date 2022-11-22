# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:29:08 2022

@author: LENOVO
"""


import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
 
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
 
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
 
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
def train_runner(model, device, train_loader, optimizer, epoch, model_path):
    model.train()
    total = 0
    correct = 0.0
    total_loss = 0.0
    for i ,data in enumerate(train_loader, 0):
       inputs, labels = data
       inputs, labels = inputs.to(device), labels.to(device)
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = F.cross_entropy(outputs, labels)
       total_loss += loss.item()
       predict = outputs.argmax(dim=1)
       total += labels.size(0)
       correct += (predict == labels).sum().item()
       loss.backward()
       optimizer.step()
       if i%1000 == 0:
           print("Train Epoch{} \t Loss: {:.6f}, accuracy: {:.6f}%".format(epoch, (total_loss/total), 100*(correct/total)))
           torch.save(model, model_path + "/checkpoint" + '.pth.tar')
    return total_loss/total, correct/total
 
def test_runner(model, device, test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            current_loss = F.cross_entropy(output, targets).item()
            test_loss += current_loss
            predict = output.argmax(dim=1)
            total += targets.size(0)
            correct += (predict == targets).sum().item()
        print("test_loss: {:.6f}, accuracy: {:.6f}%".format(test_loss/total, 100*(correct/total)))
    return test_loss/total, correct/total
 
def draw_curve(train_curve, test_curve, title, y_label, train_iterations, test_iterations):
    plt.plot(range(train_iterations), train_curve, label='train', color="#3D9140")
    plt.plot(range(test_iterations), test_curve, label='test', color="#FF8000")
    plt.xlabel('epoch times')
    plt.legend(loc="upper right")
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig('./Resnet_0.00001/'+title+time.strftime('%Y_%m_%d_%H%M%S')+'.jpg')
    plt.show()
 
def main():
    data_path = "./data"
    model_path = "./model"
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(300),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    image_sets = datasets.ImageFolder(os.path.join(data_path), transform=transform)
    train_length = int(0.8 * len(image_sets))
    test_length = len(image_sets) - train_length
    train_set, test_set = torch.utils.data.random_split(image_sets, [train_length, test_length])
    train_set = train_set.dataset
    test_set = test_set.dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    epoch = 50
    train_cost_curve = []
    train_accuracy_curve = []
    test_cost_curve = []
    test_accuracy_curve = []
    print("start_time:" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(1, epoch+1):
        train_cost_curve_current, train_accuracy_curve_current = train_runner(model, device, train_loader, optimizer,
                                                                              epoch, model_path)
        test_cost_curve_current, test_accuracy_curve_current = test_runner(model, device, test_loader)
        train_accuracy_curve.append(train_accuracy_curve_current)
        train_cost_curve.append(train_cost_curve_current)
        test_accuracy_curve.append(test_accuracy_curve_current)
        test_cost_curve.append(test_cost_curve_current)
    print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\n')
    draw_curve(train_accuracy_curve, test_accuracy_curve, 'model accuracy curve', 'cost', epoch, epoch)
    draw_curve(train_cost_curve, test_cost_curve, 'model cost curve', 'cost', epoch, epoch)
 
if __name__ == '__main__':
    main()
