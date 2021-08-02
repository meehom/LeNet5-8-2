# -*- coding: utf-8 -*-
"""
 @Time    : 2021/8/1 22:31
 @Author  : meehom
 @Email   : meehomliao@163.com
 @File    : main.py
 @Software: PyCharm
"""
import torch
from torch import nn,optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from lenet5 import Lenet5

def main():
    batch_size = 32
    cifar_train = datasets.CIFAR10('cifar', True, transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download = True)
    print("running ----")
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)
    x, label = iter(cifar_train).next()
    print("x:", x.shape, "label:",label)
    model = Lenet5()
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        for batchidx, (x, label) in enumerate(cifar_train):
            logits = model(x)
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # loss print
        print(epoch,loss.item())

        # test
        total_correct = 0
        toral_num = 0
        for x, label in cifar_test:
            logits = model(x)
            # print("logits:",logits)
            pre = logits.argmax(dim=1)
            total_correct += torch.eq(pre, label).float().sum()
            toral_num += x.size(0)

        acc = total_correct / toral_num
        print(epoch, acc)


if __name__ == '__main__':
    main()
