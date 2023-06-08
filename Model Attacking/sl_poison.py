import torchvision.models as models
import torchvision
import time
import csv
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

import os
import argparse

import torch.nn as nn
import torch.nn.functional as F

from modelutl import DLModel

from device_vgg import get_device_vgg
from edge_vgg import get_edge_vgg
from vgg import VGG

import copy
import matplotlib.pyplot as plt

import random
import numpy as np

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# def byz(v, net, lr, f):
#     #f攻击者个数
#     # v: 64 * 6 * 12 * 12
#     vi_shape = v[0].shape   # 6 * 12 * 12

#     v_tran = nd.concat(*v, dim=1)#元素都被分成一个一个的独立元素，然后做拼接
    
#     maximum_dim = nd.max(v_tran, axis=1).reshape(vi_shape)
    
#     minimum_dim = nd.min(v_tran, axis=1).reshape(vi_shape)
#     direction = nd.sign(nd.sum(nd.concat(*v, dim=1), axis=-1, keepdims=True))
#     directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim
#     # let the malicious clients (first f clients) perform the attack
#     for i in range(f):
#         random_12 = 1. + nd.random.uniform(shape=vi_shape)
#         v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
#     return v   

# def byz_pt(v):
#     # v: 64 * 6 * 12 * 12
#     random_12 = 10
#     vi_shape = v[0].shape   # 6 * 12 * 12
#     v_tran = rearrange(v, 'b h w c -> b (h w c)')   # 64 * (6*12*12)
#     maximum_dim, _ = torch.max(v_tran, 0)              # (6*12*12)
#     minimum_dim, _ = torch.min(v_tran, 0)              # (6*12*12)
#     direction = torch.sign(torch.sum(v_tran, 0))        # (6*12*12)
#     directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim    # (6*12*12)
#     v_s = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    
#     v_s = rearrange(v_s, "(h w c) -> 1 (h) (w) (c)", h=vi_shape[0], w=vi_shape[1], c=vi_shape[2])
#     v = torch.cat([v_s for _ in range(v.shape[0])])
#     return v
def drawLoss(loss_log):
    x2 = range(0,200)
    y2 = loss_log                
    plt.plot(x2, y2)
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.show()
    plt.savefig("loss_sl3.3.jpg")

def drawAcc(loss_log):
    x1 = range(0,200)
    y1 = loss_log                
    plt.plot(x1, y1)
    plt.xlabel('epoch')
    plt.ylabel('test acc')
    plt.show()
    plt.savefig("accuracy_sl3.3.jpg")
from mpl_toolkits.axes_grid1 import host_subplot
def draw2(train_loss,test_accuracy,trainlist):
    
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
    par1 = host.twinx()
    # set labels
    host.set_xlabel("iterations")
    host.set_ylabel("log loss")

    p1, = host.plot(trainlist, train_loss, label="training log loss")
    p2, = par1.plot(trainlist, test_accuracy, label="validation accuracy")
    
    # set location of the legend, 
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)
    
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([-1500, 160000])
    par1.set_ylim([0., 1.05])
    
    plt.draw()
    plt.show()
    plt.savefig("accuracy_loss_sl3.jpg")

def fix_random_seed():
    SEED = 1024
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

fix_random_seed()


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def valid(device_model, edge_model, test_loader):
    
    device_model.eval()
    edge_model.eval()
    
    with torch.no_grad():
        correct, total = 0, 0
        for sample, target in test_loader:
            
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long() #数据移至GPU
            
            output = device_model.forward(sample)
            output = edge_model.forward(output)
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
            
    acc_test = float(correct) * 100 / total
    return acc_test


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--model_name', default='VGG19', type=str, help='model name')
parser.add_argument('--gpu', default='cuda:0', type=str, help='GPU')
parser.add_argument('--epoch_num', default=200, type=int, help='Epoch Number')
parser.add_argument('--log_file_name', default='single_VGG19', type=str, help='single_VGG19')
parser.add_argument('--split_layer', default=11, type=int, help='split layer')

args = parser.parse_args()


DEVICE = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
print('Use Device:', DEVICE)

CPU_DEVICE = torch.device('cpu')

if args.gpu != 'cpu':
    torch.cuda.empty_cache()


device_num = 6

device_lr = 0.001
edge_lr = 0.001

epoch_num = args.epoch_num


# ==================================
# 生成模型

model_names = ['VGG11', 'VGG13', 'VGG16', 'VGG19']

model_name = args.model_name
split_layer = args.split_layer

device_list = []
edge_list = []
generator0 = get_device_vgg(model_name, split_layer).to(CPU_DEVICE)
optimizer_G = optim.SGD(generator0.parameters(), lr = device_lr)
generator0.load_state_dict(torch.load('right100vgg19_GAN_diffloss3.pt'))
generator = copy.deepcopy(DLModel(generator0, optimizer_G))

# generator1 = get_device_vgg(model_name, split_layer).to(CPU_DEVICE)
# optimizer_G = optim.SGD(generator1.parameters(), lr = device_lr)
# generator1.load_state_dict(torch.load('100vgg19_GAN_diffloss3.pt'))
# generator2 = copy.deepcopy(DLModel(generator1, optimizer_G))

# 生成client模型
for i in range(6):

    fix_random_seed()

    device_model = get_device_vgg(model_name, split_layer).to(CPU_DEVICE)
    device_optimizer = optim.SGD(device_model.parameters(), lr = device_lr)
    
    temp_device = copy.deepcopy(DLModel(device_model, device_optimizer))
    device_list.append(temp_device)

# 生成server模型


edge_model =  get_edge_vgg(model_name, split_layer).to(CPU_DEVICE)
edge_optimizer = optim.SGD(edge_model.parameters(), lr = edge_lr)
    
edge = DLModel(edge_model, edge_optimizer)
# 模型生成结束
# =================================   



# ===================================
# 生成数据集
num_traindata = 50000 // 6

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

from torch.utils.data import Subset

indices = list(range(50000))

train_loaders = []

trainset = torchvision.datasets.CIFAR10 (root='./cifar10', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10 (root='./cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)


for i in range(6):
    
    # 切分数据集
    part_tr = indices[num_traindata * i : num_traindata * (i + 1)]
    
    trainset_sub = Subset(trainset, part_tr)
    train_loader = torch.utils.data.DataLoader(trainset_sub, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    
    train_loaders.append(train_loader)
    
# 数据集生成结束
# =================================    



criterion = nn.CrossEntropyLoss()

start_time = time.time()
def main():
    loss_log2= []
    test_acc_log = []
    trainlist=[]
    for epoch in range(epoch_num):
        trainlist.append(str(epoch))
        loss_all=0
        loss_log = []
        test_acc_log = []
        train_acc_log = []

        for device_index in range(len(device_list)):

            device = device_list[device_index]
            
            train_loader = train_loaders[device_index]

            if args.gpu != 'cpu':
                torch.cuda.empty_cache()

            device.model.to(DEVICE)
            edge.model.to(DEVICE)

            device.train()
            edge.train()

            correct, total_loss = 0, 0
            total = 0
        
            for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Epoch '+str(epoch+1)+', Device '+str(device_index))):
                
                device.zero_grad()
                edge.zero_grad()
                
                sample, target = data
                
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long() #数据移至GPU
                
                
                if(device_index<5):
                    inter_matrix = device.forward(sample)
                    
                else:
                    #
                    generator.model.to(DEVICE)

                    inter_matrix=generator.forward(sample)


                output = edge.forward(inter_matrix)
                loss = criterion(output, target)
                
                loss.backward()
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()

                edge.backward()
                device.backward()
            
            device.model.to(CPU_DEVICE)
            edge.model.to(CPU_DEVICE)

            acc_train = float(correct) * 100.0 / total
            print('Train Acc:', acc_train, ', Loss', total_loss/total)

            loss_log.append(str(total_loss/total))
            train_acc_log.append(str(acc_train))
            loss_all+=total_loss/total
        
        loss_log2.append(loss_all/6)

        
        # =======================================
        # 聚合client侧和server侧的模型

        # 聚合client侧
        weights_list = []
        weights_list2 = []
        #device_num=6

        for device_index in range(device_num):
            
            device = device_list[device_index]
            device.eval()
            # if(device_index==5):
            #     print("poison")
            #     print(device.model.state_dict())
            # elif(device_index==1):
            #     print("normal")
            #     print(device.model.state_dict())
            weights_list.append(device.model.state_dict())
        
        device_weights = average_weights(weights_list)
         
        for device_index in range(device_num):
            device = device_list[device_index]
            device.model.load_state_dict(device_weights)
#只聚合正常的5个
        # for device_index in range(4):
            
        #     device = device_list[device_index]
        #     device.eval()
            
        #     weights_list2.append(device.model.state_dict())
        
        # device_list[0].model.load_state_dict(average_weights(weights_list2))
        # device = device_list[0]
        # device.model.to(DEVICE)
        # edge.model.to(DEVICE)
        # acc_test = valid(device, edge, test_loader)
        # print("5normalclient testacc")
        # print(device_weights)
        # print("after")
        # print(average_weights(weights_list2))

        # device_list[0].model.load_state_dict(device_weights)
        # 聚合server
        
        # 聚合结束
        # =======================================


        # =======================================
        # Test for Every Epoch

        print('Test for Every Epoch')

        device = device_list[0]

        if args.gpu != 'cpu':
            torch.cuda.empty_cache()
        device.model.to(DEVICE)
        edge.model.to(DEVICE)
        acc_test = valid(device, edge, test_loader)
        device.model.to(CPU_DEVICE)
        edge.model.to(CPU_DEVICE)
        test_acc_log.append(acc_test)
        print('Test Acc:', acc_test,', Train Acc:', acc_train, ', Loss', total_loss/total)

        # Test End
        # =======================================

        # 写入log
        with open(args.log_file_name, 'a', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            write_list = [model_name, str(split_layer), str(epoch+1), str(acc_test)] + loss_log + train_acc_log
            csv_writer.writerow(write_list)
    drawLoss(loss_log2)
    drawAcc(test_acc_log)
    draw2(test_acc_log,loss_log2,trainlist)

if __name__ == '__main__':
    main()