import torchvision.models as models
import torchvision
import time
import csv
import torch
from einops import rearrange
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt
from einops import rearrange
import os
import argparse

import torch.nn as nn
import torch.nn.functional as F

from modelutl import DLModel

from device_vgg import get_device_vgg
from edge_vgg import get_edge_vgg
from vgg import VGG

import copy

from collections import Counter

import random
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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

def Diffloss(gen_output, output, eps=1e-15):
    import numpy as np
    a=gen_output-output
    
    diff = torch.norm(a, p=2, dim=1, keepdim=True)#1是行 [64]

    loss = 1/(diff.sum())
    # lossfinal=torch.tensor(loss).requires_grad_()
    
    return loss/64



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

cuda = True if torch.cuda.is_available() else False

img_shape=(64,256,8,8)

# Initialize generator and discriminator
# generator = Generator()


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--model_name', default='VGG19', type=str, help='model name')
parser.add_argument('--gpu', default='cuda:0', type=str, help='GPU')
parser.add_argument('--epoch_num', default=1500, type=int, help='Epoch Number')
parser.add_argument('--log_file_name', default='single_VGG19', type=str, help='single_VGG19')
parser.add_argument('--split_layer', default=11, type=int, help='split layer')

args = parser.parse_args()


# optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

DEVICE = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
print('Use Device:', DEVICE)

CPU_DEVICE = torch.device('cpu')

if args.gpu != 'cpu':
    torch.cuda.empty_cache()


device_num = 6

device_lr = 0.001#原来是0.001
edge_lr = 0.001

epoch_num = args.epoch_num


# ==================================
# 生成模型

model_names = ['VGG11', 'VGG13', 'VGG16', 'VGG19']

model_name = args.model_name
split_layer = args.split_layer

device_list = []


# 生成client模型
for i in range(6):

    fix_random_seed()

    device_model = get_device_vgg(model_name, split_layer).to(CPU_DEVICE)
    device_optimizer = optim.SGD(device_model.parameters(), lr = device_lr)
    
    temp_device = copy.deepcopy(DLModel(device_model, device_optimizer))
    device_list.append(temp_device)

# generator0 = get_device_vgg(model_name, split_layer).to(CPU_DEVICE)
# G_optimizer = optim.SGD(generator0.parameters(), lr = device_lr)    
# generator = copy.deepcopy(DLModel(generator0, G_optimizer))

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
class Generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)#归一化
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d*2, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d*2)#这里*2
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
        self.downsample = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.deconv5_bn = nn.BatchNorm2d(d*2)#这里*2

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        # print("input")
        # print(input.shape)
        x = F.relu(self.deconv1_bn(self.deconv1(input)))#[64, 1024, 4, 4]
        
        x = F.relu(self.deconv2_bn(self.deconv2(x)))#[64, 1024, 4, 4]
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))#[64, 256, 32, 32]
        
        # x=self.downsample(x)
        x = F.relu(self.deconv5_bn(self.downsample(x)))
        x = F.relu(self.deconv5_bn(self.downsample(x)))
        x = F.relu(self.deconv5_bn(self.downsample(x)))#[64, 256, 4, 4]
        # x = F.tanh(self.deconv5(x))#这里更改了如下
        #x = torch.tanh(self.deconv5(x))#[64, 3, 64, 64]
        x = torch.tanh(x)
        # print("xshape")
        # print(x.shape)

        return x
generator = Generator()
G_optimizer = optim.SGD(generator.parameters(), lr = edge_lr)

# GANmodel = DLModel(cmodel, optimizer_G)
# GANmodel.model.load_state_dict(torch.load('vgg19_GAN.pt').)
# pthfile='D:/download/研究生/科研/coding/sl_new/vgg19_GAN.pt'
# GANmodel.model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(pthfile).items()})

class discriminator(nn.Module):

    # 初始化网络模型
    def __init__(self):
        super(discriminator, self).__init__()
        self.flatten = nn.Flatten(1,-1)            # 展平层在线性层之前调整网络形状
        self.linear = nn.Linear(256*4*4, 10)     # 图片大小为28*28像素，因此输入为28*28；分类数为10，因此输出为10
        self.softmax = nn.Softmax(dim=1)       # 对线性层的输出取Softmax，转换为概率

    # 前向传播
    def forward(self, x):
        x = self.flatten(x)
        #print('flatten output：', x, 'shape：', x.shape)
        # print("xshape")
        # print(x.size())
        x = self.linear(x)
        #print('linear output：', x, 'shape：', x.shape)
        x = self.softmax(x)
        # print('softmax output：', x, 'shape：', x.shape)
        return x


# 生成server模型
edge_model =  get_edge_vgg(model_name, split_layer).to(CPU_DEVICE)
edge_optimizer = optim.SGD(edge_model.parameters(), lr = edge_lr)
    
edge = DLModel(edge_model, edge_optimizer)

# dis_model =  get_edge_vgg(model_name, split_layer).to(CPU_DEVICE)

# Discriminator = DLModel(dis_model, dis_optimizer)
Discriminator = discriminator()
D_optimizer = optim.SGD(Discriminator.parameters(), lr = edge_lr)
# 模型生成结束
# =================================   



# ===================================
# 生成数据集
num_traindata = 50000 // 6

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3,1,1)),#minist专用
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    # transforms.Resize((32,32)), #后面
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3,1,1)),#minist专用
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

from torch.utils.data import Subset

indices = list(range(50000))

train_loaders = []

trainset = torchvision.datasets.CIFAR10 (root='./cifar10', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10 (root='./cifar10', train=False, download=True, transform=transform_test)

# trainset = torchvision.datasets.MNIST (root='./minist', train=True, download=True, transform=transform_train)

# testset = torchvision.datasets.MNIST (root='./minist', train=False, download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)


for i in range(6):
    
    # 切分数据集
    part_tr = indices[num_traindata * i : num_traindata * (i + 1)]
    
    trainset_sub = Subset(trainset, part_tr)
    train_loader = torch.utils.data.DataLoader(trainset_sub, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    
    train_loaders.append(train_loader)
    
# 数据集生成结束
# =================================    
L1_loss = nn.L1Loss(reduction='mean')
BCE_loss = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

start_time = time.time()
def main():
    loss_log2= []
    test_acc_log = []
    trainlist=[]
   
    for epoch in range(epoch_num):
        trainlist.append(str(epoch))
        loss_log = []
        
        train_acc_log = []
        grad_list=[]
        loss_all=0.0
        acctest_all=0.0

        for device_index in range(6):

            device = device_list[device_index]
            
            train_loader = train_loaders[device_index]

            if args.gpu != 'cpu':
                torch.cuda.empty_cache()

            device.model.to(DEVICE)
            edge.model.to(DEVICE)
            generator.to(DEVICE)#这里去掉.model
            Discriminator.to(DEVICE) #这里去掉.model

            device.train()
            edge.train()
            Discriminator.train()
            generator.train()

            correct, total_loss,total_loss2 = 0, 0,0
            total = 0
            attackeracc=0
        
            
            
            if(device_index<1):
                for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Epoch '+str(epoch+1)+', Device '+str(device_index))):
                       
                    device.zero_grad()
                    # edge.zero_grad()
                    Discriminator.zero_grad()
                    generator.zero_grad()
                    
                    sample, target = data
                    
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long() #数据移至GPU
                    
                    z_ = torch.randn([64, 100, 1, 1])#[64, 3, 32, 32]
                    z_ = Variable(z_.cuda())
                    X_real = device.forward(sample)
                    G_result=generator.forward(z_)#生成GAN时用
                    
                    
                    # generator.train()
                    # Discriminator.eval()
                    Y_real = Discriminator.forward(X_real).data#毒化
                    Y_fake = Discriminator.forward(G_result)
                    GXminloss=criterion(X_real, G_result)
                    GYmaxloss=1/criterion(Y_fake, Y_real)
                    G_loss=GXminloss+GYmaxloss
                    G_loss.requires_grad_(True)
                    G_loss.backward(retain_graph=True)
                    # print("G_loss")
                    # print(G_loss)

                    # generator.eval()
                    # Discriminator.train()
                    
                    Dminloss=criterion(Y_real, target)
                    Dmaxloss=1/criterion(Y_fake, target)
                    D_loss=Dminloss+Dmaxloss
                    # print("D_loss")
                    # print(D_loss)
                    D_loss.requires_grad_(True)
                    
                    D_loss.backward()

                    G_optimizer.step()
                    D_optimizer.step()
                    device.backward()


                    #loss = criterion(output, target)#[64, 10] [64] 
                    
                    #diffloss=Diffloss(gen_output,output)
                    # tarloss=1/criterion(gen_output, target)
                    
                    # #loss.backward(retain_graph=True)#第一个不释放梯度
                    # #diffloss.backward()
                    # tarloss.backward()

                    _, predicted = torch.max(Y_fake.data, 1)
                    total += target.size(0)
                    print(total)
                    attackeracc += (predicted == target).sum()


                    
                    # generator.backward()
                    # Discriminator.backward()
            # else:
            #     for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Epoch '+str(epoch+1)+', Device '+str(device_index))):
                    
            #         device.zero_grad()
            #         edge.zero_grad()
                    
                    
            #         sample, target = data
                    
            #         sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long() #数据移至GPU
                    
            #         inter_matrix = device.forward(sample)
                    
            #         output = edge.forward(inter_matrix)
                    
            #         loss = criterion(output, target)
                    
            #         loss.backward()
                    
            #         total_loss += loss.item()
            #         _, predicted = torch.max(output.data, 1)
            #         total += target.size(0)
            #         correct += (predicted == target).sum()

            #         edge.backward()
            #         device.backward()


                
            device.model.to(CPU_DEVICE)
            
            if(device_index<1):
                attackeracc = float(attackeracc) * 100.0 / total
                print('attacker acc',attackeracc)
            # else:
            #     acc_train = float(correct) * 100.0 / total
            #     print('Train Acc:', acc_train, ', Loss', total_loss/total)
            #     loss_log.append(str(total_loss/total))
            #     train_acc_log.append(str(acc_train))
            #     loss_all+=total_loss/total
            

            
        #Discriminator.load_state_dict(edge.model.state_dict())#删掉了model.
        loss_log2.append(loss_all/3)

        

        # 聚合client侧和server侧的模型

        # 聚合client侧
        # weights_list = []

        # for device_index in range(6):
        #     device = device_list[device_index]
        #     device.eval()
            
        #     weights_list.append(device.state_dict())#删掉了model.
            
        # device_weights = average_weights(weights_list)
   


        # for device_index in range(6):
        #     device = device_list[device_index]
        #     device.load_state_dict(device_weights)#删掉了model.
        #        # Test for Every Epoch

        # print('Test for Every Epoch')

        device = device_list[0]
        
        if args.gpu != 'cpu':
            torch.cuda.empty_cache()
        device.model.to(DEVICE)#删掉了model.
        
        if(epoch==1499):
                 
                 torch.save(generator.state_dict(), '1500GAN.pt')#删掉了model.
        # if(epoch==198):
        #          torch.save(generator.state_dict(), '200GAN.pt')#删掉了model.
        acc_test = valid(device, edge, test_loader)
        device.model.to(CPU_DEVICE)#删掉了model.
        
        test_acc_log.append(acc_test)
        #print('Test Acc:', acc_test,', Train Acc:', acc_train, ', Loss', total_loss/total)

        # Test End
        # =======================================

        # 写入log
        with open(args.log_file_name, 'a', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            write_list = [model_name, str(split_layer), str(epoch+1), str(acc_test)] + loss_log + train_acc_log
            csv_writer.writerow(write_list)
    # drawLoss(loss_log2)
    # drawAcc(test_acc_log)
    # draw2(test_acc_log,loss_log2,trainlist)

if __name__ == '__main__':
    main()