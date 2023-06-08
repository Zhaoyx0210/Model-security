
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


import random
import numpy as np

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def multi_krum(all_updates, n_attackers, multi_k=False):
    nusers = all_updates.shape[0]
    candidates = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        distances = []
        for update in remaining_updates:
            distance = torch.norm((remaining_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    # print(len(remaining_updates))
    aggregate = torch.mean(candidates, dim=0)
    return aggregate, np.array(candidate_indices)

def compute_lambda(all_updates, model_re, n_attackers):

    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1)
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(distances[:, :n_benign - 2 - n_attackers], dim=1)
    min_score = torch.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0])
    max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (torch.sqrt(torch.Tensor([d]))[0])

    return (term_1 + max_wre_dist)


def get_malicious_updates_fang(all_updates, model_re, deviation, n_attackers):

    lamda = compute_lambda(all_updates, model_re, n_attackers)

    threshold = 1e-5
    mal_update = []

    while lamda > threshold:
        mal_update = (-lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        # print(mal_updates.shape, n_attackers)
        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=False)
        if krum_candidate < n_attackers:
            # print('successful lamda is ', lamda)
            return mal_update
        else:
            mal_update = []

        lamda *= 0.5

    if not len(mal_update):
        mal_update = (model_re - lamda * deviation)
        
    return mal_update

def our_attack_mkrum(all_updates, model_re, n_attackers,dev_type='unit_vec'):

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
        
    lamda = torch.Tensor([20.0]).cuda() #compute_lambda_our(all_updates, model_re, n_attackers)
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=True)

        if np.sum(krum_candidate < n_attackers) == n_attackers:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    return mal_update

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
parser.add_argument('--gpu', default='cuda', type=str, help='GPU')
parser.add_argument('--epoch_num', default=600, type=int, help='Epoch Number')
parser.add_argument('--log_file_name', default='single_VGG19', type=str, help='single_VGG19')
parser.add_argument('--split_layer', default=3, type=int, help='split layer')
parser.add_argument('--dev_type', default='unit_vec', type=str, help='dev_type')

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

# 生成client模型
for i in range(6):

    fix_random_seed()

    device_model = get_device_vgg(model_name, split_layer).to(CPU_DEVICE)
    device_optimizer = optim.SGD(device_model.parameters(), lr = device_lr)
    
    temp_device = copy.deepcopy(DLModel(device_model, device_optimizer))
    device_list.append(temp_device)

# 生成server模型
for i in range(6):

    fix_random_seed()

    edge_model =  get_edge_vgg(model_name, split_layer).to(CPU_DEVICE)
    edge_optimizer = optim.SGD(edge_model.parameters(), lr = edge_lr)
    
    temp_edge = copy.deepcopy(DLModel(edge_model, edge_optimizer))
    edge_list.append(temp_edge)


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
    for epoch in range(epoch_num):
        
        loss_log = []
        test_acc_log = []
        train_acc_log = []
        
        for device_index in range(len(device_list)):

            device = device_list[device_index]
            edge = edge_list[device_index]
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
                
                inter_matrix = device.forward(sample)
                
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

        
        # =======================================
        # 聚合client侧和server侧的模型

        # 聚合client侧
        weights_list = []

        for device_index in range(device_num):
            
            device = device_list[device_index]
            device.eval()
            
            weights_list.append(device.model.state_dict())
            
        device_weights = average_weights(weights_list)
        
        for device_index in range(device_num):
            device = device_list[device_index]
            device.model.load_state_dict(device_weights)

        # 聚合server
        weights_list = []

        for device_index in range(device_num):
            edge = edge_list[device_index]
            edge.eval()
            
            weights_list.append(edge.model.state_dict())
            
        edge_weights = average_weights(weights_list)
        
        for device_index in range(device_num):
            edge = edge_list[device_index]
            edge.model.load_state_dict(edge_weights)

        # 聚合结束
        # =======================================


        # =======================================
        # Test for Every Epoch

        print('Test for Every Epoch')

        device = device_list[0]
        edge = edge_list[0]
        if args.gpu != 'cpu':
            torch.cuda.empty_cache()
        device.model.to(DEVICE)
        edge.model.to(DEVICE)
        acc_test = valid(device, edge, test_loader)
        device.model.to(CPU_DEVICE)
        edge.model.to(CPU_DEVICE)
        print('Test Acc:', acc_test,', Train Acc:', acc_train, ', Loss', total_loss/total)

        # Test End
        # =======================================

        # 写入log
        with open(args.log_file_name, 'a', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            write_list = [model_name, str(split_layer), str(epoch+1), str(acc_test)] + loss_log + train_acc_log
            csv_writer.writerow(write_list)

if __name__ == '__main__':
    main()