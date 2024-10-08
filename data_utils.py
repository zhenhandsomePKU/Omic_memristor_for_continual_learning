import numpy as np
import pandas as pd
import torch
from torch.functional import Tensor
from torch.utils.data.dataset import TensorDataset
import torchvision
import matplotlib.pyplot as plt
import os
import json
from torchvision import transforms
from collections import OrderedDict
from datetime import datetime
from PIL import Image
import glob
from torch.utils.data import Dataset
import cv2

#构建kannada数据集
kannada_train= pd.read_csv('kannada_pytorch/kannada_mnist/train.csv')

def split_indices(n, val_pct):  #随机分割验证集和测试集
    n_val = int(n*val_pct)
    idx = np.random.permutation(n)
    return idx[:n_val], idx[n_val:]

test_idx, train_idx = split_indices(len(kannada_train), 0.2)
labels = kannada_train.pop('label')

labels_train = torch.tensor(labels[train_idx].to_numpy(),dtype=torch.long)
labels_test = torch.tensor(labels[test_idx].to_numpy(),dtype=torch.long)

train_X = kannada_train.iloc[train_idx, :].values
test_X = kannada_train.iloc[test_idx, :].values

train_X=np.uint8(train_X)  #转换格式，用于ToTensor
test_X=np.uint8(test_X)
train_X=train_X.reshape(train_X.shape[0],28,28,-1)
test_X=test_X.reshape(test_X.shape[0],28,28,-1)

kannada_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0,),(1.0))])
images_train=torch.empty(train_X.shape[0],1,28,28)
images_test=torch.empty(test_X.shape[0],1,28,28)

for i in range(len(train_X)):
    images_train[i,:,:,:]=kannada_transform(train_X[i,:,:,:])
for i in range(len(test_X)):
    images_test[i,:,:,:]=kannada_transform(test_X[i,:,:,:])

kamnist_dset_train=TensorDataset(images_train,labels_train)
kamnist_dset_test=TensorDataset(images_test,labels_test)

kamnist_train_loader= torch.utils.data.DataLoader(kamnist_dset_train, batch_size=100, shuffle=True, num_workers=1)
kamnist_test_loader= torch.utils.data.DataLoader(kamnist_dset_test, batch_size=100, shuffle=False, num_workers=1)



#构建chinese MNIST

#数据集标签字典
label_mapper = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 100,
    13: 1000,
    14: 10000,
    15: 100000000
}

#获取图片的一些信息
def get_digit_label(image_path):   
    """
    This function returns the code for a digit and the actual value
    of the digit by mapping the `label_code` using the `label_mapper`.
    """
    digit_file_name =  image_path.split('/')[-1].split('.')[0]
    label_code = int(digit_file_name.split('_')[-1])
    label = label_mapper[label_code]
    return label_code, label      #获得相应的标签   均为整数类型

#将0-10的数据提出来
def train_val_split_0_9(image_paths, ratio=0.15):  #获得0-9的一些图片
    """
    This functiont takes all the `image_paths` list and 
    splits the into a train and validation list.
    
    :param image_paths: list containing all the image paths
    :param ratio: train/test split ratio
    """
    new_image_paths=[]
    
    for i in image_paths:
        if (get_digit_label(i)[0]>0) & (get_digit_label(i)[0]<11):
            new_image_paths.append(i)
        
    num_images = len(new_image_paths)
    valid_ratio = int(num_images*ratio)
    train_ratio = num_images - valid_ratio
    train_images = new_image_paths[0:train_ratio]
    # leave the last 10 images to use them in final testing
    valid_images = new_image_paths[train_ratio:] 
    return train_images, valid_images    #返回训练的图片列表和验证图片列表


class ChineseMNISTDataset(Dataset):
    def __init__(self, image_list, is_train=True):
        self.image_list = image_list
        self.is_train = is_train
        
        # training transforms and augmentations
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((28, 28)),
                #transforms.Resize((28,28)),
                transforms.ToTensor(),
                transforms.Normalize(
                                     mean=[0,0,0],  #这里的图像预处理很重要，否则在BNN中几乎训练不起来
                                     std=[1.0,1.0,1.0]  ##这里的图像预处理很重要
                )
            ])
            
        # validation transforms
        if not self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((28, 28)),
                #transforms.Resize((28,28)),
                transforms.ToTensor(),
                transforms.Normalize(
                                     mean=[0,0,0],
                                     std=[1.0,1.0,1.0]
                )
            ])
            
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        image = cv2.dilate(image,kernel,iterations=3)
        image = self.transform(image)
        label_code, label = get_digit_label(self.image_list[index])
        label_code = label_code - 1
        
        return {
            'image': image,
            'label': label_code
        }

image_paths = glob.glob('cmnist_pytorch/input/data/data/*.jpg')
train_images, test_images = train_val_split_0_9(image_paths)
cmnist_dset_train = ChineseMNISTDataset(train_images, is_train=True)
cmnist_train_loader = torch.utils.data.DataLoader(cmnist_dset_train, batch_size=100, shuffle=True, num_workers=1)
cmnist_dset_test = ChineseMNISTDataset(test_images, is_train=False)
cmnist_test_loader = torch.utils.data.DataLoader(cmnist_dset_test, batch_size=100, shuffle=True, num_workers=1)


transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

kmnist_dset_train = torchvision.datasets.KMNIST('./kmnist_pytorch',train=True,transform=transform,target_transform=None, download=True)
kmnist_train_loader= torch.utils.data.DataLoader(kmnist_dset_train, batch_size=100, shuffle=True, num_workers=1)

kmnist_dset_test = torchvision.datasets.KMNIST('./kmnist_pytorch',train=False,transform=transform,target_transform=None, download=True)
kmnist_test_loader= torch.utils.data.DataLoader(kmnist_dset_test, batch_size=100, shuffle=False, num_workers=1)

mnist_dset_train = torchvision.datasets.MNIST('./mnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
mnist_train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=100, shuffle=True, num_workers=1)

mnist_dset_test = torchvision.datasets.MNIST('./mnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
mnist_test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=100, shuffle=False, num_workers=1)

fmnist_dset_train = torchvision.datasets.FashionMNIST('./fmnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
fashion_mnist_train_loader = torch.utils.data.DataLoader(fmnist_dset_train, batch_size=100, shuffle=True, num_workers=1)

fmnist_dset_test = torchvision.datasets.FashionMNIST('./fmnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
fashion_mnist_test_loader = torch.utils.data.DataLoader(fmnist_dset_test, batch_size=100, shuffle=False, num_workers=1)

usps_transform = torchvision.transforms.Compose( [torchvision.transforms.Resize((28,28)),
   torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

usps_dset_train = torchvision.datasets.USPS('./usps_pytorch', train=True, transform=usps_transform, target_transform=None, download=True)
usps_train_loader = torch.utils.data.DataLoader(usps_dset_train, batch_size=100, shuffle=True, num_workers=1)

usps_dset_test = torchvision.datasets.USPS('./usps_pytorch', train=False, transform=usps_transform, target_transform=None, download=True)
usps_test_loader = torch.utils.data.DataLoader(usps_dset_test, batch_size=100, shuffle=False, num_workers=1)

def create_permuted_loaders(task):
    
    permut = torch.from_numpy(np.random.permutation(784))
        
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                      torchvision.transforms.Lambda(lambda x: x.view(-1)[permut].view(1, 28, 28) ),
                      torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

    if task=='MNIST':
        dset_train = torchvision.datasets.MNIST('./mnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
        train_loader = torch.utils.data.DataLoader(dset_train, batch_size=100, shuffle=True, num_workers=1)

        dset_test = torchvision.datasets.MNIST('./mnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
        test_loader = torch.utils.data.DataLoader(dset_test, batch_size=1000, shuffle=False, num_workers=1)

        
    elif task=='FMNIST':
   
        dset_train = torchvision.datasets.FashionMNIST('./fmnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
        train_loader = torch.utils.data.DataLoader(dset_train, batch_size=100, shuffle=True, num_workers=1)

        dset_test = torchvision.datasets.FashionMNIST('./fmnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
        test_loader = torch.utils.data.DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=1)

    return train_loader, test_loader, dset_train


class DatasetProcessing(torch.utils.data.Dataset): 
    def __init__(self, data, target, transform=None): 
        self.transform = transform
        self.data = data.astype(np.float32)[:,:,None]
        self.target = torch.from_numpy(target).long()
    def __getitem__(self, index): 
        if self.transform is not None:
            return self.transform(self.data[index]), self.target[index]
        else:
            return self.data[index], self.target[index]
    def __len__(self): 
        return len(list(self.data))

def process_features(X_train, X_test, mode):
    if mode=="cutoff":
        cutoff = 8
        threshold_train = np.zeros((np.shape(X_train)[0],1))
        threshold_test = np.zeros((np.shape(X_test)[0],1)) 
        for i in range(np.shape(X_train)[0]):
            threshold_train[i,0] = np.unique(X_train[i,:])[-cutoff]
        for i in range(np.shape(X_test)[0]):
            threshold_test[i,0] = np.unique(X_test[i,:])[-cutoff]
        X_train =   (np.sign(X_train  - threshold_train + 1e-6 ) + 1.0)/2
        X_test =  (np.sign (X_test  - threshold_test +1e-6 ) + 1.0)/2
    elif mode=="mean_over_examples":
        X_train = ( X_train - X_train.mean(axis = 0, keepdims = True) )/ X_train.var(axis =0, keepdims = True) # ???
        X_test = ( X_test - X_test.mean(axis=0, keepdims = True) ) /X_test.var(axis = 0, keepdims = True)
    elif mode=="mean_over_examples_sign":
        X_train =   (np.sign(X_train  - X_train.mean(axis = 0, keepdims = True) ) + 1.0)/2
        X_test =  (np.sign (X_test  - X_test.mean(axis = 0, keepdims = True) ) + 1.0)/2
    elif mode=="mean_over_pixels":
        X_train = ( X_train - X_train.mean(axis = 1, keepdims = True) )/ X_train.var(axis =1, keepdims = True)  # Instance norm
        X_test = ( X_test - X_test.mean(axis=1, keepdims = True) ) /X_test.var(axis = 1, keepdims = True)
    elif mode=="mean_over_pixels_sign":
        X_train =   (np.sign(X_train  - X_train.mean(axis = 1, keepdims = True) ) + 1.0)/2  
        X_test =  (np.sign (X_test  - X_test.mean(axis = 1, keepdims = True) ) + 1.0)/2
    elif mode=="global_mean":
        X_train = ( X_train - X_train.mean(keepdims = True) )/ X_train.var(keepdims = True) # Batch norm
        X_test = ( X_test - X_test.mean(keepdims = True) ) /X_test.var(keepdims = True)
    elif mode=="rescale":
        X_train =  (X_train / X_train.max(axis = 1, keepdims = True) )
        X_test =  (X_test / X_test.max(axis = 1, keepdims = True) )
    return X_train, X_test


def relabel(label):
    label_map = [5,6,0,1,2,3,4,7,8,9]
    return label_map[label]

vrelabel = np.vectorize(relabel)


def process_cifar10(subset):

    cifar_X_train = torch.load('cifar10_features_dataset/train.pt').cpu().numpy()
    cifar_Y_train = torch.load('cifar10_features_dataset/train_targets.pt').cpu().numpy() 
    cifar_X_test = torch.load('cifar10_features_dataset/test.pt').cpu().numpy() 
    cifar_Y_test = torch.load('cifar10_features_dataset/test_targets.pt').cpu().numpy()

    cifar_Y_train = vrelabel(cifar_Y_train)
    cifar_Y_test = vrelabel(cifar_Y_test)

    if subset=='animals':
        partition = np.vectorize(lambda l: l < 5) 
    elif subset=='vehicles':
        partition = np.vectorize(lambda l: l >= 5)  
    else:
        raise('error unsuported subset')
 
    mode = 'mean_over_pixels'
    sub_X_train = cifar_X_train[partition(cifar_Y_train)] 
    sub_X_test = cifar_X_test[partition(cifar_Y_test)] 
 
    sub_X_train, sub_X_test = process_features(sub_X_train, sub_X_test, mode) 
 
    sub_Y_train = cifar_Y_train[partition(cifar_Y_train)] 
    sub_Y_test = cifar_Y_test[partition(cifar_Y_test)] 
 
    sub_dset_train = DatasetProcessing(sub_X_train, sub_Y_train) 
    sub_train_loader = torch.utils.data.DataLoader(sub_dset_train, batch_size=100, shuffle=True, num_workers=4) 
 
    sub_dset_test = DatasetProcessing(sub_X_test, sub_Y_test) 
    sub_test_loader = torch.utils.data.DataLoader(sub_dset_test, batch_size=100, shuffle=False, num_workers=0) 
 
    return sub_train_loader, sub_test_loader, sub_dset_train 



def process_cifar100(n_subset):
    subset_size = 100//n_subset

    train_loader_list = []
    test_loader_list = []
    dset_train_list = []

    cifar100_X_train = torch.load('cifar100_features_dataset/train.pt').cpu().numpy()
    cifar100_Y_train = torch.load('cifar100_features_dataset/train_targets.pt').cpu().numpy()
    cifar100_X_test = torch.load('cifar100_features_dataset/test.pt').cpu().numpy()
    cifar100_Y_test = torch.load('cifar100_features_dataset/test_targets.pt').cpu().numpy()

    for k in range(n_subset):
        partition = np.vectorize(lambda l: ((l < (k+1)*subset_size) and (l >= k*subset_size)) )
        mode = 'mean_over_pixels'
        sub_X_train = cifar100_X_train[partition(cifar100_Y_train)]
        sub_X_test = cifar100_X_test[partition(cifar100_Y_test)]

        sub_X_train, sub_X_test = process_features(sub_X_train, sub_X_test, mode)

        sub_Y_train = cifar100_Y_train[partition(cifar100_Y_train)]
        sub_Y_test = cifar100_Y_test[partition(cifar100_Y_test)]

        sub_dset_train = DatasetProcessing(sub_X_train, sub_Y_train)
        sub_train_loader = torch.utils.data.DataLoader(sub_dset_train, batch_size=20, shuffle=True, num_workers=4)

        sub_dset_test = DatasetProcessing(sub_X_test, sub_Y_test)
        sub_test_loader = torch.utils.data.DataLoader(sub_dset_test, batch_size=20, shuffle=False, num_workers=0)

        train_loader_list.append(sub_train_loader)
        test_loader_list.append(sub_test_loader)
        dset_train_list.append(sub_dset_train)

    return train_loader_list, test_loader_list, dset_train_list


def createHyperparametersFile(path, args):

    hyperparameters = open(path + r"/hyperparameters.txt","w+")
    L = ["- scenario: {}".format(args.scenario) + "\n",
        "- interleaved: {}".format(args.interleaved) + "\n",
        "- hidden layers: {}".format(args.hidden_layers) + "\n",
        "- normalization: {}".format(args.norm) + "\n",
        "- net: {}".format(args.net) + "\n",
        "- task sequence: {}".format(args.task_sequence) + "\n",
        "- lr: {}".format(args.lr) + "\n",
        "- gamma: {}".format(args.gamma) + "\n",
        "- meta: {}".format(args.meta) + "\n",
        "- beaker: {}".format(args.beaker) + "\n",
        "- number of beakers: {}".format(args.n_bk) + "\n",
        "- ratios: {}".format(args.ratios) + "\n",
        "- areas: {}".format(args.areas) + "\n",
        "- feedback: {}".format(args.fb) + "\n",
        "- ewc: {}".format(args.ewc) + "\n",
        "- ewc lambda: {}".format(args.ewc_lambda) + "\n",
        "- SI: {}".format(args.si) + "\n",
        "- Binary Path Integral: {}".format(args.bin_path) + "\n",
        "- SI lambda: {}".format(args.si_lambda) + "\n",
        "- decay: {}".format(args.decay) + "\n",
        "- epochs per task: {}".format(args.epochs_per_task) + "\n",
        "- init: {}".format(args.init) + "\n",
        "- init width: {}".format(args.init_width) + "\n",
        "- seed: {}".format(args.seed) + "\n",
        "- bit_num :{}".format(args.bit_num)+"\n",
        "- upper_bound :{}".format(args.upper_bound)+"\n",
        "- noise_std :{}".format(args.noise_std)+"\n"]
   
    hyperparameters.writelines(L)
    hyperparameters.close()
        







