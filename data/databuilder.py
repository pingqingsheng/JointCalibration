#!/usr/env/bin python
from typing import Tuple, Callable

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler, RandomSampler
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import copy

from data.images.MNIST import MNIST, MNIST_CSKD, PairBatchSampler
from data.images.CIFAR import CIFAR10, CIFAR10_CSKD
from data.images.ImageNet32 import ImageNet32Dataset
from .noise import noisify_with_P

class DataBuilder():
    
    def __init__(self, datasetname:str, train_ratio:float, batch_size:int, calibrate_datasetname=None, **kwargs) -> None:
        
        self.datasetname = datasetname
        self.train_ratio = train_ratio
        self.batch_size  = batch_size 
        self.calibrate_datasetname = calibrate_datasetname
        
    @staticmethod
    def get_dataset(dataset_name: str) -> Callable:
        
        if dataset_name == 'mnist':
            return MNIST
        elif dataset_name == 'mnist_cskd':
            return MNIST_CSKD
        elif dataset_name == 'cifar10':
            return CIFAR10
        elif dataset_name == 'cifar10_cskd':
            return CIFAR10_CSKD
        elif 'imagenet32' in dataset_name:
            return ImageNet32Dataset
        else:
            raise NotImplementedError('Dataset name should be in (mnist, mnist_cskd, mnist_ours, cifar10, cifar10_cskd and cifar10_cskd) but got '+f'{dataset_name} !')

    @staticmethod
    def get_transform(dataset_name: str) -> Tuple[Callable, Callable]:

        if  'mnist' in dataset_name:
            
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            
        elif 'cifar' in dataset_name:
            
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
        
        elif dataset_name == 'imagenet32_4_mnist':
            
            transform_base = [
                transforms.Resize(28),
                transforms.Grayscale(1),  # Single-channel grayscale image
                transforms.ToTensor(), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
            transform_train = transforms.Compose(
                [transforms.RandomCrop(28, padding=2)] + transform_base
            )           
            transform_test  = transform_test
        
        elif dataset_name == 'imagenet32_4_cifar10':
            
            transform_base = [
                transforms.ToTensor(), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
            
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            ] + transform_base)
            
            transform_test = transform_test
        
        else:
            
            raise NotImplementedError("Expect the dataset to be in (mnist, cifar) but found {dataset_name}!")
        
        return transform_train, transform_test   
    
    @staticmethod
    def get_sampler(dataset_name: str, batch_size:int) -> Sampler:
        
        if 'cskd' in dataset_name:
            return lambda x: PairBatchSampler(x, batch_size=batch_size)
        else:
            return lambda x: BatchSampler(RandomSampler(x), batch_size=batch_size, drop_last=False)
    
    def create(self):
        
        dataset = self.get_dataset(self.datasetname)
        transform_train, transform_test = self.get_transform(self.datasetname)
        sampler = self.get_sampler(self.datasetname, batch_size=self.batch_size)
        
        self.trainset = dataset(root="./download", split="train", train_ratio=self.train_ratio, download=True, transform=transform_train)
        self.validset = dataset(root="./download", split="valid", train_ratio=self.train_ratio, download=True, transform=transform_test)
        self.testset  = dataset(root='./download', split="test", download=True, transform=transform_test)
        
        self.train_loader = DataLoader(self.trainset, batch_sampler=sampler(self.trainset), num_workers=1)
        self.valid_loader = DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.test_loader  = DataLoader(self.testset,  batch_size=self.batch_size, shuffle=False, num_workers=1)        
        
        if self.calibrate_datasetname in ['mnist', 'cifar10']:
            self.calibrate_dataset = self.validset
        elif 'imagenet32' in self.calibrate_datasetname:
            calibrate_transform_train, _ = self.get_transform(self.calibrate_datasetname)
            calibrate_dataset = self.get_dataset(self.calibrate_datasetname)
            self.calibrate_dataset = calibrate_dataset(root="/scr/songzhu/imagenet32", transform=calibrate_transform_train)
        elif self.calibrate_datasetname in ['mnist_ours', 'cifar10_ours']:
            self.calibrate_dataset = copy.deepcopy(self.trainset)
            
        self.calibrate_loader = DataLoader(self.calibrate_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        
    def inject_noise(self, dataloader:DataLoader, f_star:torch.nn.Module, noise_type:str, noise_strength:float, mode:str) -> Dataset:
        
        eta_temp_pair = [(torch.softmax(f_star(images.to(f_star.device)), 1).detach().cpu(), indices) for _, (indices, images, _, _) in enumerate(dataloader)] 
        eta_temp, eta_indices = torch.cat([x[0] for x in eta_temp_pair]), torch.cat([x[1] for x in eta_temp_pair]).squeeze()
        
        # ground truth eta
        eta = eta_temp[eta_indices.argsort()]
        f_star_outs = eta.argmax(1).squeeze()       

        # inject noise and get eta_tilde
        if noise_type=='rcn':
            y_tilde, P, _ = noisify_with_P(np.array(copy.deepcopy(f_star_outs)), nb_classes=10, noise=noise_strength)
            self.eta_tilde = np.matmul(F.one_hot(f_star_outs, num_classes=10), P)
        elif noise_type=='idl':
            y_tilde = [int(np.where(np.random.multinomial(1, np.clip(x, 1e-3, 0.999)/np.clip(x, 1e-3, 0.999).sum(), 1).squeeze())[0]) for x in np.array(eta)]
            self.eta_tilde = copy.deepcopy(eta)
        else:
            raise NotImplementedError(f"Expecte the noise type to be in (rcn, idl), but got {noise_type}!")
        dataloader.dataset.update_eta(self.eta_tilde)
        
        # corrupt labels
        if mode == 'train':
            dataloader.dataset.update_labels(y_tilde)
        elif mode == 'eval':
            dataloader.dataset.update_labels(f_star_outs)
        else:
            raise NotImplementedError(f"Expecte the mode to be in (train, eval), but got {mode}!")