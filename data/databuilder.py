#!/usr/env/bin python
from typing import Tuple, Callable, List

from torch.utils.data import DataLoader, Sampler, BatchSampler, RandomSampler
from torchvision import transforms
import copy

from data.images.MNIST import MNIST
from data.images.CIFAR import CIFAR10
from data.images.ImageNet32 import ImageNet32Dataset
from data.datautils import PairBatchSampler

class DataBuilder():
    
    def __init__(self, datasetname:str, train_ratio:float, batch_size:int, methodname_list:List[Tuple], **kwargs) -> None:
        
        self.datasetname = datasetname
        self.train_ratio = train_ratio
        self.batch_size  = batch_size 
        self.methodname_list = methodname_list
        self.calibrate_loader_dict = {}
    
    #TODO: transfer all these method conditioned operation to pre_calibration
    
    @staticmethod
    def get_dataset(dataset_name: str) -> Callable:
        
        if dataset_name == 'mnist':
            return MNIST
        elif dataset_name == 'cifar10':
            return CIFAR10
        elif 'imagenet32' in dataset_name:
            return ImageNet32Dataset
        else:
            raise NotImplementedError('Dataset name should be in (mnist, cifar10, imagenet32) but got '+f'{dataset_name} !')

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
        
        elif dataset_name == 'imagenet32_1channel':
            
            transform_base = [
                transforms.Resize(28),
                transforms.Grayscale(1),  # Single-channel grayscale image
                transforms.ToTensor(), 
                transforms.Normalize((0.1307,), (0.3081,))
            ]
            transform_train = transforms.Compose(
                [transforms.RandomCrop(28, padding=2)] + transform_base
            )           
            transform_test  = transform_base
        
        elif dataset_name == 'imagenet32_3channel':
            
            transform_base = [
                transforms.ToTensor(), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
            
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            ] + transform_base)
            
            transform_test = transform_base
        
        else:
            
            raise NotImplementedError("Expect the dataset to be in (mnist, cifar) but found {dataset_name}!")
        
        return transform_train, transform_test   
    
    @staticmethod
    def get_sampler(method_list: List, batch_size:int) -> Sampler:
        
        if 'cskd' in method_list:
            if 'oursv3' in method_list:
                return lambda x: PairBatchSampler(x, num_samples=int(0.8*len(x)), batch_size=batch_size)
            else:
                return lambda x: PairBatchSampler(x, batch_size=batch_size)
        else:
            if 'oursv3' in method_list:
                return lambda x: BatchSampler(RandomSampler(x, num_samples=int(0.8*len(x))), batch_size=batch_size, drop_last=False)
            else:
                return lambda x: BatchSampler(RandomSampler(x), batch_size=batch_size, drop_last=False)
    
    def create(self):
        
        dataset = self.get_dataset(self.datasetname)
        transform_train, transform_test = self.get_transform(self.datasetname)
        sampler = self.get_sampler(self.methodname_list, batch_size=self.batch_size)
        
        self.trainset = dataset(root="./download", split="train", train_ratio=self.train_ratio, download=True, transform=transform_train)
        self.validset = dataset(root="./download", split="valid", train_ratio=self.train_ratio, download=True, transform=transform_test)
        self.testset  = dataset(root='./download', split="test",  download=True, transform=transform_test)
        
        self.train_loader = DataLoader(self.trainset, batch_sampler=sampler(self.trainset), num_workers=1)
        self.valid_loader = DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.test_loader  = DataLoader(self.testset,  batch_size=self.batch_size, shuffle=False, num_workers=1)        
        
        for method in self.methodname_list:
            
            if method == 'lula':
                in_channels = 1 if self.datasetname=='mnist' else 3
                calibrate_dataset = f'imagenet32_{in_channels}channel'
                calibrate_transform_train, _ = self.get_transform(calibrate_dataset)
                calibrate_dataset = self.get_dataset(calibrate_dataset)
                calibrate_dataset = calibrate_dataset(root="/scr/songzhu/", transform=calibrate_transform_train)
            elif method in ['gp', 'ours']:
                calibrate_dataset = copy.deepcopy(self.trainset)
            else:
                calibrate_dataset = self.validset
            
            calibrate_loader = DataLoader(calibrate_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
            self.calibrate_loader_dict[method] = calibrate_loader
            if method == 'lula':
                self.calibrate_loader_dict[method] = (self.train_loader, self.valid_loader, calibrate_loader)