#!/usr/bin/env python 

import os
from typing import MutableMapping
import random

import numpy as np
import torch

from data.databuilder import DataBuilder
from networks.networkbuilder import NetWorkBuilder
from trainer import Trainer
from utils import *


class Environ():
    
    def __init__(self, config: MutableMapping) -> None:
        
        self.config = config
        
    
    def create_environ(self) -> MutableMapping:
        
        environ = {}
        
        seed = int(self.config['args']['seed'])
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['args']['gpu']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        environ['result_dir'] = self.config['args']['result_dir']
        environ['checkpoint_dir'] = self.config['args']['checkpoint_dir']
        environ['fig_dir'] = self.config['args']['fig_dir']
        
        return environ
    
    
    def create_dataset(self) -> DataBuilder: 
        
        if self.config['args']['method'] == 'cskd':
            datasetname = self.config['args']['dataset']+'_cskd'
            calibrate_datasetname = self.config['args']['dataset']
        elif self.config['args']['method'] == 'lula':
            datasetname = self.config['args']['dataset']
            calibrate_datasetname = 'imagenet32_4_'+self.config['args']['dataset']
        elif self.config['args']['method'] == 'ours':
            datasetname = self.config['args']['dataset']+'_ours'        
            calibrate_datasetname = datasetname
        else:
            datasetname = self.config['args']['dataset']
            calibrate_datasetname = datasetname
        
        databuilder = DataBuilder(
            datasetname = datasetname, 
            train_ratio = self.config['train'][self.config['args']['dataset']]['TRAIN_VALIDATION_RATIO'], 
            batch_size  = self.config['train'][self.config['args']['dataset']]['BATCH_SIZE'], 
            calibrate_datasetname = calibrate_datasetname
        )
        databuilder.create()
        
        return databuilder

    
    def create_network(self) -> NetWorkBuilder:
        
        num_classes = int(self.config['data'][self.config['args']['dataset']]['N_CLASSES'])
        in_channels = int(self.config['data'][self.config['args']['dataset']]['N_CHANNELS'])
        
        if self.config['args']['method'] == 'mcdrop':
            networkname = 'resnet18_mc'
        elif self.config['args']['method'] == 'gp':
            networkname = 'resnet18_gp'
        else:
            networkname = 'resnet18'
        
        networkbuilder = NetWorkBuilder(
            networkname = networkname, 
            num_classes = num_classes, 
            in_channels = in_channels
        )
        networkbuilder.create()
        
        return networkbuilder
    
    
    def create_trainer(self) -> Trainer: 
        pass
    
    
    def create_calibrator(self) -> None:
        pass



# for test purpose only, remove before releasing
if __name__ == '__main__':
    
    test = Environ
    test.create_environ()