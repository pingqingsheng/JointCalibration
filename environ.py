#!/usr/bin/env python 

import os
from typing import MutableMapping, List
import random

import numpy as np
import torch
from datetime import datetime
import importlib

from data.databuilder import DataBuilder
from networks.networkbuilder import NetWorkBuilder
from trainer import Trainer
from calibrator.basecalibrator import BaseCalibrator


class Environ():
    
    _available_method = {'raw':'BaseCalibrator', 
                         'ts':'TemperatureScaling'}
    
    def __init__(self, config: MutableMapping) -> None:
        
        self.config = config
        self.timestamp =  datetime.today().strftime("%Y%m%d_%H%M%S")
    
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
        environ['timestamp'] = self.timestamp
        
        return environ
    
    
    def create_dataset(self) -> DataBuilder: 
        
        if self.config['args']['method'] == 'cskd':
            datasetname = self.config['args']['dataset']+'_cskd'
            calibrate_datasetname = self.config['args']['dataset']
        elif self.config['args']['method'] == 'lula':
            datasetname = self.config['args']['dataset']
            calibrate_datasetname = 'imagenet32_4_'+self.config['args']['dataset']
        else:
            datasetname = self.config['args']['dataset']
            calibrate_datasetname = datasetname
        
        databuilder = DataBuilder(
            datasetname = datasetname, 
            train_ratio = self.config['data'][self.config['args']['dataset']]['TRAIN_VALIDATION_RATIO'], 
            batch_size  = self.config['train'][self.config['args']['dataset']]['BATCH_SIZE'], 
            calibrate_datasetname = calibrate_datasetname
        )
        databuilder.create()
        
        noise_type = self.config['args']['noise_type']
        noise_strength = float(self.config['args']['noise_strength'])
        clean_model_path = self.config['clean_model'][self.config['args']['dataset']]['PATH']
        if noise_type == 'idl':
            clean_model_path = '.'+clean_model_path.split('.')[1]+f'_{int(noise_strength*100)}.pth'
            
        #TODO: remake the clean model
        f_star = NetWorkBuilder(networkname='resnet18', num_classes=10, in_channels=self.config['data'][self.config['args']['dataset']]['N_CHANNELS'])
        f_star.create()
        orig_state_dict  = f_star.model.state_dict()
        clean_state_dict = {k.replace('module.', '').replace('downsample', 'shortcut').replace('fc', 'linear'):v for k, v in torch.load(clean_model_path).items()}
        clean_state_dict = {k:v for k, v in clean_state_dict.items() if (k in orig_state_dict) and (v.shape==orig_state_dict[k].shape)}
        orig_state_dict.update(clean_state_dict)
        f_star.model.load_state_dict(orig_state_dict)
        f_star.model = f_star.model.to(self.device)
        setattr(f_star.model, 'device', self.device)
        
        databuilder.inject_noise(dataloader=databuilder.train_loader, f_star=f_star.model, noise_type=noise_type, noise_strength=noise_strength, mode='train')
        databuilder.inject_noise(dataloader=databuilder.valid_loader, f_star=f_star.model, noise_type=noise_type, noise_strength=noise_strength, mode='eval')
        databuilder.inject_noise(dataloader=databuilder.test_loader,  f_star=f_star.model, noise_type=noise_type, noise_strength=noise_strength, mode='eval')
        if self.config['args']['method'] != 'lula':
            databuilder.inject_noise(dataloader=databuilder.calibrate_loader, f_star=f_star.model, noise_type=noise_type, noise_strength=noise_strength, mode='eval')
        
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
        
        datasetname = self.config['args']['dataset']
        methodname  = self.config['args']['method']
        noise_type  = self.config['args']['noise_type']
        noise_strength = self.config['args']['noise_strength']
        seed = self.config['args']['seed']
        
        checkpoint_dir = self.config['args']['checkpoint_dir']
        
        trainer = Trainer(
            lr = self.config['train'][datasetname]['LR'], 
            weight_decay = self.config['train'][datasetname]['WEIGHT_DECAY'], 
            mile_stone=self.config['train'][datasetname]['MILE_STONE'], 
            n_epoch=self.config['train'][datasetname]['N_EPOCHS'], 
            checkpoint_path=os.path.join(checkpoint_dir, f'{datasetname}_{methodname}_{noise_type}_{noise_strength}_{seed}_{self.timestamp}.pth'), 
            device = self.device
        )
        
        return trainer
    
    
    def create_calibrator(self) -> List[BaseCalibrator]:
        
        methodname_list = self.config['args']['method'].split('+')
        calibrators = []
        
        for method in methodname_list:
            
            if method == 'raw':
                calibrators.append(BaseCalibrator())
            elif method in self._available_method:
                
                if self.config['args']['dataset'] in self.config['algorithm']:
                    calibrator_config = {k:v for k,v in self.config['algorithm'][self.config['args']['dataset']].items()}
                    
                module = importlib.import_module(method)
                calibrators.append(getattr(module(config=calibrator_config), self._available_method[method]))
            else:
                raise NotImplementedError(f"Calibrator {method} is not defined !")
            
        return calibrators
        
        
# for test purpose only, remove before releasing
if __name__ == '__main__':
    
    test = Environ
    test.create_environ()