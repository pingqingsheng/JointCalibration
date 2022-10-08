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
from noise import inject_noise

class Environ():
    
    _available_method = {'raw':'BaseCalibrator', 
                         'ts':'TemperatureScaling', 
                         'ensemble':'Ensemble', 
                         'mcdrop': 'MCDrop', 
                         'cskd': 'CSKD', 
                         'focal': 'Focal', 
                         'bm': 'BeliefMatching', 
                         'gp': 'GP', 
                         'lula': 'LULA'}
    
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
        
        gpu = self.config['args']['gpu']
        self.device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
        
        environ['gpu'] = gpu
        environ['result_dir'] = self.config['args']['result_dir']
        environ['checkpoint_dir'] = self.config['args']['checkpoint_dir']
        environ['fig_dir'] = self.config['args']['fig_dir']
        environ['timestamp'] = self.timestamp
        
        return environ
    
    
    def create_dataset(self) -> DataBuilder: 
                
        methodname_list = self.config['args']['method'].split('+')
        datasetname = self.config['args']['dataset']
            
        databuilder = DataBuilder(
            datasetname = datasetname, 
            train_ratio = self.config['data'][self.config['args']['dataset']]['TRAIN_VALIDATION_RATIO'], 
            batch_size  = self.config['train'][self.config['args']['dataset']]['BATCH_SIZE'], 
            methodname_list=methodname_list
        )
        databuilder.create()
        
        noise_type = self.config['args']['noise_type']
        noise_strength = float(self.config['args']['noise_strength'])
        
        #>>> TODO: remake the clean model
        clean_model_path = self.config['clean_model'][self.config['args']['dataset']]['PATH']
        if noise_type == 'idl':
            clean_model_path = '.'+clean_model_path.split('.')[1]+f'_{int(noise_strength*100)}.pth'

        f_star = NetWorkBuilder(networkname='resnet18', num_classes=10, in_channels=self.config['data'][self.config['args']['dataset']]['N_CHANNELS'])
        f_star.create()
        orig_state_dict  = f_star.model.state_dict()
        clean_state_dict = {k.replace('module.', '').replace('downsample', 'shortcut').replace('fc', 'linear'):v for k, v in torch.load(clean_model_path).items()}
        clean_state_dict = {k:v for k, v in clean_state_dict.items() if (k in orig_state_dict) and (v.shape==orig_state_dict[k].shape)}
        orig_state_dict.update(clean_state_dict)
        f_star.model.load_state_dict(orig_state_dict)
        f_star.model = f_star.model.to(self.device)
        setattr(f_star.model, 'device', self.device)
        # <<<
        
        databuilder.train_loader = inject_noise(databuilder.train_loader, f_star=f_star.model, noise_type=noise_type, noise_strength=noise_strength, mode='train')
        databuilder.valid_loader = inject_noise(databuilder.valid_loader, f_star=f_star.model, noise_type=noise_type, noise_strength=noise_strength, mode='eval')
        databuilder.test_loader  = inject_noise(databuilder.test_loader,  f_star=f_star.model, noise_type=noise_type, noise_strength=noise_strength, mode='eval')
        for k in databuilder.calibrate_loader_dict:
            if k != 'lula':
                databuilder.calibrate_loader_dict[k] = inject_noise(databuilder.calibrate_loader_dict[k], 
                                                                    f_star = f_star.model, 
                                                                    noise_type=noise_type, 
                                                                    noise_strength=noise_strength, 
                                                                    mode='eval')
            
        return databuilder

    
    def create_network(self) -> NetWorkBuilder:
        
        num_classes = int(self.config['data'][self.config['args']['dataset']]['N_CLASSES'])
        in_channels = int(self.config['data'][self.config['args']['dataset']]['N_CHANNELS'])
        in_dim = None
        
        if 'mcdrop' in self.config['args']['method']:
            networkname = 'resnet18_mc'
        elif 'gp' in self.config['args']['method']:
            networkname = 'resnet18_gp'
            in_dim = self.config['algorithm']['gp']['ENCODE_DIM']
        else:
            networkname = 'resnet18'
        
        networkbuilder = NetWorkBuilder(
            networkname = networkname, 
            num_classes = num_classes, 
            in_channels = in_channels,
            in_dim = in_dim
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
    
    
    def create_calibrator(self, databuilder: DataBuilder) -> List[BaseCalibrator]:
        
        methodname_list = self.config['args']['method'].split('+')
        calibrators = []
        calibrator_config = {'device':self.device}
        
        for method in methodname_list:
            
            calibrate_loader = databuilder.calibrate_loader_dict[method]
            
            if method == 'raw':
                calibrators.append(BaseCalibrator(calibrate_loader))
            elif method in self._available_method:
                if method in self.config['algorithm']:
                    calibrator_config.update({k:v for k,v in self.config['algorithm'][method].items()})    
                module = importlib.import_module('calibrator.'+method)
                calibrator = getattr(module, self._available_method[method])
                calibrators.append(calibrator(calibrate_loader=calibrate_loader, config=calibrator_config))
            else:
                raise NotImplementedError(f"Calibrator {method} is not defined !")
            
        return calibrators
        
        
# for test purpose only, remove before releasing
if __name__ == '__main__':
    
    test = Environ
    test.create_environ()