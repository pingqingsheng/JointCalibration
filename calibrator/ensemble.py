#!/usr/env/bin python
from typing import Tuple, Union, List

import torch
from copy import deepcopy

from .basecalibrator import BaseCalibrator

class Ensemble(BaseCalibrator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.num_models = kwargs['config']['NUM_MODELS']
    
    def forward(self, x:torch.Tensor, **kwargs) -> List[Union[torch.Tensor, torch.distributions.Distribution]]:
        
        ensemble_logits = []
        for model in self.model_list: 
            logits = model(x)
            ensemble_logits.append(logits) 
        
        return ensemble_logits
    
    def get_prob(self, logits_list: List[Union[torch.Tensor, torch.distributions.Distribution]], **kwargs) -> torch.Tensor:
        
        assert isinstance(logits_list, List), f'logits_list is supposed to be of type {List}!'
        
        prob_ensemble = 0
        for i in range(len(self.model_list)):
            model_i = self.model_list[i]
            prob_ensemble += model_i.get_prob(logits_list[i])
        
        return prob_ensemble/len(self.model_list)
    
    def pre_calibrate(self, 
                      model: torch.nn.Module, 
                      optimizer: torch.optim.Optimizer, 
                      **kwargs) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        
        self.model = model
        network = model
        while isinstance(network, BaseCalibrator):
            network = network.model
        
        if hasattr(self.model, 'likelihood'): # GP module
            self.common_likelihood = self.model.likelihood
            self.common_gplayer = network.gp_layer
        
        self.model_list = []
        exist_dataptr = set([x.data_ptr() for x in optimizer.param_groups[0]['params']])
        
        for i in range(self.num_models):
            
            model_i = deepcopy(model)
            for module in model_i.children():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    module.reset_parameters()
            
            if hasattr(model_i, 'likelihood'):
                # share likelihood and gp-layer
                model_i.likelihood = self.common_likelihood  
                model_i.model.gp_layer = self.common_gplayer
                self.model_list.append(model_i.to(self.device))
                optimizer = model_i.update_optimizer(model_i.model, optimizer, self.model.likelihood)
            else:
                orig_lr = optimizer.param_groups[0]['lr']
                orig_weightdecay = optimizer.param_groups[0]['weight_decay']
                optimizer.add_param_group({'params': [x for x in model_i.parameters() if x.data_ptr() not in exist_dataptr], 'lr':orig_lr, 'weight_decay':orig_weightdecay})
                self.model_list.append(model_i.to(self.device))
                
        return self, optimizer
    
    @staticmethod
    def loss(*args, **kwargs) -> Union[torch.Tensor, int]:
        return 0