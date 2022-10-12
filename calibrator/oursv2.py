#! /usr/env/bin python

from typing import Tuple, Union, List

import torch
from torch.utils.data import BatchSampler, RandomSampler

from .basecalibrator import BaseCalibrator
from networks.networks import ResNet18

class JointCalibrationV2(BaseCalibrator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.criterion_mse = torch.nn.MSELoss()
        self.in_channel = 1 if len(self.calibrate_loader.dataset.data[0].shape) == 2 else 3
    
    def forward(self, x, mode='train'):
        
        if mode=='train':
            self.calibrate_network.train()
        else:
            self.calibrate_network.eval()
        
        f = self.model(x)
        g = self.calibrate_network(x)
        
        return torch.cat([f, g], 1)
    
    def get_prob(self, logits:Union[torch.Tensor, List[Union[torch.Tensor, torch.distributions.Distribution]]], **kwargs) -> torch.Tensor: 
        
        f = self.model.get_prob(logits)
        
        if isinstance(logits, List): # Ensemble
            g = 0
            for i in range(len(logits)):
                logits_i = logits[i]
                if isinstance(logits_i, torch.Tensor):
                    calibrate_logits = logits_i[:, -1]
                else: # for GP module
                    calibrate_logits = logits_i.sample_n(32).mean(0)[:, -1]
                g += torch.sigmoid(calibrate_logits).view(-1, 1)
            g /= len(logits)
        else:
            if isinstance(logits, torch.Tensor):
                calibrate_logits = logits[:, -1]
            else:
                calibrate_logits = logits.sample_n(32).mean(0)[:, -1]
            g = torch.sigmoid(calibrate_logits).view(-1, 1)
        
        _, pred = f.max(1)
        pred = pred.view(-1, 1)
        # re-normalize the prob simplex according to calibration probability
        f.scatter_(0, pred, g)
        f /= f.sum(1).view(-1, 1)
        
        return f
    
    def post_calibrate(self, **kwargs) -> None:
        # """Tune the tempearature of the model (using the validation set).
        # We're going to set it to optimize NLL.
        # valid_loader (DataLoader): validation set loader
        # """
        
        if isinstance(self.model, BaseCalibrator):
            self.model.post_calibrate()
        
        self.calibrate_network = ResNet18(in_channels=self.in_channel, num_classes=1).to(self.device)
        optimizer_calibrate = torch.optim.SGD(self.calibrate_network.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9, nesterov=True)
        
        self.calibrate_network.train()
        for _ in range(5):
            for _, (_, imgs, labels, _) in enumerate(self.calibrate_loader):
                
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                calibrate_logits = self.calibrate_network(imgs)
                loss = self.criterion_mse(torch.sigmoid(calibrate_logits).squeeze(), labels.float())
                
                optimizer_calibrate.zero_grad()
                loss.backward()
                optimizer_calibrate.step()    
    
    @staticmethod
    def loss(logits:Union[torch.Tensor, torch.nn.Module], targets:torch.Tensor, **kwargs) -> Union[torch.Tensor, int]:
        return 0