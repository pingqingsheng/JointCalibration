#!/usr/env/bin python

from typing import List, MutableMapping

import torch
from tqdm import tqdm
from datetime import datetime

from calibrator.basecalibrator import BaseCalibrator
from utils import AverageMeter

class Trainer():
    
    def __init__(self, 
                 lr:float=1e-2, 
                 weight_decay:float=1e-3,
                 mile_stone: List = None, 
                 n_epoch:int=200, 
                 checkpoint_path:str=None, 
                 checkpoint_window:int=10, 
                 verbose:bool=True, 
                 monitor_window:int=1) -> None:
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.mile_stone = mile_stone
        self.n_epoch = n_epoch
        self.checkpoint_window = checkpoint_window
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.monitor_window = monitor_window
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric_raw  = AverageMeter(['train_l1', 'test_l1', 'train_ece', 'test_ece', 'train_acc', 'test_acc', 'train_loss', 'test_loss'])
        self.metric_cali = AverageMeter(['train_l1', 'test_l1', 'train_ece', 'test_ece', 'train_acc', 'test_acc', 'train_loss', 'test_loss'])
        self.timestamp   = datetime.today().strftime("%Y%m%d%H%M%S")
        
    def train(self, 
              model_raw: torch.nn.Moduel, 
              trainloader: torch.utils.data.DataLoader, 
              validloader: torch.utils.data.DataLoader, 
              testloader: torch.utils.data.DataLoader, 
              calibrateloader: torch.utils.data.DataLoader, 
              calibrator: List[BaseCalibrator]):
    
        optimizer_raw = torch.optim.SGD(
            model_raw.parameters(), 
            lr = self.lr,
            weight_decay = self.weight_decay, 
            momentum = 0.9, 
            nesterov = True
        )
        
        model_cali, optimizer_cali = calibrator.pre_calibrate(model_raw, optimizer_raw)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_cali, T_max=self.n_epoch)
        
        for epoch in tqdm(range(self.n_epoch), ncols=100, leave=True, position=0):
            
            self.metric.reset()
            self.model_cali.train()
            
            for _, (ind, images, labels, _) in enumerate(trainloader):
                
                images, labels = images.to(model_cali.device), labels.to(model_cali.device)
                outs_raw = model_raw(images)
                loss_raw = self.criterion
                
                outs = model_cali(images)
                loss = self.criterion(outs, labels) + calibrator.criterion(outs, labels)
                
                optimizer_cali.zero_grad()
                loss.backward()
                optimizer_cali.step()
                
                self.metric.update(outs, labels, loss)
                
            calibrator.post_calibrate(model=model_cali, optimizer=optimizer_cali, calibrateloader=calibrateloader)
            
            if self.verbose and epoch%self.monitor_window==0:
                
                self.eval(testloader)
                
                tqdm.write(100*"-")
                tqdm.write(f"[{epoch:2d}|{int(self.n_epoch):2d}] \t train loss:\t\t{self.metric_history['train_loss'].val:.3f} \t\t test loss:\t{100*self.metric_history['test_loss'].val:.3f}%")
                tqdm.write(f"\t\t train acc:\t\t{self.metric_history['train_acc'].val:.3f} \t\t test acc:\t{100*self.metric_history['test_acc'].val:.3f}%")
                tqdm.write(f"\t\t train l1:\t{100*self.metric_history['train_l1'].val:.3f}% \t test l1:\t\t{100*self.metric_history['test_l1'].val:.3f}%")
                tqdm.write(f"\t\t train ece:\t{100*self.metric_history['train_ece'].val:.3f}% \t test ece:\t\t{100*self.metric_history['test_ece'].val:.3f}%")

            if epoch%self.checkpoint_window==0:
                torch.save(self.model_cali, self.checkpoint_path)
            
    def eval(self, 
             testloader: torch.utils.data.DataLoader, 
             use_best:bool = False) -> MutableMapping:
        
        pass
    