#!/usr/env/bin python

from typing import List, MutableMapping
from unittest import result

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
                 monitor_window:int=1, 
                 device:torch.DeviceObjType=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> None:
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.mile_stone = mile_stone
        self.n_epoch = n_epoch
        self.checkpoint_window = checkpoint_window
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.monitor_window = monitor_window
        self.device = device
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric_train = AverageMeter(['l1', 'ece', 'acc', 'loss'], name='train')
        self.metric_valid = AverageMeter(['l1', 'ece', 'acc', 'loss'], name='valid')
        self.metric_test  = AverageMeter(['l1', 'ece', 'acc', 'loss'], name='test')        
        self.timestamp   = datetime.today().strftime("%Y%m%d%H%M%S")
        
    def train(self, 
              model: torch.nn.Module, 
              trainloader: torch.utils.data.DataLoader, 
              validloader: torch.utils.data.DataLoader, 
              testloader: torch.utils.data.DataLoader, 
              calibrateloader: torch.utils.data.DataLoader, 
              calibrators: List[BaseCalibrator]):
    
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr = self.lr,
            weight_decay = self.weight_decay, 
            momentum = 0.9, 
            nesterov = True
        )
        
        for calibrator in calibrators:
            model, optimizer = calibrator.pre_calibrate(model=model, optimizer=optimizer, calibrateloader=calibrateloader)
        model = model.to(self.device)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.n_epoch)
        
        for epoch in tqdm(range(self.n_epoch), ncols=100, leave=True, position=0):
            
            model.train()
            for _, (ind, images, labels, eta_tilde) in enumerate(trainloader):
                
                images, labels = images.to(self.device), labels.to(self.device)
                outs = model(images)
                loss = self.criterion(outs, labels) + sum([calibrator.criterion(outs, labels) for calibrator in calibrators])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                self.metric_train.update(outs.detach().cpu(), labels.detach().cpu(), loss.item(), eta_tilde)
            
            scheduler.step()
            self.metric_train.flush()
            
            for calibrator in calibrators:
                calibrator.post_calibrate(model=model, optimizer=optimizer, calibrateloader=calibrateloader)
            
            if self.verbose and epoch%self.monitor_window==0:
                
                self.eval(model, validloader, self.metric_valid)
                self.eval(model, testloader,  self.metric_test)
                
                tqdm.write(30*"-" + f"[{int(self.n_epoch):3d}|{epoch+1:3d}]" + 30*"-")
                tqdm.write(self.logging(self.metric_train, 'loss') + '\t'+ self.logging(self.metric_valid, 'loss')+ '\t'+ self.logging(self.metric_test, 'loss'))
                tqdm.write(self.logging(self.metric_train, 'l1')  +  '\t' + self.logging(self.metric_valid, 'l1') + '\t' + self.logging(self.metric_test, 'l1'))
                tqdm.write(self.logging(self.metric_train, 'acc') +  '\t' + self.logging(self.metric_valid, 'acc')+ '\t' + self.logging(self.metric_test, 'acc'))
                tqdm.write(self.logging(self.metric_train, 'ece') +  '\t' + self.logging(self.metric_valid, 'ece')+ '\t' + self.logging(self.metric_test, 'ece'))

            if epoch%self.checkpoint_window==0:
                torch.save(model, self.checkpoint_path)
                
            
    @torch.no_grad()
    def eval(self, 
             model: torch.nn.Module, 
             testloader: torch.utils.data.DataLoader, 
             logger:AverageMeter = None) -> MutableMapping:
        
        if logger is None:
            logger = AverageMeter(['l1', 'ece', 'acc', 'loss'])
        
        model.eval()
        for _, (ind, images, labels, eta_tilde) in enumerate(testloader):
            
            images, labels = images.to(self.device), labels.to(self.device)
            outs = model(images)
            loss = self.criterion(outs, labels)
            
            logger.update(outs.detach().cpu(), labels.detach().cpu(), loss.detach().cpu(), eta_tilde)
        
        logger.flush()
        
        result_dict = {}
        for metric in logger.metric_list:
            result_dict[metric] = logger.metric_dict[f'{metric}_history']
        
        return result_dict
        
    @staticmethod
    def logging(meter:AverageMeter, metric:str):
        return f'{meter.name:5s}_{metric:4s}: {meter.get(metric):.3f}'
    