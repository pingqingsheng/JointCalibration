#!/usr/env/bin python
import torch

from .basecalibrator import BaseCalibrator

class TemperatureScaling(BaseCalibrator):
    
    def __init__(self, **kwargs) -> None:
        super(TemperatureScaling, self).__init__()
        
        self.device = kwargs['config']['device']
        self.calibrate_loader = kwargs['calibrate_loader']

        self.temperature = 1.5*torch.ones(1).to(self.device)
        self.temperature.requires_grad_()
    
    def forward(self, x:torch.Tensor, mode:str='train'):
        
        logits = self.model(x)
        if mode == 'eval':
            logits = self.temperature_scale(logits)
            
        return logits 
    
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature         
    
    def post_calibrate(self, 
                       optimizer: torch.optim.Optimizer, 
                       **kwargs) -> None:
        
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        
        if isinstance(self.model, BaseCalibrator):
            self.model.post_calibrate(optimizer)
        
        nll_criterion = torch.nn.CrossEntropyLoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for _, input, label, _ in self.calibrate_loader:
                input  = input.to(self.device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        # before_temperature_nll = nll_criterion(logits, labels).item()
        # before_temperature_ece = ece_criterion(logits, labels).item()
        # print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)        # after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        # after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()

        # Calculate NLL and ECE after temperature scaling

        # print('Optimal temperature: %.3f' % self.temperature.item())
        # print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        
    @staticmethod
    def criterion(*args, **kwargs):
        return 0