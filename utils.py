#!/usr/env/bin python

from typing import List, Union

import torch
from torch.nn import functional as F
            

class ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=30, reduction='mean'):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.reduction = reduction
        self.n_bins = n_bins
        
    def forward(self, softmaxes, labels):
        
        assert all(torch.isclose(softmaxes.sum(1), torch.ones(len(softmaxes)).to(softmaxes.device))), 'softmaxes should all sum up to 1 !'
        
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        if self.reduction == 'mean':
            ece = torch.zeros(1, device=softmaxes.device)
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                # Calculated |confidence - accuracy| in each bin
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            return ece
        
        elif self.reduction == 'sum':
            bins = []
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                # Calculated |confidence - accuracy| in each bin
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                correct_in_bin = accuracies[in_bin].float().sum()
                sum_confidence_in_bin = confidences[in_bin].sum()
                bins.append((correct_in_bin.item(), sum_confidence_in_bin.item()))    
            return bins
    
    
class AverageMeter():
    
    _available_metric = ['loss', 'acc', 'l1', 'ece']
    
    def __init__(self, metric_list:List, name:str=None) -> None:
        
        self.name = name
        
        self.criterion_ce = torch.nn.CrossEntropyLoss(reduction='sum')
        self.criterion_l1 = torch.nn.L1Loss(reduction='sum')
        self.criterion_ece = ECELoss(reduction='sum')
        
        self.metric_list = metric_list
        
        self.metric_dict = {}
        
        for metric in metric_list: 
            assert metric in self._available_metric, f'metric {metric} is not defined !'
            self.metric_dict[f'{metric}_val'] = 0
            self.metric_dict[f'{metric}_count'] = 0
            self.metric_dict[f'{metric}_history'] = []
            
    def update(self, outputs:Union[torch.Tensor, torch.nn.Module]=None, labels:torch.Tensor=None, loss:torch.Tensor=None, eta_tilde:torch.Tensor=None):
        
        if not isinstance(outputs, torch.Tensor): # GP module
            outputs = outputs.sample(32).mean(1)
        
        if (outputs is not None) and (labels is not None):    
            self.metric_dict['loss_val'] = float(((self.metric_dict['loss_val']*self.metric_dict['loss_count'])+self.criterion_ce(outputs, labels))/(self.metric_dict['loss_count']+len(outputs)))
            self.metric_dict['acc_val']  = float(((self.metric_dict['acc_val']*self.metric_dict['acc_count'])+outputs.argmax(1).eq(labels).sum().item())/(self.metric_dict['acc_count']+len(outputs)))
            self.metric_dict['loss_count'] += len(outputs)
            self.metric_dict['acc_count']  += len(outputs)
            
            new_ece_outcome = self.criterion_ece(outputs, labels)
            if self.metric_dict['ece_val'] == 0:
                self.metric_dict['ece_val'] = [[(self.metric_dict['ece_val']+new_ece_outcome[i][j]) for j in range(2)] for i in range(len(new_ece_outcome))]
            else:
                self.metric_dict['ece_val'] = [[(self.metric_dict['ece_val'][i][j]+new_ece_outcome[i][j]) for j in range(2)] for i in range(len(new_ece_outcome))]
            self.metric_dict['ece_count'] += len(outputs)
            
        if (outputs is not None) and (eta_tilde is not None):
            confidence = torch.softmax(outputs, 1).max(1)[0].squeeze().detach().cpu()
            eta_tilde_max = eta_tilde.max(1)[0].squeeze()
            self.metric_dict['l1_val'] = float(((self.metric_dict['l1_val']*self.metric_dict['l1_count'])+self.criterion_l1(confidence, eta_tilde_max))/(self.metric_dict['l1_count']+len(outputs)))
            self.metric_dict['l1_count'] += len(outputs)
        
    def get(self, metric:str, ind:int=-1):
        return self.metric_dict[f'{metric}_history'][ind]
        
    def flush(self):
        for metric in self.metric_list: 
            assert metric in ('loss', 'acc', 'l1', 'ece'), f'metric {metric} is not defined !'

            if metric != 'ece':
                self.metric_dict[f'{metric}_history'].append(self.metric_dict[f'{metric}_val'])   
            else:
                ece = sum(abs(bin[0]-bin[1])/self.metric_dict['ece_count'] for bin in self.metric_dict['ece_val'])
                self.metric_dict[f'{metric}_history'].append(ece)
                
            self.metric_dict[f'{metric}_val'] = 0
            self.metric_dict[f'{metric}_count'] = 0
            
            
def clean_grad(model: torch.nn.Module) -> None:
    for params in model.parameters():
        if params.grad is not None:
            params.grad = None