#!/usr/bin/env python

import os

import yaml
import argparse

from environ import ENVIRON


if __name__  == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Configuration file path', default='./experiment_configuration.yml')
    parser.add_argument('--dataset', type=str, help='Experiment dataset', choices={'mnist', 'cifar10'})
    parser.add_argument('--noise_type', type=str, help='Label noise type', choices={'rcn', 'idl'})
    parser.add_argument('--noise_strength', type=str, help='Label noise strength')
    parser.add_argument('--method', type=str, help='Calibration algorithm to choose', choice={'ts', 'mcdrop', 'cskd', 'ensemble', 'focal', 'bm', 'lula', 'gp', 'ours'})
    parser.add_argument('--result_dir', type=str, help='Results save dir', default='./result')
    parser.add_argument('--checkpoint_dir', type=str, help='Training checkpoint save dir', default='./checkpoint')
    parser.add_argument('--fig_dir', type=str, help='figures save dir', default='./figures')
    parser.add_argument('--gpu', type=str, help='Device id of the GPU to be used', default='0')
    parser.add_argument('--seed', type=str, help='Random seed of the experiments')
    args = parser.parse_args()
    
    with open(args.checkpoint_dir) as f:
        config = yaml.safe_load(f)
    f.close()
    config['args'] = {}
    for k, v in args._get_kwargs():
        config['args'][k] = v
    
    envconfig = ENVIRON(config=config)
    env = envconfig.create_environ()
    dataset = envconfig.create_dataset()
    network = envconfig.create_network()
    trainer = envconfig.create_trainer()
    calibrator = envconfig.create_calibrator()
    
    
    # trainer.train(
    #     trainloader = dataset.trainloader, 
    #     validloader = dataset.validloader, 
    #     testloader  = dataset.testloader, 
    #     calibrator = calibrator, 
    #     calibrateloader = dataset.calibrateloader, 
    # )
    
    # result_dict = trainer.eval(
    #     testloader = dataset.testloader
    # )
    
    # save result then 
    
    
    
    
    
    