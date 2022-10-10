#!/usr/bin/env python

import os

import yaml
import pickle as pkl
import argparse

from environ import Environ


if __name__  == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Configuration file path', default='./experiment_configuration.yml')
    parser.add_argument('--dataset', type=str, help='Experiment dataset', choices={'mnist', 'cifar10'})
    parser.add_argument('--noise_type', type=str, help='Label noise type', choices={'rcn', 'idl'})
    parser.add_argument('--noise_strength', type=str, help='Label noise strength')
    parser.add_argument('--method', type=str, help='Calibration method to choose. Use + between different method. options=(raw, ts, mcdrop, cskd, ensemble, focal, bm, lula, gp, ours)')
    parser.add_argument('--result_dir', type=str, help='Results save dir', default='./result')
    parser.add_argument('--checkpoint_dir', type=str, help='Training checkpoint save dir', default='./checkpoint')
    parser.add_argument('--fig_dir', type=str, help='figures save dir', default='./figures')
    parser.add_argument('--gpu', type=str, help='Device id of the GPU to be used', default='0')
    parser.add_argument('--seed', type=str, help='Random seed of the experiments')
    args = parser.parse_args()
    
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    f.close()
    config['args'] = {}
    for k, v in args._get_kwargs():
        config['args'][k] = v
    
    envconfig = Environ(config=config)
    env = envconfig.create_environ()
    dataset = envconfig.create_dataset()
    network = envconfig.create_network()
    trainer = envconfig.create_trainer()
    calibrators = envconfig.create_calibrator(databuilder=dataset)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = env['gpu']
    
    os.makedirs(env['checkpoint_dir'], exist_ok=True)
    model = trainer.train(
        model = network.model,
        trainloader = dataset.train_loader, 
        validloader = dataset.valid_loader, 
        testloader  = dataset.test_loader, 
        calibrators = calibrators
    )
    
    result_dict = trainer.eval( 
        model = model, 
        calibrators = calibrators, 
        testloader  = dataset.test_loader, 
        logger = trainer.metric_test
    )
    
    # Save result
    os.makedirs(env['result_dir'], exist_ok=True)
    save_filepath = os.path.join(env['result_dir'], f"{args.dataset}_{args.method}_{args.noise_type}_{args.noise_strength}_{args.seed}_{env['timestamp']}.pkl")
    with open(save_filepath, 'wb') as f:
        pkl.dump(config, f)
        pkl.dump(result_dict, f)
    f.close()
    
    
    
    
    
    