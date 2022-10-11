#!/usr/bin/env python

import torch
from torch.nn import functional as F
from torch.utils.data import  DataLoader
import numpy as np
import copy

def inject_noise(dataloader: torch.utils.data.DataLoader, 
                 f_star: torch.nn.Module, 
                 noise_type: str, 
                 noise_strength: float, 
                 mode: str = 'train') -> torch.utils.data.DataLoader:

    # use default random sampler 
    tempdataloader = DataLoader(dataloader.dataset, batch_size=128)
    eta_temp_pair = [(torch.softmax(f_star(images.to(f_star.device)), 1).detach().cpu(), indices) for _, (indices, images, _, _) in enumerate(tempdataloader)] 
    eta_temp, eta_indices = torch.cat([x[0] for x in eta_temp_pair]), torch.cat([x[1] for x in eta_temp_pair]).squeeze()
    
    # ground truth eta
    eta = eta_temp[eta_indices.argsort()]
    f_star_outs = eta.argmax(1).squeeze()       

    # inject noise and get eta_tilde
    if noise_type=='rcn':
        y_tilde, P, _ = noisify_with_P(np.array(copy.deepcopy(f_star_outs)), nb_classes=10, noise=noise_strength)
        eta_tilde = np.matmul(F.one_hot(f_star_outs, num_classes=10), P)
    elif noise_type=='linear':
        eta_tilde = perturb_eta(eta, noise_type, noise_strength)
        y_tilde = [int(np.where(np.random.multinomial(1, np.clip(x, 1e-3, 0.999)/np.clip(x, 1e-3, 0.999).sum(), 1).squeeze())[0]) for x in eta_tilde]
    elif noise_type=='idl':
        y_tilde = [int(np.where(np.random.multinomial(1, np.clip(x, 1e-3, 0.999)/np.clip(x, 1e-3, 0.999).sum(), 1).squeeze())[0]) for x in np.array(eta)]
        eta_tilde = copy.deepcopy(eta)
    else:
        raise NotImplementedError(f"Expecte the noise type to be in (rcn, idl), but got {noise_type}!")
    dataloader.dataset.update_eta(eta_tilde)
    
    # corrupt labels
    if mode == 'train':
        dataloader.dataset.update_labels(y_tilde)
    elif mode == 'eval':
        dataloader.dataset.update_labels(f_star_outs)
    else:
        raise NotImplementedError(f"Expecte the mode to be in (train, eval), but got {mode}!")

    return dataloader


def perturb_eta(eta: torch.tensor, noise_type: str, noise_strength: float):
    eta_tilde = copy.deepcopy(eta)
    if noise_type == 'linear':
        for i in range(len(eta)):
            eta_i = eta[i].squeeze()
            max_ind, sec_ind = eta_i.argsort(descending=True)[:2]
            delta_eta = eta_i[max_ind] - eta_i[sec_ind]
            # Always preserve Bayess optimal prediction (Our assumption)
            eta_i[max_ind] = eta_i[max_ind] - noise_strength*delta_eta
            eta_i[sec_ind] = eta_i[sec_ind] + noise_strength*delta_eta
            eta_tilde[i] = eta_i
    else:
        for i in range(len(eta)):
            eta_i = eta[i].squeeze()
            max_ind = eta_i.argmax()[0]
            dirich_samp = np.random.dirichlet(np.ones(len(eta_i)), 1)
            samp_max_ind = dirich_samp.argmax()[0]
            # Always preserve Bayess optimal prediction (Our assumption)
            eta_i[max_ind], eta_i[samp_max_ind] = eta_i[samp_max_ind], eta_i[max_ind]
            eta_tilde[i] = eta_i
    return eta_tilde


def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = noise / (size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (1 - noise) * np.ones(size))

    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify_with_P(y_train, nb_classes, noise, random_state=None):

    if noise > 0.0:
        P = build_uniform_P(nb_classes, noise)
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)
        keep_indices = np.arange(len(y_train))

    return y_train, P, keep_indices


def noisify_mnist_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        1 <- 7
        2 -> 7
        3 -> 8
        5 <-> 6
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise
    keep_indices = np.arange(len(y_train))

    if n > 0.0:
        # 1 <- 7
        P[7, 7], P[7, 1] = 1. - n, n
        # 2 -> 7
        P[2, 2], P[2, 7] = 1. - n, n
        # 5 <-> 6
        P[5, 5], P[5, 6] = 1. - n, n
        P[6, 6], P[6, 5] = 1. - n, n
        # 3 -> 8
        P[3, 3], P[3, 8] = 1. - n, n
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices


def noisify_cifar10_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    keep_indices = np.arange(len(y_train))
    n = noise

    if n > 0.0:
        # automobile <- truck
        P[9, 9], P[9, 1] = 1. - n, n

        # bird -> airplane
        P[2, 2], P[2, 0] = 1. - n, n

        # cat <-> dog
        P[3, 3], P[3, 5] = 1. - n, n
        P[5, 5], P[5, 3] = 1. - n, n

        # automobile -> truck
        P[4, 4], P[4, 7] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices
