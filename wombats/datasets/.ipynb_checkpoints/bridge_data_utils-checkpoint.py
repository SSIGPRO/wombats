import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.signal import argrelextrema

def energy_threshold(data, bandwidth=0.000005, N=2000, verbose=False):
    """
    Computes the energy threshold between signal and noise as the 
    first local minima of KDE 
    """

    energy = np.sum(data**2, axis=1)
    inv_energy = 1/energy
    # model definition and fitting
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(inv_energy[:,None])
    # model test
    inv_energy_test = np.linspace(min(inv_energy) , max(inv_energy), N)
    log_dens = kde.score_samples(inv_energy_test[:, None]) # resitituisce densità in scala log_e
    # min computing
    loc_min = argrelextrema(log_dens, np.less)
    inv_threshold = inv_energy_test[loc_min]
    threshold = 1/inv_threshold
    
    if verbose:
        fig, ax = plt.subplots(figsize = (8,4))
        ax.plot(inv_energy_test, np.exp(log_dens))
        ax.axvline(inv_threshold[0], c='k')
        ax.set(yscale='log',
               title='KDE of the inverse energy')
        ax.legend()

    return threshold[0]

def standardize(data, scaler=None, mode='all'):
    shape = data.shape
    if mode == 'channel':
        data = data.reshape(-1, shape[-1])
    elif mode == 'all':
        data = data.flatten()[:, np.newaxis] 
    if scaler is None:   
        scaler = StandardScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    
    if mode == 'channel':
        data = data.reshape(-1, shape[-2], shape[-1])
    elif mode == 'all':
        data = data.reshape(shape)
    return data, scaler

def multi_to_single_ch(X):
    X = X.swapaxes(1, 2)
    shape = X.shape
    X = X.reshape(shape[0], shape[1]*shape[2])
    return X

def preprocess(data, n, fs, threshold=None, scaler=None, mode='all', verbose=False):
    tw = n*pd.Timedelta(1/fs, 's')
    axes = data.columns.values
    data = data.groupby(data.index.floor(tw)).filter(lambda x: len(x) == n)
    # reshape
    data = data.values.reshape(-1, n, len(axes))
    data = multi_to_single_ch(data)
    # filter energy
    energy = np.sum(data**2, axis=1)
    if threshold is None:
        threshold = energy_threshold(data, verbose=verbose)   
    data = data[energy > threshold]
    data = data.reshape(-1, len(axes), n).swapaxes(1, 2)
    # # standardize
    # if std is None:
    #     std = np.std(data)
    # if mean is None:
    #     mean = np.mean(data)
    # data = data/std - mean
    data, scaler = standardize(data, scaler, mode)
    data = multi_to_single_ch(data)
    columns = pd.MultiIndex.from_product([axes, np.arange(n)])
    data = pd.DataFrame(data, columns=columns, dtype=np.float64)
    return data, threshold, scaler