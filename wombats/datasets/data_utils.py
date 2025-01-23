import os
import pandas as pd
import numpy as np
from . import DATA_DIR

def load_data_training(channels, n, data_type=None, val_frac=0.1, augmentation=False, SNRdB=40):
    
    data_path_dic = {
                     'bridge': os.path.join(DATA_DIR, 'bridge', f'instances-sensor=1D30-period=2017-10-02 00:00:00-2017-10-08 23:59:59-ch=x_y_z-n={n}-fs=100-mode=channel-train.pkl'),
                     'ecg': os.path.join(DATA_DIR, 'ecg', f'ecgSyn_n={n}_scaled_train_snr={SNRdB}dB.pkl')
                    }

    X = pd.read_pickle(data_path_dic[data_type]).dropna(axis=0)
            
    if data_type == 'bridge':
        X = X[channels].values
        nch = len(channels)
        ninst = len(X)
        new_shape = (-1, nch, X.shape[1]//nch)
        X = X.reshape(new_shape)
        X = X.swapaxes(1, 2)
        
    elif data_type == 'ecg':
        nch = 1
        ninst = len(X)
        X = X.values.reshape((ninst, n, nch))
        
    indexes = np.random.permutation(ninst)
    
    X_train = X[indexes[:-int(ninst*val_frac)]]
    X_val = X[indexes[-int(ninst*val_frac):]]
    
    return X_train, X_val


def load_data_test(channels, n, data_type=None, SNRdB=40):
    data_path_dic = {
                     'bridge': os.path.join(DATA_DIR, 'bridge', f'instances-sensor=1D30-period=2017-10-09 00:00:00-2017-10-15 23:59:59-ch=x_y_z-n={n}-fs=100-mode=channel-test.pkl'),
                     'ecg': os.path.join(DATA_DIR, 'ecg', f'ecgSyn_n={n}_scaled_test_snr={SNRdB}dB.pkl')
                    }
    
    X = pd.read_pickle(data_path_dic[data_type]).dropna(axis=0)
    if data_type == 'bridge':
        X = X[channels].values
        nch = len(channels)
        new_shape = (-1, nch, X.shape[1]//nch)
        X = X.reshape(new_shape)
        X = X.swapaxes(1, 2)
        ninst = 10_000
        rng = np.random.default_rng(12345) #seed: 12345
        indexes = rng.integers(0, X.shape[0], size=ninst)
        X = X[indexes]
    elif data_type == 'ecg':
        nch = 1
        ninst = 10_000
        X = X.values[:ninst, 0:n].reshape((ninst, -1, nch))
    
    return X