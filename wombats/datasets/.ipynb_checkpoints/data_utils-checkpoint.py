import os
import sys
import pandas as pd
import numpy as np


def load_data_training(channels, n, data_type=None, val_frac=0.1, augmentation=False, SNRdB=40):
    
    data_path_dic = {
                     'bridge': os.path.join('..', '..', '..',
                                            'bridge-data',
                                           f'instances-sensor=1D30-period=2017-10-02 00:00:00-2017-10-08 23:59:59-ch=x_y_z-n={n}-fs=100-mode=channel-train.pkl'),
                     'ecg': os.path.join('..', '..', '..',
                                         'ecg-data', f'ecgSyn_n={n}_scaled_train_snr={SNRdB}dB.pkl'),
                     # 'ecg_ar': os.path.join('..', 'ecg-data', f'ar_n={n}_ecgSyn_n512_Ny256_(HR 80 100)_scaled_train.pkl')
                    }

    X = pd.read_pickle(data_path_dic[data_type]).dropna(axis=0)
    if data_type == 'satellite':
        X = X[X['AOC_mode'] == 1282]
    
        if augmentation:
            with mp.Pool(processes=mp.cpu_count()//2) as pool:
                X = pool.starmap(
                    window_shifting, 
                    [(X[channels], n, shift) for shift in range(0, n, 16)]
                )
            X = np.concatenate(X)
            ninst = len(X)

        else:
            X = X[channels].values
            nch = len(channels)
            ninst = len(X) // (nch * n)
            X = X[:ninst * nch * n]
            X = X.reshape(ninst,nch,n).swapaxes(1, 2)
            
    elif data_type in ['rfi', 'bridge']:
        X = X[channels].values
        nch = len(channels)
        ninst = len(X)
        new_shape = (-1, nch, X.shape[1]//nch)
        X = X.reshape(new_shape)
        X = X.swapaxes(1, 2)
        
    elif 'ecg' in data_type:
        # SNRdB = 40
        # nch = 1
        # if 'ar' in data_type:
        #     ninst = len(X)
        # else:
        #     ninst = len(X) * X.shape[-1] // n
        # X = X.values.reshape((ninst, n, nch))
        # X = X + np.random.randn(*X.shape)*(10**(-SNRdB/20))
        nch = 1
        ninst = len(X)
        X = X.values.reshape((ninst, n, nch))
        
    indexes = np.random.permutation(ninst)
    
    X_train = X[indexes[:-int(ninst*val_frac)]]
    X_val = X[indexes[-int(ninst*val_frac):]]
    
    return X_train, X_val


def load_data_test(channels, n, data_type=None, data_path=None, augmentation=False, SNRdB=40):
    data_path_dic = {
                     'bridge': os.path.join('..', '..', '..',
                                            'bridge-data',
                                            f'instances-sensor=1D30-period=2017-10-09 00:00:00-2017-10-15 23:59:59-ch=x_y_z-n={n}-fs=100-mode=channel-test.pkl'),
                     'ecg': os.path.join('..', '..', '..',
                                         'ecg-data',
                                         f'ecgSyn_n={n}_scaled_test_snr={SNRdB}dB.pkl')
                    }
    
    if data_path is None:
        data_path = data_path_dic[data_type]
    X = pd.read_pickle(data_path).dropna(axis=0)
    if data_type == 'satellite':
        X = X[X['AOC_mode'] == 1282]
        X = X[ : '2019-04-15']
    
        if augmentation:
            with mp.Pool(processes=mp.cpu_count()//2) as pool:
                X = pool.starmap(
                    window_shifting, 
                    [(xdf, n, shift) for shift in range(0, n, 16)]
                )
            X = np.concatenate(X)
            ninst = len(X)

        else:
            X = X[channels].values
            nch = len(channels)
            # ninst = len(X) // (nch * n)
            ninst = 10_000
            X = X[:ninst * nch * n]
            X = X.reshape(ninst,nch,n).swapaxes(1, 2)
    elif data_type in ['rfi', 'bridge']:
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
        # SNRdB = 40
        # nch = 1
        # # ninst = len(X) * X.shape[-1] // n
        # # X = X.values.reshape((ninst, n, nch))
        # ninst = 10_000
        # X = X.values[:ninst, 0:n].reshape((ninst, -1, nch))
        # # indexes = np.random.permutation(ninst)
        # # X = X[indexes]
        # # np.random.seed(0)
        # X = X + np.random.randn(*X.shape)*10**(-SNRdB/20)
        nch = 1
        ninst = 10_000
        X = X.values[:ninst, 0:n].reshape((ninst, -1, nch))
    
    return X