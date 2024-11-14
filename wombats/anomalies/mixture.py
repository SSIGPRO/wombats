import numpy as np

from scipy import optimize
from scipy import linalg
from scipy import interpolate
from scipy import signal

from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

import matplotlib.pyplot as plt


def deviation_fully(delta_spectrum, delta_basis, Xok, SNRdB=None):

    spectrum = SpectralAlteration(delta_spectrum).fit(Xok, SNRdB)
    basis = PrincipalSubspaceAlteration(delta_basis).fit(Xok)
    
    loks_sqrt = np.sqrt(spectrum.loks)
    lokn_sqrt = np.sqrt(spectrum.lokn)
    
    # rotate sqrt of eighenvalues and build marix C
    lkos_sqrt = spectrum.Rb_theta @ loks_sqrt
    lko_sqrt = np.concatenate([lkos_sqrt, lokn_sqrt])
    
    C = basis.Rb_theta @ basis.Uok @ np.diag(lko_sqrt*np.sqrt(spectrum.lok)) @ basis.Uok.T
    
    energy_per_component = np.sum(spectrum.lok) / spectrum.n

    return 2 * energy_per_component - (2/spectrum.n) * np.trace(C)


def mix(mixture, deviation, Xok, SNRdB=None, verbose=True):
    # normalize anomaly weights
    anomalies = list(mixture.keys())
    weights = list(mixture.values())
    weights = weights/np.sum(weights)
    deltas = deviation * weights
    mixture = dict(zip(anomalies, deltas))
    
    Xko = Xok.copy()
    
    # apply first SpectralAlteration and then PrincipalSubspaceAlteration anomalies 
    if ('SpectralAlteration' in anomalies) and ('PrincipalSubspaceAlteration' in anomalies):
        
        # invert function relating imposed deviation (delta) and resulting deviation (deviation)
        delta_spectrum = mixture['SpectralAlteration']
        delta_basis = mixture['PrincipalSubspaceAlteration']
        deviation = delta_spectrum + delta_basis
        
        weight_spectrum = delta_spectrum / deviation
        weight_basis = delta_basis / deviation
        
        weight_max = max(weight_basis, weight_spectrum)
        
        deltas_list = np.linspace(0, 2/weight_max, 100)
        
        deviations = np.array([deviation_fully(weight_spectrum*delta, weight_basis*delta, Xok, SNRdB) for delta in deltas_list])        
        delta = np.interp(deviation, deviations, deltas_list)
        
        if verbose:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(deltas_list, deviations)
            ax.scatter(delta, deviation, marker='o', c='r')
            ax.set(xlabel='imposed $\delta$', ylabel='resulting $\delta$')
            ax.grid()
    
        spectrum = SpectralAlteration(weight_spectrum*delta).fit(Xok, SNRdB)
        basis = PrincipalSubspaceAlteration(weight_basis*delta).fit(Xok)

        Xko = spectrum.distort(Xok)
        Xko = basis.distort(Xko)
        
        mixture.pop('SpectralAlteration')
        mixture.pop('PrincipalSubspaceAlteration')
               
    elif ('SpectralAlteration' in anomalies):
        delta = mixture['SpectralAlteration']
        subspace = anomalies_dict['SpectralAlteration'](delta).fit(Xok, SNRdB)
        Xko = subspace.distort(Xko)
        mixture.pop('SpectralAlteration')
        
    elif 'PrincipalSubspaceAlteration' in anomalies:
        delta = mixture['PrincipalSubspaceAlteration']
        subspace = anomalies_dict['PrincipalSubspaceAlteration'](delta).fit(Xok)
        Xko = subspace.distort(Xko)
        mixture.pop('PrincipalSubspaceAlteration')
    
    # apply first Clipping and Dead-Zone anomalies   
    elif 'Clipping' in anomalies:
        delta = mixture['Clipping']
        decreasing = anomalies_dict['Clipping'](delta).fit(Xok)
        Xko = decreasing.distort(Xko)  
        mixture.pop('Clipping')
        
    elif 'Dead-Zone' in anomalies:
        delta = mixture['Dead-Zone']
        decreasing = anomalies_dict['Dead-Zone'](delta).fit(Xok)
        Xko = decreasing.distort(Xko)  
        mixture.pop('Dead-Zone')
        
    # apply increasing anomalies
    for (anomaly, delta) in mixture.items():  
        Xko = anomalies_dict[anomaly](delta).fit(Xok).distort(Xko)      
    return Xko
        
        