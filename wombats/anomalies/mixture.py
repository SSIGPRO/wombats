import numpy as np

from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

import matplotlib.pyplot as plt


def deviation_fully(delta_spectrum: float, delta_basis: float, Xok: np.ndarray, SNRdB: float = None) -> float:
    """
    Computes the deviation based on the spectral and principal subspace alterations.

    :param delta_spectrum: The deviation parameter for spectral alteration.
    :param delta_basis: The deviation parameter for principal subspace alteration.
    :param Xok: A 2D array containing normal data (N, n).
    :param SNRdB: Signal-to-noise ratio in decibels (optional).

    :return: A scalar value representing the deviation.
    """
    # Apply spectral and basis alterations
    spectrum = SpectralAlteration(delta_spectrum).fit(Xok, SNRdB)
    basis = PrincipalSubspaceAlteration(delta_basis).fit(Xok)
    
    loks_sqrt = np.sqrt(spectrum.loks)
    lokn_sqrt = np.sqrt(spectrum.lokn)
    
    # Rotate sqrt of eighenvalues and build marix C
    lkos_sqrt = spectrum.Rb_theta @ loks_sqrt
    lko_sqrt = np.concatenate([lkos_sqrt, lokn_sqrt])
    
    C = basis.Rb_theta @ basis.Uok @ np.diag(lko_sqrt*np.sqrt(spectrum.lok)) @ basis.Uok.T
    
    energy_per_component = np.sum(spectrum.lok) / spectrum.n

    return 2 * energy_per_component - (2/spectrum.n) * np.trace(C)


def mix(mixture: dict, deviation: float, Xok: np.ndarray, SNRdB: float = None, verbose: bool = True) -> np.ndarray:
    """
    Applies various anomaly distortions based on a mixture of different anomaly types.

    :param mixture: A dictionary where keys are anomaly types (e.g., 'SpectralAlteration') and values are their corresponding deviations.
    :param deviation: The overall deviation to apply.
    :param Xok: A 2D array containing normal data (N, n).
    :param SNRdB: Signal-to-noise ratio in decibels (optional).
    :param verbose: If True, plots the deviation comparison.

    :return: The distorted data after applying all anomalies.
    """
    # Normalize anomaly weights
    anomalies = list(mixture.keys())
    weights = list(mixture.values())
    weights = weights/np.sum(weights)
    deltas = deviation * weights
    mixture = dict(zip(anomalies, deltas))
    
    Xko = Xok.copy()
    
    # Apply first SpectralAlteration and then PrincipalSubspaceAlteration anomalies 
    if ('SpectralAlteration' in anomalies) and ('PrincipalSubspaceAlteration' in anomalies):
        
        # Invert function relating imposed deviation (delta) and resulting deviation (deviation)
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
        subspace = SpectralAlteration(delta).fit(Xok, SNRdB)
        Xko = subspace.distort(Xko)
        mixture.pop('SpectralAlteration')
        
    elif 'PrincipalSubspaceAlteration' in anomalies:
        delta = mixture['PrincipalSubspaceAlteration']
        subspace = PrincipalSubspaceAlteration(delta).fit(Xok)
        Xko = subspace.distort(Xko)
        mixture.pop('PrincipalSubspaceAlteration')
    
    # Apply first Clipping and Dead-Zone anomalies   
    elif 'Clipping' in anomalies:
        delta = mixture['Clipping']
        decreasing = Clipping(delta).fit(Xok)
        Xko = decreasing.distort(Xko)  
        mixture.pop('Clipping')
        
    elif 'Dead-Zone' in anomalies:
        delta = mixture['Dead-Zone']
        decreasing = DeadZone(delta).fit(Xok)
        Xko = decreasing.distort(Xko)  
        mixture.pop('Dead-Zone')
        
    # Apply increasing anomalies
    increasing_anomalies ={
        'GWN': GWN,
        'Impulse': Impulse,
        'Step': Step,
        'Constant': Constant,  
        'GNN': GNN
    }
    for (anomaly, delta) in mixture.items():
        
        Xko = increasing_anomalies[anomaly](delta).fit(Xok).distort(Xko)      
    return Xko
        
        