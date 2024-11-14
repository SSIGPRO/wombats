import numpy as np
from scipy import signal


# def pacf(X, p=None, method='burg'):
#     if method


def burg_pacf(x, p=None):
    """
    Burg's algorithm for estimating the Partial AutoCorrelation
    Function (PACF) of a signal. (Section 5.1.2 in [1])

    [1] P. J. Brockwell, R. A. Davis. 2016.
        "Introduction to Time Series and Forecasting". Springer

    Parameters
    ----------
    x: 1D or 2D numpy.ndarray or iterable of 1D numpy.ndarray,
        signal chunks arranged as element of an iterable
    p: int, optional (default None)
        order of the PACF. p must be less than the length of the
        shortest signal chunk.

    Returns
    -------
    (p+1,) numpy.ndarray
        estimation of the PACF.
    (p+1,) numpy.ndarray
        error variance associated at each element of the PACF
    """
    if isinstance(x, np.ndarray):
        pacf, sigma2 = burg_pacf_ndarray(x, p=p)
    elif isinstance(x, (list, tuple)):
        pacf, sigma2 = burg_pacf_iterable(x, p=p)
    else:
        raise ValueError('type {type(x)} not supported for x')
    return pacf, sigma2


def burg_pacf_ndarray(x, p=None):
    """
    Burg's algorithm for estimating the Partial AutoCorrelation
    Function (PACF) of a signal. (Section 5.1.2 in [1])

    [1] P. J. Brockwell, R. A. Davis. 2016.
        "Introduction to Time Series and Forecasting". Springer

    Parameters
    ----------
    x: 1D or 2D numpy.ndarray or iterable of 1D numpy.ndarray,
        signal or signal chunks arranged as rows of a matrix
    p: int, optional (default None)
        order of the PACF. p must be less than the length of the
        signal or less than the shortest signal chunk.

    Returns
    -------
    (p+1,) numpy.ndarray
        estimation of the PACF.
    (p+1,) numpy.ndarray
        error variance associated at each element of the PACF
    """
    nchunks = 1 if x.ndim == 1 else x.shape[0]
    n = x.shape[-1]  # signal chunk dimension
    if p is None:
        p = n-1
    elif p >= n:
        raise ValueError('p must be less than signal dimension.')
    
    d = np.zeros(p+1)
    pacf = np.zeros(p+1)

    d[0] = 2*np.sum(x ** 2)
    pacf[0] = 1

    u = x.T[::-1].copy()
    v = x.T[::-1].copy()

    d[1] = np.sum(u[:-1]**2) + np.sum(v[1:]**2)
    pacf[1] = 2/d[1] * np.sum(u[:-1] * v[1:])

    for i in range(1, p):
        u_old, v_old = u.copy(), v.copy()
        u[1:] = u_old[:-1] - pacf[i] * v_old[1:]
        v[1:] = v_old[1:] - pacf[i] * u_old[:-1]
        d[i+1] = (1 - pacf[i]**2) * d[i] - np.sum(v[i]**2) - np.sum(u[-1]**2)
        pacf[i+1] = 2/d[i+1] * np.sum(u[i:-1] * v[i + 1:])
    sigma2 = (1 - pacf**2)*d / (2*nchunks*(n - np.arange(0, p+1)))
    
    return pacf, sigma2




def burg_pacf_iterable(x, p=None):
    """
    Burg's algorithm for estimating the Partial AutoCorrelation Function
    (PACF) of a signal. (Section 5.1.2 in [1])

    [1] P. J. Brockwell, R. A. Davis. 2016.
        "Introduction to Time Series and Forecasting". Springer

    Parameters
    ----------
    data: iterable of 1D numpy.ndarray,
        signal chunks arranged as elements of an iterable.
    p: int, optional (default None)
        order of the PACF. The value for p must be less than the length
        of the shortest signal chunk.

    Returns
    -------
    (p+1,) numpy.ndarray
        estimation of the PACF
    (p+1,) numpy.ndarray
        error variance associated at each element of the PACF
    """
    nchunks = len(x)
    nobs = np.array([len(chunk) for chunk in x])
    nobs_min = np.min(nobs)
    nobs_tot = np.sum(nobs)
    if p is None:
        p = nobs_min - 1
    elif p >= nobs_min:
        raise ValueError('p must be less than signal dimension.')

    d = np.zeros(p + 1)
    pacf = np.zeros(p + 1)

    d[0] = 2 * np.sum([np.sum(chunk**2) for chunk in x])
    pacf[0] = 1

    u_list = [chunk[::-1].copy() for chunk in x]
    v_list = [chunk[::-1].copy() for chunk in x]

    uv = 0
    for u, v in zip(u_list, v_list):
        d[1] += np.sum(u[:-1] ** 2) + np.sum(v[1:] ** 2)
        uv += np.sum(u[:-1] * v[1:])
    pacf[1] = 2 * uv / d[1]

    for i in range(1, p):
        u2v2, uv = 0, 0
        for j, (u, v) in enumerate(zip(u_list, v_list)):
            u_old, v_old = u.copy(), v.copy()
            u[1:] = u_old[:-1] - pacf[i] * v_old[1:]
            v[1:] = v_old[1:] - pacf[i] * u_old[:-1]
            u2v2 += v[i]**2 + u[-1]**2
            uv += np.sum(u[i:-1] * v[i+1:])
        d[i+1] = (1 - pacf[i]**2) * d[i] - u2v2
        pacf[i+1] = 2 * uv / d[i + 1]
    sigma2 = (1 - pacf**2) * d / (2 * (nobs_tot - nchunks*np.arange(0, p+1)))

    return pacf, sigma2
    

def durbin_levinson_acf(pacf):
    """
    Durbin-Levinson algorithm for estimating the AutoCorrelation Function
    (ACF) and the AutoRegressive (AR) parameters from the Partial
    AutoCorrelation Function. (Section 5.1.2 and 2.5.3 in [1])

    [1] P. J. Brockwell, R. A. Davis. 2016.
        "Introduction to Time Series and Forecasting". Springer

    Parameters
    ----------
    pacf: (n+1,) numpy.ndarray
        PACF from 0 to n

    Returns
    -------
    ar_params: (n,) numpy.ndarray
        AR coefficients computed from the PACF.
    acf : (n+1,) numpy.ndarray
        The ACF computed from the PACF
    """

    if pacf[0] != 1.:
        raise ValueError('pacf[0] must be 1.')
    pacf = pacf[1:]
    n = pacf.shape[0]

    acf = np.zeros(n+1)
    acf[1] = pacf[0]
    nu = np.cumprod(1 - pacf**2)
    ar_params = pacf.copy()
    for i in range(1, n):
        prev = ar_params[: -(n-i)].copy()
        ar_params[: -(n-i)] = prev - ar_params[i]*prev[::-1]
        acf[i+1] = ar_params[i]*nu[i-1] + prev@acf[1:-(n-i)][::-1]
    acf[0] = 1
    return ar_params, acf


def burg(x, ar_order=0):
    """
    Burg's method for AR coefficients estimation.
    (Section 5.1.2 in [1])

    [1] P. J. Brockwell, R. A. Davis. 2016.
        "Introduction to Time Series and Forecasting". Springer

    Parameters
    ----------
    x: 1D or 2D numpy.ndarray,
        signal chunks arranged as rows of a matrix
    ar_order: int, optional (default 0)
        order of the PACF.

    Returns
    -------
    numpy.array,
        parameters of the AR model
    """
    pacf, pacf_sigma2 = burg_pacf(x, p=ar_order)
    ar_params, acf = durbin_levinson_acf(pacf)
    
    return ar_params, acf, pacf, pacf_sigma2


class AR:
    """
    Autoregressive (AR) model

    Attributes
    ----------
    p: int,
        order of the AR model
    ar_params: (p,) numpy.ndarray,
        autoregressive parameters
    acf: (p,) numpy.ndarray,
        autocorrelation function
    pacf: (p,) numpy.ndarray,
        partial autocorrelation function
    sigma2: float,
        error variance

    Methods
    -------
    fit:
    predict:
    """

    def __init__(self, order, ar_params=None):
        self.p = order              # model order
        self.ar_params = ar_params  # autoregressive parameters
        self.acf = None             # autocorrelation function
        self.pacf = None            # partial autocorrelation function
        self.pacf_sigma2 = None     # pacf estimation error
        
    def fit(self, x, method='burg'):
        """
        Fit (estimate) parameters of the model from data.

        Parameters
        ----------
        x: numpy.ndarray,
            dataset
        method: str {'burg'}, optional (default 'burg')
            specify the method adopted to fit parameters to data.

        Returns
        -------
        AR
            autoregressive model
        """
        if method == 'burg':
            ar_params, acf, pacf, pacf_sigma2 = burg(x, ar_order=self.p)
            self.ar_params = ar_params
            self.acf = acf
            self.pacf = pacf
            self.pacf_sigma2 = pacf_sigma2
        
        # elif method == 'yule-walker':
        #     pass
        
        else:
            raise ValueError(f'method {method} not supported')
    
        return self
        
    def predict(self, x):
        """
        1-step prediction

        Parameters
        ----------
        x: 1D or 2D numpy.ndarray,
            dataset

        Returns
        -------
        x.shape numpy.array,
            predicted values
        """
        # TODO: add n-step prediction functionality
        y = signal.lfilter(self.ar_params, 1, x)
        y[..., :self.p-1] = np.nan

        return y
        
        
    