import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d


def deconv(y: np.ndarray, k: np.ndarray, lam: float,
           beta=1, beta_rate=2*np.sqrt(2), beta_max=256) -> np.ndarray:
    """Perform non-blind deconvolution of a corrupted input image using
    the given blur kernel. It is assumed that the exponent alpha=2/3.

    Parameters
    ----------
    y : np.ndarray
        Corrupted input image

    k : np.ndarray
        Blur kernel for non-blind deconvolution

    lam : float
        Regularization parameter that trades off closeness to input image
        vs. closeness to assumed gradient statistics
        small => preserve L2 closeness to input image
        large => preserve assumed hyper-Laplacian gradient statistics

    beta : float
        Initial value of beta parameter for half-quadratic splitting iteration

    beta_rate : float
        Multiplicative factor by which to increase beta each iteration

    beta_max : float
        Terminate iteration when beta exceeds this value

    Returns
    -------
    np.ndarray
        Deconvolved image
    """

    alpha = 2/3  # TODO: allow any value of alpha
    nomin1, denom1, denom2 = precompute_fft_terms(y, k)

    k = k / np.sum(k)
    xr = y

    while beta < beta_max:
        v1 = circ_diff_1(xr)
        v2 = circ_diff_2(xr)

        w1 = compute_w(v1, alpha, beta)
        w2 = compute_w(v2, alpha, beta)

        xr = compute_x(nomin1, denom1, denom2, w1, w2, lam, beta)

        # translate to compensate for off-center kernel
        xr = np.roll(xr, (k.shape[0]//2-1, k.shape[1]//2-1), axis=(0, 1))

        beta *= beta_rate

    return xr


def pad_to_shape(a: np.ndarray, shape: tuple) -> np.ndarray:
    a = np.array(a)
    b = np.zeros(shape)
    ny, nx = a.shape
    b[:ny, :nx] = a
    return b


def circ_diff_1(a: np.ndarray) -> np.ndarray:
    # return np.diff(np.hstack([a, a[:, 0, np.newaxis]]), axis=1)
    return np.hstack([np.diff(a, axis=1), a[:, 0, np.newaxis] - a[:, -1, np.newaxis]])


def circ_diff_2(a: np.ndarray) -> np.ndarray:
    # return np.diff(np.vstack([a, a[0, :]]), axis=0)
    return np.vstack([np.diff(a, axis=0), a[0, :] - a[-1, :]])


def precompute_fft_terms(y: np.ndarray, k: np.ndarray) -> tuple:
    kp = pad_to_shape(k, y.shape)
    FK = fft2(kp)
    nomin1 = np.conj(FK) * fft2(y)
    denom1 = np.abs(FK)**2
    FF1 = fft2(pad_to_shape([[1, -1]], y.shape))
    FF2 = fft2(pad_to_shape([[1], [-1]], y.shape))
    denom2 = np.abs(FF1)**2 + np.abs(FF2)**2
    return (nomin1, denom1, denom2)


def compute_x(nomin1: np.ndarray, denom1: np.ndarray, denom2: np.ndarray,
              w1: np.ndarray, w2: np.ndarray, lam: float, beta: float) -> np.ndarray:
    gamma = beta / lam
    denom = denom1 + gamma * denom2
    w11 = -circ_diff_1(w1)
    w22 = -circ_diff_2(w2)
    nomin2 = w11 + w22
    nomin = nomin1 + gamma * fft2(nomin2)
    xr = np.real(ifft2(nomin / denom))
    return xr


def compute_w(v: np.ndarray, alpha: np.ndarray, beta: float):
    # TODO: extend this function to handle any value of alpha
    if np.allclose(alpha, 2/3):
        return compute_w23(v, beta)
    raise ValueError('only alpha=2/3 is currently supported')


def compute_w23(v: np.ndarray, beta: float):
    # direct analytic solution when alpha=2/3
    # see Algorithm 3 in the source paper
    eps = 1e-6
    m = 8/(27*beta**3)
    t1 = -9/8*v**2
    t2 = v**3/4
    t3 = -1/8*m*v**2
    t4 = -t3/2 + np.sqrt(0j - m**3/27 + (m*v**2)**2/256)
    t5 = np.power(t4, 1/3)
    t6 = 2*(-5/18*t1 + t5 + m/(3*t5))
    t7 = np.sqrt(t1/3 + t6)
    r1 = 3*v/4 + np.sqrt(t7 + np.sqrt(0j-(t1+t6+t2/t7)))/2
    r2 = 3*v/4 + np.sqrt(t7 - np.sqrt(0j-(t1+t6+t2/t7)))/2
    r3 = 3*v/4 + np.sqrt(-t7 + np.sqrt(0j-(t1+t6-t2/t7)))/2
    r4 = 3*v/4 + np.sqrt(-t7 - np.sqrt(0j-(t1+t6-t2/t7)))/2
    r = [r1, r2, r3, r4]
    c1 = np.abs(np.imag(r)) < eps
    c2 = np.real(r)*np.sign(v) > np.abs(v)/2
    c3 = np.real(r)*np.sign(v) < np.abs(v)
    wstar = np.max((c1 & c2 & c3) * np.real(r)*np.sign(v), axis=0)*np.sign(v)
    return wstar

# TODO: implement analytic solution to w-subproblem for alpha=1/2
# def compute_w12(v, beta):
#     # direct analytic solution when alpha=1/2
#     # see Algorithm 2 in the source paper
#     eps = 1e-6
#     m = -np.sign(v)/(4*beta**2)
#     t1 = 2*v/3
#     t2 = np.power(0j-27*m-2*v**3+3*np.sqrt(0j+3*(27*m**2+4*m*v**3)), 1/3)
#     t2[np.abs(t2) < eps] = eps  # prevent RuntimeWarning: invalid value
#     t3 = v**2/t2
#     r1 = t1 + 1/(3*2**(1/3))*t2 + (2**(1/3))/3*t3
#     r2 = t1 - (1 - np.sqrt(3)*1j) / (6*2**(1/3)) * t2 - (1 + np.sqrt(3)*1j) / (3*2**(2/3)) * t3
#     r3 = t1 - (1 + np.sqrt(3)*1j) / (6*2**(1/3)) * t2 - (1 - np.sqrt(3)*1j) / (3*2**(2/3)) * t3
#     r = [r1, r2, r3]
#     c1 = np.abs(np.imag(r)) < eps
#     c2 = np.real(r)*np.sign(v) > (2/3)*np.abs(v)
#     c3 = np.real(r)*np.sign(v) < np.abs(v)
#     wstar = np.max((c1 & c2 & c3) * np.real(r)*np.sign(v), axis=0)*np.sign(v)
#     return wstar
