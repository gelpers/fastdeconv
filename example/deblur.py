import numpy as np
import numpy.random
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from fastdeconv import deconv


def snr(y, x):
    signal = np.var(x)
    noise = np.mean((y-x)**2)
    return 10*np.log10(signal / noise)


def ascent():
    # Example image: ascent
    from scipy.misc import ascent
    return ascent()


def lena():
    # Example image: lena
    import os
    from imageio import imread
    filename = os.path.join(os.path.dirname(__file__), 'lena.png')
    return imread(filename, as_gray=True)


def gaussian_blur_kernel(shape):
    # Example kernel simulating Gaussian blur
    ny, nx = shape
    tx = np.linspace(-10, 10, nx)
    ty = np.linspace(-10, 10, ny)
    kx = np.exp(-0.1 * tx**2)
    ky = np.exp(-0.1 * ty**2)
    k = kx[np.newaxis, :] * ky[:, np.newaxis]
    k /= np.sum(k)
    return k


def streak_kernel(shape):
    # Example kernel simulating horizontal motion blur
    ny, nx = shape
    k = np.zeros(shape)
    k[ny//2-1, :] = 0.5
    k[ny//2+0, :] = 1.
    k[ny//2+1, :] = 0.5
    k[:, 0] *= 0.
    k[:, 1] *= 0.5
    k[:, -2] *= 0.5
    k[:, -1] *= 0.
    k /= np.sum(k)
    return k


def plot_results(x, k, y, xr):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) \
        = plt.subplots(2, 3, figsize=(12, 8))
    plt.gray()
    plt.tight_layout()

    ax1.set_title('original')
    ax1.imshow(x)

    ax2.set_title('corrupted')
    ax2.imshow(y)
    ax2.text(0, 0, f'SNR={snr(y, x):.2f}', horizontalalignment='left',
             verticalalignment='top', bbox=dict(facecolor='white'))
    inset = inset_axes(ax2,
                       width=f'{4*100*k.shape[1]/y.shape[1]}%',
                       height=f'{4*100*k.shape[0]/y.shape[0]}%',
                       loc=3)
    inset.tick_params(labelleft=False, labelbottom=False)
    inset.imshow(k)

    ax3.set_title('reconstructed')
    ax3.imshow(xr)
    ax3.text(0, 0, f'SNR={snr(xr, x):.2f}', horizontalalignment='left',
             verticalalignment='top', bbox=dict(facecolor='white'))

    alpha = 2/3
    bins = np.linspace(-50, 50, 200)

    ax4.set_yscale('log')
    ax4.set_ylim((1e0, 2e5))
    ax4.hist(np.diff(x).ravel(), bins=bins)
    ax4.plot(bins, 1e5*np.exp(-np.abs(bins)**alpha))
    ax4.set_xlabel('gradient')
    ax4.set_ylabel('count')

    ax5.set_yscale('log')
    ax5.set_ylim((1e0, 2e5))
    ax5.hist(np.diff(y).ravel(), bins=bins)
    ax5.plot(bins, 1e5*np.exp(-np.abs(bins)**alpha))

    ax6.set_yscale('log')
    ax6.set_ylim((1e0, 2e5))
    ax6.hist(np.diff(xr).ravel(), bins=bins)
    ax6.plot(bins, 1e5*np.exp(-np.abs(bins)**alpha))

    plt.show()


if __name__ == "__main__":
    numpy.random.seed(0)  # for reproducibility

    # Original input image
    x = lena()

    # Blur the original image
    k = streak_kernel((27, 27))
    y = convolve2d(x, k, mode='same', boundary='wrap')
    # Add Gaussian noise
    y += numpy.random.normal(scale=1e-3*np.median(y), size=y.shape)

    # Slightly corrupt the kernel to simulate imperfect kernel estimation
    k += numpy.random.exponential(scale=1e-4)
    k /= np.sum(k)

    # Perform deconvolution to reconstruct the original image
    lam = 3e5  # regularization parameter, tuned to maximize SNR
    xr = deconv(y, k, lam, beta_max=256)
    plot_results(x, k, y, xr)
