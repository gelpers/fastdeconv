# fastdeconv

This package implements the non-blind deconvolution algorithm of Krishnan & Fergus as described in the paper:

[D. Krishnan, R. Fergus. Fast Image Deconvolution using Hyper-Laplacian Priors. Advances in Neural Information Processing Systems 22 (NIPS 2009).](https://papers.nips.cc/paper/3707-fast-image-deconvolution-using-hyper-laplacian-priors)

![lena.png](../assets/lena.png)

## Dependencies

* `numpy`
* `scipy`

To run the examples, you'll also need the following extra dependencies:

* `matplotlib`
* `imageio`

## Installation

```sh
pip install git+git://github.com/gelpers/fastdeconv.git
```

To clone the repository and run the examples:

```sh
git clone git@github.com:gelpers/fastdeconv
cd fastdeconv
python setup.py install
pip install matplotlib imageio  # additional deps used by examples

python example/deblur.py
```

## Usage

```py
from imageio import imread
import matplotlib.pyplot as plt

from fastdeconv import deconv

# Load the corrupted image and blur kernel
y = imread('corrupted.png', as_gray=True)
k = imread('kernel.png', as_gray=True)

# Perform the deconvolution assuming alpha=2/3
lam = 7e4
x = deconv(y, k, lam)

# Plot the reconstructed image
plt.imshow(x)
plt.show()
```

## Limitations

Currently only `alpha=2/3` is supported.
