import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastdeconv-gelpers",
    version="0.0.1",
    author="Gabriel Elpers",
    author_email="mrshrdlu@gmail.com",
    description="Fast image deconvolution using hyper-Laplacian priors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gelpers/fastdeconv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "scipy",
    ],
)
