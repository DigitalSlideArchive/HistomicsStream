# histomics_stream

## Overview

The goal of this project is to create a whole-slide image file reader for machine learning with TensorFlow. This reader will allow users to extract pixel data from whole-slide image formats, and will support reading paradigms that are commonly used during training and inference.

## Installation for Python

[![PyPI Version](https://img.shields.io/pypi/v/histomics_stream.svg)](https://pypi.python.org/pypi/histomics_stream)
[![GitHub repository](https://img.shields.io/badge/Powered%20by-histomics__stream-blue.svg)](https://github.com/DigitalSlideArchive/histomics_stream)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DigitalSlideArchive/histomics_stream/blob/master/example/tensorflow_stream.ipynb?authuser=1)

histomics_stream can be easily installed with Python wheels.  If you do not want the installation to be to your current Python environment, you should first create and activate a [Python virtual environment (venv)](https://docs.python.org/3/tutorial/venv.html) to work in.  Then, run the following from the command line:

```shell-script
apt update
apt install -y python3-openslide openslide-tools
pip uninstall -y histomics_stream large_image tensorflow
pip install histomics_stream 'large_image[openslide,ometiff,openjpeg,bioformats]' --find-links https://girder.github.io/large_image_wheels
```

Launch `python3`, import the histomics_stream package, and use it

```python
import histomics_stream as hs
```

This has been tested with `tensorflow:2.6.2-gpu`.

## History

Through version 1.0.6, this project was known as tensorflow_reader.
