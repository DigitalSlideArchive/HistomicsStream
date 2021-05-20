# histomics-stream

Through version 1.0.6, this project was known as tensorflow_reader.

## Overview

The goal of this project is to create a whole-slide image file reader for TensorFlow. This reader will allow users to extract pixel data from whole-slide image formats, and will support reading paradigms that are commonly used during training and inference.

## Installation for Python

[![PyPI Version](https://img.shields.io/pypi/v/histomics-stream.svg)](https://pypi.python.org/pypi/histomics-stream)

histomics-stream can be easily installed with Python wheels.  If you do not want the installation to be to your current Python environment, you should first create and activate a [Python virtual environment (venv)](https://docs.python.org/3/tutorial/venv.html) to work in.  Then, run the following from the command-line:

```shell-script
pip install histomics-stream
```

Launch `python3`, import the histomics-stream package, and use it

```python
import histomics-stream as hs
```
