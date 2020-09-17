# tensorflow_reader

## Overview

The goal of this project is to create a whole-slide image file reader for TensorFlow. This reader will allow users to extract pixel data from whole-slide image formats, and will support reading paradigms that are commonly used during training and inference.

## Installation for Python

[![PyPI Version](https://img.shields.io/pypi/v/tfreader.svg)](https://pypi.python.org/pypi/tfreader)

tensorflow_reader and all its dependencies can be easily installed with Python wheels.  If you do not want the installation to be to your current Python environment, you should first create and activate a [Python virtual environment (venv)](https://docs.python.org/3/tutorial/venv.html) to work in.  Then, run the following from the command-line:

```shell-script
pip install tfreader
```

Launch `python3`, import the tfreader package, and use it

```python
import tfreader as tfr
```
