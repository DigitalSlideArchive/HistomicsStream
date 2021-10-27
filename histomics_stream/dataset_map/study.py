"""Whole-slide image file reader for TensorFlow.

The histomics_stream.dataset_map.study module supports transformations that operate on a
tensorflow.data.Dataset that has one element per study.  This module defines callable
classes that can be supplied to the tf.data.Dataset.map() method.

"""

import tensorflow as tf


class SetTileShape:
    """Sets the study-wide shape of a tile (a.k.a. field).

    An instance of class histomics_stream.dataset_map.study.SetTileShape can be supplied
    as an argument to tensorflow.dataset.map.

    Parameters
    ----------
    number_of_rows
        The desired height of each tile.
    number_of_columns
        The desired width of each tile.

    """

    def __init__(self, number_of_rows, number_of_columns):
        if not (isinstance(number_of_rows, int) and number_of_rows > 0):
            raise ValueError("number_of_rows must be a positive integer")
        if not (isinstance(number_of_columns, int) and number_of_columns > 0):
            raise ValueError("number_of_columns must be a positive integer")
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""
        return {
            **elem,
            "number_of_rows": number_of_rows,
            "number_of_columns": number_of_columns,
        }
