"""Whole-slide image file reader for TensorFlow.

The histomics_stream.dataset.construction module supports initialization of a
tensorflow.data.Dataset workflow.  This module defines callable classes that return a
tensorflow.data.Dataset when called.

"""

import tensorflow as tf


class Study:
    """A class used to initialize a tensorflow.data.Dataset.

    A call to an instance of class histomics_stream.dataset.construction.Study can
    assigned to a tensorflow.data.Dataset to initiate a dataset pipeline.

    Parameters
    ----------
    version: str
        This is the string "version-1".

    """

    def __init__(self, version=["version-1"]):
        if not version == "version-1":
            raise ValueError(
                "The only currently supported value for the 'version' key is 'version-1'"
            )
        self.dictionary = {"version": [version]}

    """Call an instance of this callable class to get an instance of a tf.data.Dataset.
    """

    def __call__(self):
        return tf.data.Dataset.from_tensor_slices(self.dictionary)

    # """Cast an instance of this class to a dictionary in order to get an object that can be
    # supplied to a call to tf.data.Dataset.from_tensor_slices().  The keys(self) and
    # __getitem__(self, key) methods enable the casting of this class to a dictionary.

    # """

    # def keys(self):
    #     """The method that returns the keys of the key-value pairs stored by
    #     histomics_stream.dataset.construction.Study.

    #     """
    #     return self.dictionary.keys()

    # def __getitem__(self, key):
    #     """The method that returns the value corresponding to a key by
    #     histomics_stream.dataset.construction.Study.

    #     """
    #     return self.dictionary[key]
