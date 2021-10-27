"""Whole-slide image file reader for TensorFlow.

The histomics_stream.dataset_map.generic module supports transformations that operate on
a tensorflow.data.Dataset.  This module defines callable classes that can be supplied to
the tf.data.Dataset.map() method.

"""

import tensorflow as tf


class NoOp:
    """A do-nothing class that can be supplied to tensorflow.dataset.map.

    An instance of class histomics_stream.dataset_map.generic.NoOp can be supplied as an
    argument to tensorflow.dataset.map.  It is a "no operation" callable that accepts a
    single argument.

    """

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""
        return elem


class Print:
    """A class that does nothing but print that can be supplied to tensorflow.dataset.map.

    An instance of class histomics_stream.dataset_map.generic.Print can be supplied as an
    argument to tensorflow.dataset.map.  Like histomics_stream.dataset_map.generic.NoOp,
    it is a "no operation" callable that accepts a single argument, though it has side
    effects in that it prints.  It demonstrates that tensorflow graph functionality is
    working by printing at tensorflow-graph trace time and at tensorflow-graph run time.
    It also demonstrates that tensorflow graph functionality is properly handling the
    information flow from __init__ to __call__.

    Notes
    -----
    Note that the __init__ method cannot be decorated with @tf.function for reasons that
    are not clear, but might (or might not!) be because an instance of a class (returned
    by the __init__ method) is not a tensorflow object.

    """

    def __init__(self, member):
        self.member = member
        tf.print(
            "Running histomics_stream.dataset_map.generic.Print.__init__, with member = ",
            self.member,
        )
        print("Tracing histomics_stream.dataset_map.generic.Print.__init__")

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""
        tf.print(
            "Running histomics_stream.dataset_map.generic.Print.__call__, with member = ",
            self.member,
        )
        print("Tracing histomics_stream.dataset_map.generic.Print.__call__")
        return elem
