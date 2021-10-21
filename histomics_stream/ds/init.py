"""Whole-slide image file reader for TensorFlow.

The histomics_stream.ds.init module supports transformations at the beginning of a
tensorflow.data.Dataset workflow.  This module defines objects that can be supplied to the
tensorflow.data.Dataset.from_tensor_slices() method.

"""


class Header:
    """A class used to initialize a tensorflow.data.Dataset.

    An instance of class histomics_stream.ds.init.Header can be cast to dict and supplied to
    tensorflow.data.Dataset.from_tensor_slices to create an instance of a tensorflow dataset object.
    The primary functionality of this class over an ordinary dictionary is that (1) it requires all the
    named keys, (2) it ensures that each value is a list or tuple, and (3) it expands via repetition any
    length-one lists to be the same length as the number of supplied filenames.

    Each parameter includes one value per slide to be analyzed.  However, if a parameter is a list (or
    tuple) of length 1 then that one value is used for every slide.

    Parameters
    ----------
    slides : str
        A list of names of the slides to be processed
    filenames : str
        A list of file names that contain the slides data
    cases : str
        A list of names of cases, where multiple slides could belong to each case
    magnifications : str
        A list of the desired manification levels that the slides should be analyzed at
    read_modes : str
        A list of keywords.  Currently only "tiled" is supported.
    mask_filenames : str
        A list of masks for the slides.  Each slide's mask will be used to select which tiles of the
        slide to process.  If the mask does not have one pixel per tile then it will be upsampled or
        downsampled as necessary.  An empty string indicates that no mask file is being supplied and
        that all tiles should be retained.

    Notes
    -----
    Because it has the keys and __getitem__ methods, this class can be cast to a Python dict.

    """

    def __init__(self, slides, filenames, cases, magnifications, read_modes, mask_filenames):
        self.dictionary = {
            "slide": slides,
            "filename": filenames,
            "case": cases,
            "magnification": magnifications,
            "read_mode": read_modes,
            "mask_filename": mask_filenames,
        }
        # Convert an entry to a list if it is not already a list or tuple
        for key in self.dictionary.keys():
            if not isinstance(self.dictionary[key], (list, tuple)):
                self.dictionary[key] = [
                    self.dictionary[key],
                ]
        # Make all singleton values have the same length as `filenames`
        if len(filenames) != 1:
            for key in self.dictionary.keys():
                if key != "filename" and len(self.dictionary[key]) == 1:
                    self.dictionary[key] = self.dictionary[key] * len(filenames)

    def keys(self):
        """The method that returns the keys of the key-value pairs stored by histomics_stream.ds.init.Header."""
        return self.dictionary.keys()

    def __getitem__(self, key):
        """The method that returns the value corresponding to a key by histomics_stream.ds.init.Header."""
        return self.dictionary[key]
