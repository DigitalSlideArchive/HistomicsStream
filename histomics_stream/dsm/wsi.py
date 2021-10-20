"""Whole-slide image file reader for TensorFlow.

The histomics_stream.dsm.wsi module supports transformations that operate on a tensorflow.data.Dataset
that has one element per whole-slide image.  This module defines objects that can be supplied to the
tf.data.Dataset.map() method.

"""

from PIL import Image
import itk
import math
import numpy as np
import openslide as os
import re
import tensorflow as tf
import zarr


class Header:
    """A class used to initialize a tensorflow.data.Dataset.

    An instance of class histomics_stream.dsm.wsi.Header can be cast to dict and supplied to
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
    Note that the __init__ method cannot be decorated with @tf.function for reasons that are not clear,
    but might (or might not!) be because an instance of a class (returned by the __init__ method) is not
    a tensorflow object.

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
        """The method that returns the keys of the key-value pairs stored by
        histomics_stream.dsm.wsi.Header.
        """
        return self.dictionary.keys()

    def __getitem__(self, key):
        """The method that returns the value corresponding to a key by histomics_stream.dsm.wsi.Header."""
        return self.dictionary[key]


class NoOp:
    """A class that does nothing that can be supplied to tensorflow.dataset.map

    An instance of class histomics_stream.dsm.wsi.NoOp can be supplied as an argument to
    tensorflow.dataset.map.  It is a "no operation" callable that accepts a single argument.

    This implementation is generic enough that the class can also be used at tensorflow workflow stages
    other than histomics_stream.dsm.wsi.

    """

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""
        return elem


class Print:
    """A class that does nothing but print that can be supplied to tensorflow.dataset.map

    An instance of class histomics_stream.dsm.wsi.Print can be supplied as an argument to
    tensorflow.dataset.map.  Like histomics_stream.dsm.wsi.NoOp, it is a "no operation" callable that
    accepts a single argument, though it has side effects in that it prints.  It demonstrates that
    tensorflow graph functionality is working by printing at tensorflow-graph trace time and at
    tensorflow-graph run time.  It also demonstrates that tensorflow graph functionality is properly
    handling the information flow from __init__ to __call__.

    This implementation is generic enough that the class can also be used at tensorflow workflow stages
    other than histomics_stream.dsm.wsi.

    Notes
    -----
    Note that the __init__ method cannot be decorated with @tf.function for reasons that are not clear,
    but might (or might not!) be because an instance of a class (returned by the __init__ method) is not
    a tensorflow object.

    """

    def __init__(self, member):
        self.member = member
        tf.print(
            "Running histomics_stream.dsm.wsi.Print.__init__, with member = ",
            self.member,
        )
        print("Tracing histomics_stream.dsm.wsi.Print.__init__")

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""
        tf.print(
            "Running histomics_stream.dsm.wsi.Print.__call__, with member = ",
            self.member,
        )
        print("Tracing histomics_stream.dsm.wsi.Print.__call__")
        return elem


class ComputeReadParameters:
    """A class that computes read parameters for slides in a tensorflow dataset.

    An instance of class histomics_stream.dsm.wsi.ComputeReadParameters can be supplied as an argument
    to tensorflow.dataset.map.  histomics_stream.dsm.wsi.ComputeReadParameters computes level, factor,
    width, and height from the inputs filename, magnification, and tolerance.
    histomics_stream.dsm.wsi.ComputeReadParameters adds new key-value pairs to the tensorflow dictionary
    for the newly computed values.  Ideally the implementation would be all tf.function (i.e., a
    tensorflow graph function); however, much of the code is via a tensorflow py_function because our
    current implementation for discerning the size of an image without reading in the pixel values uses
    non-tensorflow packages, such as openslide.

    Notes
    -----
    Note that the __init__ method cannot be decorated with @tf.function for reasons that are not clear,
    but might (or might not!) be because an instance of a class (returned by the __init__ method) is not
    a tensorflow object.

    """

    def __init__(self, tolerance=tf.constant(0.02, dtype=tf.float32)):
        self.tolerance = tolerance

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""

        level, factor, width, height = tf.py_function(
            func=self._py_compute_read_parameters,
            inp=[elem["filename"], elem["magnification"], self.tolerance],
            Tout=(tf.int32, tf.float32, tf.int32, tf.int32),
        )
        response = {
            **elem,
            "level": level,
            "factor": factor,
            "width": width,
            "height": height,
        }
        return response

    def _py_compute_read_parameters(self, filename_in, magnification_in, tolerance_in):
        """This method is the internal py_function (i.e. not @tf.function) that does the actual work of
        this class.
        """
        filename = filename_in.numpy().decode("utf-8")
        magnification = magnification_in.numpy()
        tolerance = tolerance_in.numpy()

        if re.compile(r"\.svs$").search(filename):
            # read whole-slide image file and create openslide object
            os_obj = os.OpenSlide(filename)

            # measure objective of level 0
            objective = np.float32(os_obj.properties[os.PROPERTY_NAME_OBJECTIVE_POWER])

            # calculate magnifications of levels
            estimated = np.array(objective / os_obj.level_downsamples)

            # Find best level to use and its factor
            level, factor = self._get_level_and_factor(magnification, estimated, tolerance)

            # get slide width, height at desired magnification. (Note width before height)
            width, height = os_obj.level_dimensions[level]

        elif re.compile(r"\.zarr$").search(filename):
            # read whole-slide image and create zarr objects
            store = zarr.DirectoryStore(filename)
            source_group = zarr.open(store, mode="r")

            # measure objective of level 0
            objective = np.float32(source_group.attrs[os.PROPERTY_NAME_OBJECTIVE_POWER])

            # calculate magnifications of levels
            estimated = np.array(objective / source_group.attrs["level_downsamples"])

            # Find best level to use and its factor
            level, factor = self._get_level_and_factor(magnification, estimated, tolerance)

            # get slide width, height at desired magnification. (Note height before width)
            height, width = source_group[format(level)].shape[0:2]

        else:
            # We don't know magnifications so assume reasonable values for level and factor.
            level = 0
            factor = 1.0
            if True:
                pil_obj = Image.open(filename)
                width, height = pil_obj.size
            else:
                # For the case that we know the image size without opening the file
                width, height = 2048, 2048

        print(f"level = {level}, factor = {factor}, width = {width}, height = {height}")
        return level, factor, width, height

    def _get_level_and_factor(self, magnification, estimated, tolerance):
        """This method computes level and factor for _py_compute_read_parameters."""

        # calculate difference with magnification levels
        delta = magnification - estimated

        # match to existing levels
        if np.min(np.abs(np.divide(delta, magnification))) < tolerance:  # match
            level = np.squeeze(np.argmin(np.abs(delta)))
            factor = 1.0
        elif np.any(delta < 0):
            value = np.max(delta[delta < 0])
            level = np.squeeze(np.argwhere(delta == value)[0])
            factor = magnification / estimated[level]
        else:  # desired magnification above base level - throw error
            raise ValueError("Cannot interpolate above scan magnification.")

        return level, factor


class AddTileDescription:
    """A class for supplying tile size and other information that can be supplied to
    tensorflow.dataset.map.

    An instance of class histomics_stream.dsm.wsi.AddTileDescription can be supplied as an argument to
    tensorflow.dataset.map.  histomics_stream.dsm.wsi.AddTileDescription adds new key-value pairs to the
    tensorflow dictionary to set the desired tile width, height, width overlap, and height overlap for
    each element.  chunk_width_factor and chunk_height_factor indicate how many tiles are read at a
    time.  The primary functionality of this class over an ordinary dictionary is that it sets all the
    required keys and no others.

    Parameters
    ----------
    tile_width : tf.constant(, dtype=tf.int32)
        The desired width of each tile.
    tile_height : tf.constant(, dtype=tf.int32)
        The desired height of each tile.
    overlap_width : tf.constant(, dtype=tf.int32)
        The amount of overlap of width between adjacent tiles.
    overlap_height : tf.constant(, dtype=tf.int32)
        The amount of overlap of height between adjacent tiles.
    chunk_width_factor : tf.constant(, dtype=tf.int32)
        The width of a chunk read from disk at one time as measured in number of (possibly) tiles.
    chunk_height_factor : tf.constant(, dtype=tf.int32)
        The height of a chunk read from disk at one time as measured in number of (possibly) tiles.

    Notes
    -----
    Note that the __init__ method cannot be decorated with @tf.function for reasons that are not clear,
    but might (or might not!) be because an instance of a class (returned by the __init__ method) is not
    a tensorflow object.

    """

    def __init__(
        self,
        tile_width=tf.constant(256, dtype=tf.int32),
        tile_height=tf.constant(256, dtype=tf.int32),
        overlap_width=tf.constant(0, dtype=tf.int32),
        overlap_height=tf.constant(0, dtype=tf.int32),
        chunk_width_factor=tf.constant(8, dtype=tf.int32),
        chunk_height_factor=tf.constant(8, dtype=tf.int32),
    ):
        self.dictionary = {
            "tw": tile_width,
            "th": tile_height,
            "ow": overlap_width,
            "oh": overlap_height,
            "cwf": chunk_width_factor,
            "chf": chunk_height_factor,
        }

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""

        return {**elem, **self.dictionary}


class ComputeResampledMask:
    """A class that uses mask information to select tiles.

    An instance of class histomics_stream.dsm.wsi.ComputeResampledMask can be supplied as an argument to
    tensorflow.dataset.map.  histomics_stream.dsm.wsi.ComputeResampledMask reads in a supplied mask and
    upsamples or downsamples it if necessary so that there is exactly one pixel in the mask for each
    tile in the input image.  Note that we are assuming that this will take care of any aspects related
    to the overlapping of tiles.  Subsequent to that, we will not be looking at the mask pixels for
    adjacent tiles even though they may overlap with the tile being considered.  Note further that we
    are assuming that the mask will be downsampled (or upsampled) to have one whole pixel per tile.

    """

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""

        mask_wsi = tf.py_function(
            func=self._py_compute_resampled_mask,
            inp=[
                elem["mask_filename"],
                elem["width"],
                elem["height"],
                elem["cwf"],
                elem["chf"],
                elem["tw"],
                elem["th"],
                elem["ow"],
                elem["oh"],
            ],
            Tout=tf.uint8,
        )
        return {**elem, "mask_wsi": mask_wsi}

    def _py_compute_resampled_mask(
        self,
        mask_filename_in,
        width_in,
        height_in,
        cwf_in,
        chf_in,
        tw_in,
        th_in,
        ow_in,
        oh_in,
    ):
        """This method is the internal py_function (i.e. not @tf.function) that does much of the actual
        work of this class.
        """

        mask_filename = mask_filename_in.numpy().decode("utf-8")
        width = width_in.numpy()
        height = height_in.numpy()
        cwf = cwf_in.numpy()
        chf = chf_in.numpy()
        tw = tw_in.numpy()
        th = th_in.numpy()
        ow = ow_in.numpy()
        oh = oh_in.numpy()

        left_bound = max(0, width - tw + 1)
        top_bound = max(0, height - th + 1)

        # The desired mask size is one pixel per tile
        resampled_width = int(np.floor((left_bound - 1) / (tw - ow)) + 1)
        resampled_height = int(np.floor((top_bound - 1) / (th - oh)) + 1)
        resampled_shape = (resampled_height, resampled_width)
        # By default assume that all tiles will be retained.
        mask = tf.convert_to_tensor(
            np.ones((resampled_height, resampled_width, 1), dtype=np.uint8),
            dtype=tf.uint8,
        )

        if mask_filename != "":
            # Read in an image from the supplied file name for the mask
            mask_itk = itk.imread(mask_filename)
            if mask_itk.GetImageDimension() != 2:
                raise ValueError("The mask should be a 2-dimensional image.")
            mask = tf.constant(itk.array_from_image(mask_itk), dtype=tf.uint8)

            # Add batch and channels dimensions
            mask = mask[tf.newaxis, ..., tf.newaxis]
            if (
                abs(math.log((resampled_width / mask.shape[2]) / (resampled_height / mask.shape[1])))
                > 0.20
            ):
                raise ValueError("The mask aspect ratio does not match the image aspect ratio.")

        # Perform the resampling
        resampled = tf.image.resize(mask, resampled_shape)[0, ...]

        # * At some point the mask gets broken up into chunks that are chunk_height_factor by
        # chunk_width_factor; to make tensorflow happy, we make sure that the dimensions of the mask are
        # multiples of those values.
        # * Also, we reshape so that the mask has shape (padded_height, padded_width, channels) where
        # channels = 1.
        # * Also, we cast the array to type np.uint8.  (Note that we use a formula with floor() rather
        # than a formula with ceil() so that we get the same answer with float or integer division for
        # the argument.)
        padded_resampled = np.zeros(
            (
                int((tf.math.floor((resampled_shape[0] - 1) / chf) + 1) * chf),
                int((tf.math.floor((resampled_shape[1] - 1) / cwf) + 1) * cwf),
                1,
            ),
            dtype=np.uint8,
        )
        padded_resampled[
            : resampled_shape[0],
            : resampled_shape[1],
            :1,
        ] = resampled
        del resampled

        # print(f"resampled_shape = {resampled_shape}")
        # print(f"padded_resampled.shape = {padded_resampled.shape}")
        # tf.print(padded_resampled[:8,:8,0])
        response = tf.convert_to_tensor(padded_resampled, dtype=tf.uint8)
        return response


class ComputeChunkPositions:
    """A class for computing the locations of chunks to be read from a whole slide.

    An instance of class histomics_stream.dsm.wsi.ComputeChunkPositions can be supplied as an argument
    to tensorflow.dataset.map.  histomics_stream.dsm.wsi.ComputeChunkPositions figures out what the read
    chunks will be based upon the tile parameters (size, overlap).  It divvys up the mask into pieces
    corresponding to the read chunks.  Note that it is important to subsequently call .unbatch() when it
    is desired that the chunks be not batched by slide.

    """

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""

        zero = tf.constant(0, dtype=tf.int32)
        one = tf.constant(1, dtype=tf.int32)
        chunk_width = elem["cwf"] * (elem["tw"] - elem["ow"]) + elem["ow"]
        chunk_height = elem["chf"] * (elem["th"] - elem["oh"]) + elem["oh"]

        # The left side of a tile cannot be as large as left_bound.  Also, the left side of a chunk
        # cannot be as large as left_bound because chunks contain a whole number of tiles.
        left_bound = tf.maximum(zero, elem["width"] - elem["tw"] + one)
        chunk_left = tf.range(zero, left_bound, chunk_width - elem["ow"])
        chunk_right = tf.clip_by_value(chunk_left + chunk_width, zero, elem["width"])

        top_bound = tf.maximum(zero, elem["height"] - elem["th"] + one)
        chunk_top = tf.range(zero, top_bound, chunk_height - elem["oh"])
        chunk_bottom = tf.clip_by_value(chunk_top + chunk_height, zero, elem["height"])

        cx = tf.tile(chunk_left, tf.stack([tf.size(chunk_top)]))
        cw = tf.tile(chunk_right - chunk_left, tf.stack([tf.size(chunk_top)]))
        cy = tf.repeat(chunk_top, tf.size(chunk_left))
        ch = tf.repeat(chunk_bottom - chunk_top, tf.size(chunk_left))
        chunk_len = tf.size(cx)

        # Compute a mask for each chunk.  The size of a mask for a chunk will be chunk_width_factor by
        # chunk_height_factor, even along the right or bottom border where it will be padded if
        # necessary.

        mask_chunks = tf.TensorArray(dtype=tf.uint8, size=chunk_len)
        mask_width = tf.shape(elem["mask_wsi"])[1]
        mask_left = tf.cast(chunk_left / (elem["tw"] - elem["ow"]), dtype=tf.int32)
        mask_right = tf.clip_by_value(mask_left + elem["cwf"], zero, mask_width)
        mask_height = tf.shape(elem["mask_wsi"])[0]
        mask_top = tf.cast(chunk_top / (elem["th"] - elem["oh"]), dtype=tf.int32)
        mask_bottom = tf.clip_by_value(mask_top + elem["chf"], zero, mask_height)
        mask_x = tf.tile(mask_left, tf.stack([tf.size(mask_top)]))
        mask_w = tf.tile(mask_right - mask_left, tf.stack([tf.size(mask_top)]))
        mask_y = tf.repeat(mask_top, tf.size(mask_left))
        mask_h = tf.repeat(mask_bottom - mask_top, tf.size(mask_left))
        mask_len = tf.size(mask_x)

        # mask_chunks = [elem["mask_wsi"][y:(y+h), x:(x+w)]
        #                for x, y, w, h in zip(mask_x, mask_w, mask_y, mask_h)]
        for i in tf.range(mask_len):
            mask_chunks = mask_chunks.write(
                i,
                tf.image.crop_to_bounding_box(
                    elem["mask_wsi"],
                    tf.gather(mask_y, i),
                    tf.gather(mask_x, i),
                    tf.gather(mask_h, i),
                    tf.gather(mask_w, i),
                ),
            )

        mask_chunks = mask_chunks.stack()
        response = {}
        for key in elem.keys():
            if key != "mask_wsi":
                # Exclude mask_wsi because it is large and we now have mask_chunk.
                response[key] = tf.repeat(elem[key], mask_len)
        response = {
            **response,
            "cx": cx,
            "cy": cy,
            "cw": cw,
            "ch": ch,
            "mask_chunk": mask_chunks,
        }
        return response
