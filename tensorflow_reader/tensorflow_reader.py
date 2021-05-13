"""Whole-slide image file reader for TensorFlow.

This module supports efficient whole-slide reading and processing in a tensorflow graph.

"""

__version__ = "1.0.7"

from datetime import datetime
from imagecodecs import jpeg_encode, jpeg_decode, jpeg2k_encode, jpeg2k_decode
from matplotlib import pyplot as plt
from napari_lazy_openslide.lazy_openslide import OpenSlideStore
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray, ensure_contiguous_ndarray, ndarray_copy
from numcodecs.registry import register_codec
from os import makedirs
from PIL import Image
import fsspec
import h5py
import itk
import math
import numpy as np
import openslide as os
import random
import re
import tensorflow as tf
import tifffile
import time
import zarr


class jpeg(Codec):
    """Codec providing jpeg compression via imagecodecs.

    Parameters
    ----------
    quality : int
        Compression level.

    Notes
    -----
    For the code that uses Zarr data storage for jpeg images, we need to supply codecs.  Note that we
    use this codec instead of that available from the zarr_jpeg package.  The latter collapses
    dimensions by default, can require us to transpose dimensions, and can miss optimizations based upon
    RGB data.

    """

    codec_id = "jpeg"

    def __init__(self, quality=100):
        self.quality = quality
        assert 0 < self.quality <= 100 and isinstance(self.quality, int)
        super().__init__()

    def encode(self, buf):
        """The method to encode a raw image into jpeg format.

        Parameters
        ----------
        buf : ndarray
            The raw image to be encoded into jpeg format

        Returns
        -------
        ndarray
            The image in jpeg format

        """

        bufa = ensure_ndarray(buf)
        assert 2 <= bufa.ndim <= 3
        return jpeg_encode(bufa, level=self.quality)

    def decode(self, buf, out=None):
        """The method to decode a jpeg image into a raw format.

        Parameters
        ----------
        buf : contiguous_ndarray
            The jpeg image to be decoded into raw format.
        out : contiguous_ndarray, optional
            Another location to write the raw image to.

        Returns
        -------
        ndarray
            The image in raw format

        """

        buf = ensure_contiguous_ndarray(buf)
        if out is not None:
            out = ensure_contiguous_ndarray(out)
        tiled = jpeg_decode(buf)
        return ndarray_copy(tiled, out)


register_codec(jpeg)


class jpeg2k(Codec):
    """Codec providing jpeg2k compression via imagecodecs.

    Parameters
    ----------
    quality : int
        Compression level.

    """

    codec_id = "jpeg2k"

    def __init__(self, quality=100):
        self.quality = quality
        assert 0 < self.quality <= 100 and isinstance(self.quality, int)
        super().__init__()

    def encode(self, buf):
        """The method to encode a raw image into jpeg2k format.

        Parameters
        ----------
        buf : ndarray
            The raw image to be encoded into jpeg2k format

        Returns
        -------
        ndarray
            The image in jpeg2k format

        """

        bufa = ensure_ndarray(buf)
        assert 2 <= bufa.ndim <= 3
        return jpeg2k_encode(bufa, level=self.quality)

    def decode(self, buf, out=None):
        """The method to decode a jpeg2k image into a raw format.

        Parameters
        ----------
        buf : contiguous_ndarray
            The jpeg2k image to be decoded into raw format.
        out : contiguous_ndarray, optional
            Another location to write the raw image to.

        Returns
        -------
        ndarray
            The image in raw format

        """

        buf = ensure_contiguous_ndarray(buf)
        if out is not None:
            out = ensure_contiguous_ndarray(out)
        tiled = jpeg2k_decode(buf)
        return ndarray_copy(tiled, out)


register_codec(jpeg2k)


class tfdHeader:
    """A class used to initialize a tensorflow.data.Dataset.

    An instance of class tfdHeader can be cast to dict and supplied to
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
        downsampled as necessary.

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
        """The method that returns the keys of the key-value pairs stored by tfdHeader."""
        return self.dictionary.keys()

    def __getitem__(self, key):
        """The method that returns the value corresponding to a key by tfdHeader."""
        return self.dictionary[key]


class tfdNoOp:
    """A class that does nothing that can be supplied to tensorflow.dataset.map

    An instance of class tfdNoOp can be supplied as an argument to tensorflow.dataset.map.  It is a "no
    operation" callable that accepts a single argument.

    """

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""
        return elem


class tfdPrint:
    """A class that does nothing but print that can be supplied to tensorflow.dataset.map

    An instance of class tfdPrint can be supplied as an argument to tensorflow.dataset.map.  Like
    tfdNoOp, it is a "no operation" callable that accepts a single argument, though it has side effects
    in that it prints.  It demonstrates that tensorflow graph functionality is working by printing at
    tensorflow-graph trace time and at tensorflow-graph run time.  It also demonstrates that tensorflow
    graph functionality is properly handling the information flow from __init__ to __call__.

    Notes
    -----
    Note that the __init__ method cannot be decorated with @tf.function for reasons that are not clear,
    but might (or might not!) be because an instance of a class (returned by the __init__ method) is not
    a tensorflow object.

    """

    def __init__(self, member):
        self.member = member
        tf.print("Running tfdPrint.__init__, with member = ", self.member)
        print("Tracing tfdPrint.__init__")

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""

        tf.print("Running tfdPrint.__call__, with member = ", self.member)
        print("Tracing tfdPrint.__call__")
        return elem


class ComputeReadParameters:
    """A class that computes read parameters for slides in a tensorflow dataset.

    An instance of class ComputeReadParameters can be supplied as an argument to tensorflow.dataset.map.
    ComputeReadParameters computes level, factor, width, and height from the inputs filename,
    magnification, and tolerance.  ComputeReadParameters adds new key-value pairs to the tensorflow
    dictionary for the newly computed values.  Ideally the implementation would be all tf.function
    (i.e., a tensorflow graph function); however, much of the code is via a tensorflow py_function
    because our current implementation for discerning the size of an image without reading in the pixel
    values uses non-tensorflow packages, such as openslide.

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
        response = {**elem, "level": level, "factor": factor, "width": width, "height": height}
        return response

    def _py_compute_read_parameters(self, filenameIn, magnificationIn, toleranceIn):
        """This method is the internal py_function (i.e. not @tf.function) that does the actual work of this class."""

        filename = filenameIn.numpy().decode("utf-8")
        magnification = magnificationIn.numpy()
        tolerance = toleranceIn.numpy()

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
    """A class for supplying tile size and other information that can be supplied to tensorflow.dataset.map

    An instance of class AddTileDescription can be supplied as an argument to tensorflow.dataset.map.
    AddTileDescription adds new key-value pairs to the tensorflow dictionary to set the desired tile
    width, height, width overlap, and height overlap for each element, and to indicate whether we want
    tiles even if they are fractional (aka of truncated size) due to partially falling off the edge of
    the image.  (These fractional tiles can be problematic because tensorflow likes its shapes to be
    uniform.)  chunk_width_factor and chunk_height_factor indicate how many tiles are read at a time.
    The primary functionality of this class over an ordinary dictionary is that it sets all the required
    keys and no others.

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
    fractional : tf.constant(, dtype=tf.bool)
        Set to True if tiles at the edges of the slide that are less than full width or full height should be retained.

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
        fractional=tf.constant(False, dtype=tf.bool),
    ):
        self.dictionary = {
            "tw": tile_width,
            "th": tile_height,
            "ow": overlap_width,
            "oh": overlap_height,
            "cwf": chunk_width_factor,
            "chf": chunk_height_factor,
            "fractional": fractional,
        }

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""

        return {**elem, **self.dictionary}


class ComputeResampledMask:
    """A class that uses mask information to select tiles.

    An instance of class ComputeResampledMask can be supplied as an argument to tensorflow.dataset.map.
    ComputeResampledMask reads in a supplied mask and upsamples or downsamples it if necessary so that
    there is exactly one pixel in the mask for each tile in the input image.  Note that we are assuming
    that this will take care of any aspects related to the overlapping of tiles.  Subsequent to that, we
    will not be looking at the mask pixels for adjacent tiles even though they may overlap with the tile
    being considered.  Note further that we are assuming that the mask will be downsampled (or
    upsampled) to have one whole pixel per tile, even if not every tile is whole; that is even if some
    are fractional tiles from the right or bottom edges.

    """

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""

        if "mask_filename" in elem.keys():
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
                    elem["fractional"],
                ],
                Tout=tf.bool,
            )
            return {**elem, "mask_wsi": mask_wsi}
        else:
            return elem

    def _py_compute_resampled_mask(
        self, mask_filenameIn, widthIn, heightIn, cwfIn, chfIn, twIn, thIn, owIn, ohIn, fractionalIn
    ):
        """This method is the internal py_function (i.e. not @tf.function) that does much of the actual work of this class."""

        mask_filename = mask_filenameIn.numpy().decode("utf-8")
        width = widthIn.numpy()
        height = heightIn.numpy()
        cwf = cwfIn.numpy()
        chf = chfIn.numpy()
        tw = twIn.numpy()
        th = thIn.numpy()
        ow = owIn.numpy()
        oh = ohIn.numpy()
        fractional = fractionalIn.numpy()

        left_bound = max(0, width - ow if fractional else width - tw + 1)
        top_bound = max(0, height - oh if fractional else height - th + 1)

        # Read in an image from the supplied file name for the mask
        mask_itk = itk.imread(mask_filename)
        if mask_itk.GetImageDimension() != 2:
            raise ValueError("The mask should be a 2-dimensional image.")
        mask = tf.constant(itk.array_from_image(mask_itk))
        # Add batch and channels dimensions
        mask = mask[tf.newaxis, ..., tf.newaxis]
        # The new size is one pixel per tile
        resampled_width = int(np.floor((left_bound - 1) / (tw - ow)) + 1)
        resampled_height = int(np.floor((top_bound - 1) / (th - oh)) + 1)
        if abs(math.log((resampled_width / mask.shape[2]) / (resampled_height / mask.shape[1]))) > 0.20:
            raise ValueError("The mask aspect ratio does not match the image aspect ratio.")
        resampled_shape = (resampled_height, resampled_width)
        # Perform the resampling
        resampled = tf.image.resize(mask, resampled_shape)[0, ...]

        # * At some point the mask gets broken up into chunks that are chunk_height_factor by
        # chunk_width_factor; to make tensorflow happy, we make sure that the dimensions of the mask are
        # multiples of those values.
        # * Also, we reshape so that the mask has shape (padded_height, padded_width, channels) where
        # channels = 1.
        # * Also, we cast the array to type np.bool.  (Note that we use a formula with floor() rather than a
        # formula with ceil() so that we get the same answer with float or integer division for the
        # argument.)
        padded_resampled = np.zeros(
            (
                int((tf.math.floor((resampled_shape[0] - 1) / chf) + 1) * chf),
                int((tf.math.floor((resampled_shape[1] - 1) / cwf) + 1) * cwf),
                1,
            ),
            dtype=np.bool,
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
        return padded_resampled


class ComputeChunkPositions:
    """A class for computing the locations of chunks to be read from a whole slide.

    An instance of class ComputeChunkPositions can be supplied as an argument to tensorflow.dataset.map.
    ComputeChunkPositions figures out what the read chunks will be based upon the tile parameters (size,
    overlap, fractional).  It divvys up the mask into pieces corresponding to the read chunks.  Note
    that it is important to subsequently call .unbatch() when it is desired that the chunks be not
    batched by slide.

    """

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""

        zero = tf.constant(0, dtype=tf.int32)
        one = tf.constant(1, dtype=tf.int32)
        chunk_width = elem["cwf"] * (elem["tw"] - elem["ow"]) + elem["ow"]
        chunk_height = elem["chf"] * (elem["th"] - elem["oh"]) + elem["oh"]

        # The left side of a tile cannot be as large as left_bound.  Also, the left side of a chunk cannot
        # be as large as left_bound because chunks contain a whole number of tiles.
        left_bound = tf.maximum(
            zero,
            elem["width"] - elem["ow"] if elem["fractional"] else elem["width"] - elem["tw"] + one,
        )
        chunk_left = tf.range(zero, left_bound, chunk_width - elem["ow"])
        chunk_right = tf.clip_by_value(chunk_left + chunk_width, zero, elem["width"])

        top_bound = tf.maximum(
            zero,
            elem["height"] - elem["oh"] if elem["fractional"] else elem["height"] - elem["th"] + one,
        )
        chunk_top = tf.range(zero, top_bound, chunk_height - elem["oh"])
        chunk_bottom = tf.clip_by_value(chunk_top + chunk_height, zero, elem["height"])

        cx = tf.tile(chunk_left, tf.stack([tf.size(chunk_top)]))
        cw = tf.tile(chunk_right - chunk_left, tf.stack([tf.size(chunk_top)]))
        cy = tf.repeat(chunk_top, tf.size(chunk_left))
        ch = tf.repeat(chunk_bottom - chunk_top, tf.size(chunk_left))
        len = tf.size(cx)

        # If a mask was supplied, compute a mask for each chunk.  The size of a mask for a chunk will be
        # chunk_width_factor by chunk_height_factor, even along the right or bottom border where it will be
        # padded if necessary.
        has_mask = "mask_wsi" in elem.keys()

        mask_chunks = tf.TensorArray(dtype=tf.bool, size=len)
        if has_mask:
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
            len = tf.size(mask_x)

            # mask_chunks = [elem["mask_wsi"][y:(y+h), x:(x+w)] for x, y, w, h in zip(mask_x, mask_w, mask_y, mask_h)]
            condition = lambda i, _: tf.less(i, len)
            body = lambda i, mask_chunks: (
                i + 1,
                mask_chunks.write(
                    i,
                    tf.image.crop_to_bounding_box(
                        elem["mask_wsi"],
                        tf.gather(mask_y, i),
                        tf.gather(mask_x, i),
                        tf.gather(mask_h, i),
                        tf.gather(mask_w, i),
                    ),
                ),
            )
            _, mask_chunks = tf.while_loop(condition, body, [0, mask_chunks])

        mask_chunks = mask_chunks.stack()
        response = {}
        for key in elem.keys():
            if key != "mask_wsi":
                # Exclude mask_wsi because it is large and we now have mask_chunk.
                response[key] = tf.repeat(elem[key], len)
        response = {**response, "cx": cx, "cy": cy, "cw": cw, "ch": ch}
        if has_mask:
            response = {**response, "mask_chunk": mask_chunks}
        return response


class ReadAndSplitChunk:
    """A class that reads a chunk from disk and splits it into tiles.

    An instance of class ReadAndSplitChunk can be supplied as an argument to tensorflow.dataset.map.
    ReadAndSplitChunk reads in each chunk (unless no tile from the chunk is needed due to the mask), and
    then splits it into tile, retaining only those tiles indicated by them mask.  Note that it is
    important to subsequently call .unbatch() when it is desired that the tiles be not batched by chunk.

    """

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""

        zero8 = tf.constant(0, dtype=tf.uint8)
        zero32 = tf.constant(0, dtype=tf.int32)
        one8 = tf.constant(1, dtype=tf.uint8)
        one32 = tf.constant(1, dtype=tf.int32)

        left_bound = tf.maximum(
            zero32,
            elem["cw"] - elem["ow"] if elem["fractional"] else elem["cw"] - elem["tw"] + one32,
        )
        tile_left = tf.range(zero32, left_bound, elem["tw"] - elem["ow"])
        tile_right = tf.clip_by_value(tile_left + elem["tw"], zero32, elem["cw"])
        usable_mask_width = tf.size(tile_left)

        top_bound = tf.maximum(
            zero32,
            elem["ch"] - elem["oh"] if elem["fractional"] else elem["ch"] - elem["th"] + one32,
        )
        tile_top = tf.range(zero32, top_bound, elem["th"] - elem["oh"])
        tile_bottom = tf.clip_by_value(tile_top + elem["th"], zero32, elem["ch"])
        usable_mask_height = tf.size(tile_top)

        x = tf.tile(tile_left, tf.stack([tf.size(tile_top)]))
        w = tf.tile(tile_right - tile_left, tf.stack([tf.size(tile_top)]))
        y = tf.repeat(tile_top, tf.size(tile_left))
        h = tf.repeat(tile_bottom - tile_top, tf.size(tile_left))
        len = tf.size(x)
        tiles = tf.TensorArray(dtype=tf.uint8, size=len)

        mask_chunk_supplied = "mask_chunk" in elem.keys()
        if mask_chunk_supplied:
            mask_chunk = elem["mask_chunk"]

        # Handle the case that we need to read from disk
        read_chunk = not mask_chunk_supplied or tf.math.reduce_any(mask_chunk)
        if read_chunk:
            chunk = tf.py_function(
                func=self._py_read_chunk,
                inp=[
                    elem["filename"],
                    elem["level"],
                    elem["cx"],
                    elem["cy"],
                    elem["cw"],
                    elem["ch"],
                ],
                Tout=tf.uint8,
            )

            condition = lambda i, _: tf.less(i, len)
            body = lambda i, tiles: (
                i + 1,
                tiles.write(
                    i,
                    tf.image.crop_to_bounding_box(
                        chunk,
                        tf.gather(y, i),
                        tf.gather(x, i),
                        tf.gather(h, i),
                        tf.gather(w, i),
                    ),
                ),
            )
            _, tiles = tf.while_loop(condition, body, [0, tiles])
            del chunk

        tiles = tiles.stack()

        # Figure out which tiles we are going to keep
        if not read_chunk:
            # Keep nothing
            where = tf.where(tf.repeat(False, len))
        elif mask_chunk_supplied:
            # Use the supplied mask
            where = tf.where(
                tf.reshape(
                    mask_chunk[:usable_mask_height, :usable_mask_width, 0],
                    [
                        len,
                    ],
                )
            )
        else:
            # Keep everything
            where = tf.where(tf.repeat(True, len))
        # Change shape from (tf.size(where), 1) to (tf.size(where),)
        where = tf.reshape(where, [tf.size(where)])

        # Construct the response
        all_tiles = {}
        for key in elem.keys():
            if key != "mask_chunk":
                # Exclude mask_chunk because it is useless at the tile level
                all_tiles[key] = tf.repeat(elem[key], len)
        all_tiles = {
            **all_tiles,
            "tx": elem["cx"] + x,
            "ty": elem["cy"] + y,
            "tw": w,
            "th": h,
            "tile": tiles,
        }

        # Keep only the tiles that we want.  If we have read in the chunk and don't have mask_chunk supplied
        # then there is nothing to do because we are keeping everything.

        response = {}
        for key in all_tiles.keys():
            response[key] = tf.gather(all_tiles[key], where)

        return response

    def _py_read_chunk(self, filenameIn, level, x, y, w, h):
        """This method is the internal py_function (i.e. not @tf.function) that invokes the openslide package for reading."""

        filename = filenameIn.numpy().decode("utf-8")
        if re.compile(r"\.svs$").search(filename):
            if True:
                # Use OpenSlide to read SVS image
                os_obj = os.OpenSlide(filename)

                # read chunk and convert to tensor
                chunk = np.array(
                    os_obj.read_region((x.numpy(), y.numpy()), level.numpy(), (w.numpy(), h.numpy()))
                )
            elif True:
                # Use fsspec to read the SVS image
                # This is NOT working code!!!
                with fsspec.open(filename) as store:
                    source_group = zarr.open(store, mode="r")
                    # Zarr formats other than using zarr-jpeg package have shape (height, width, colors)
                    # using order="C".
                    chunk = source_group[format(level.numpy())][
                        y.numpy() : (y.numpy() + h.numpy()),
                        x.numpy() : (x.numpy() + w.numpy()),
                        :,
                    ]
            elif False:
                # Read the SVS image with napari_lazy_openslide.lazy_openslide.OpenSlideStore
                store = OpenSlideStore(filename, tilesize=2048)
                source_group = zarr.open(store, mode="r")
                # Zarr formats other than zarr-jpeg have shape (height, width, colors) using order="C".
                chunk = source_group[format(level.numpy())][
                    y.numpy() : (y.numpy() + h.numpy()),
                    x.numpy() : (x.numpy() + w.numpy()),
                    :,
                ]
            else:
                # Use tifffile to read the SVS image 'aszarr'
                # store = tifffile.imread(filename)
                store = tifffile.imread(filename, aszarr=True)
                # store = tifffile.imread(filename, aszarr=True, chunkmode="page")
                source_group = zarr.open(store, mode="r")
                chunk = source_group[format(level.numpy())][
                    y.numpy() : (y.numpy() + h.numpy()),
                    x.numpy() : (x.numpy() + w.numpy()),
                    :,
                ]

        elif re.compile(r"\.zarr$").search(filename):
            # `filename` is a directory that stores an image in Zarr format.
            store = zarr.DirectoryStore(filename)
            source_group = zarr.open(store, mode="r")
            # Zarr formats other than using zarr-jpeg package have shape (height, width, colors) using
            # order="C".
            chunk = source_group[format(level.numpy())][
                y.numpy() : (y.numpy() + h.numpy()),
                x.numpy() : (x.numpy() + w.numpy()),
                :,
            ]
            # Do chunk width and height need to be transposed to be consistent with SVS?!!!

        else:
            pil_obj = Image.open(filename)
            chunk = np.asarray(pil_obj)[
                x.numpy() : (x.numpy() + w.numpy()), y.numpy() : (y.numpy() + h.numpy()), :
            ]

        # Do we want to support other than uint8?!!!
        return tf.convert_to_tensor(chunk[..., :3], dtype=tf.uint8)


def _merge_dist_tensor(strategy, distributed, axis=0):
    """A helper function for predict_example."""

    # check if input is type produced by distributed.Strategy.run
    if isinstance(distributed, tf.python.distribute.values.PerReplica):
        return tf.concat(strategy.experimental_local_results(distributed), axis=axis)
    else:
        raise ValueError("Input to _merge_dist_tensor not a distributed PerReplica tensor.")


def _merge_dist_dict(strategy, distributed, axis=0):
    """A helper function for predict_example."""

    # check if input is type produced by distributed.Strategy.run
    if isinstance(distributed, dict):
        for key in distributed.keys():
            distributed[key] = _merge_dist_tensor(strategy, distributed[key], axis=axis)
        return distributed
    else:
        raise ValueError("Input to _merge_dist_tensor not a dict.")


def strategy_example():
    """Example code for seting up a tensorflow strategy."""

    tic = time.time()
    # find available GPUs
    devices = [
        gpu.name.replace("/physical_device:", "/").lower()
        for gpu in tf.config.experimental.list_physical_devices("GPU")
    ]
    #
    devices = devices[0:8]

    # define strategy
    strategy = tf.distribute.MirroredStrategy(devices=devices)

    # generate network model
    with strategy.scope():
        model = tf.keras.applications.VGG16(include_top=True, weights="imagenet")
        model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("fc1").output)

    print("strategy_example: %f seconds" % (time.time() - tic))
    return devices, strategy, model


def read_example(devices, strategy):
    """Example code for the file reading portion of a tensorflow dataset pipeline."""

    tic = time.time()

    dataset_map_options = {
        "num_parallel_calls": tf.data.experimental.AUTOTUNE,
        "deterministic": False,
    }
    dataset_map_options = {"num_parallel_calls": tf.data.experimental.AUTOTUNE}
    # Each key=value pair in the dictionary should have a value that is a list of the same length.
    # Taken together, the Nth entry from each list comprise a dictionary that is the Nth element in the
    # dataset.

    all_wsi_images = [
        "jpeg75.stack/0.0.jpeg",
        "jpeg75.stack/0.10240.jpeg",
        "jpeg75.stack/0.12288.jpeg",
        "jpeg75.stack/0.14336.jpeg",
        "jpeg75.stack/0.16384.jpeg",
        "jpeg75.stack/0.18432.jpeg",
        "jpeg75.stack/0.20480.jpeg",
        "jpeg75.stack/0.2048.jpeg",
        "jpeg75.stack/0.22528.jpeg",
        "jpeg75.stack/0.24576.jpeg",
        "jpeg75.stack/0.26624.jpeg",
        "jpeg75.stack/0.28672.jpeg",
        "jpeg75.stack/0.30720.jpeg",
        "jpeg75.stack/0.32768.jpeg",
        "jpeg75.stack/0.34816.jpeg",
        "jpeg75.stack/0.36864.jpeg",
        "jpeg75.stack/0.38912.jpeg",
        "jpeg75.stack/0.40960.jpeg",
        "jpeg75.stack/0.4096.jpeg",
        "jpeg75.stack/0.43008.jpeg",
        "jpeg75.stack/0.45056.jpeg",
        "jpeg75.stack/0.47104.jpeg",
        "jpeg75.stack/0.49152.jpeg",
        "jpeg75.stack/0.51200.jpeg",
        "jpeg75.stack/0.53248.jpeg",
        "jpeg75.stack/0.55296.jpeg",
        "jpeg75.stack/0.57344.jpeg",
        "jpeg75.stack/0.59392.jpeg",
        "jpeg75.stack/0.61440.jpeg",
        "jpeg75.stack/0.6144.jpeg",
        "jpeg75.stack/0.63488.jpeg",
        "jpeg75.stack/0.65536.jpeg",
        "jpeg75.stack/0.67584.jpeg",
        "jpeg75.stack/0.69632.jpeg",
        "jpeg75.stack/0.71680.jpeg",
        "jpeg75.stack/0.73728.jpeg",
        "jpeg75.stack/0.75776.jpeg",
        "jpeg75.stack/0.77824.jpeg",
        "jpeg75.stack/0.79872.jpeg",
        "jpeg75.stack/0.81920.jpeg",
        "jpeg75.stack/0.8192.jpeg",
        "jpeg75.stack/100352.0.jpeg",
        "jpeg75.stack/100352.10240.jpeg",
        "jpeg75.stack/100352.12288.jpeg",
        "jpeg75.stack/100352.14336.jpeg",
        "jpeg75.stack/100352.16384.jpeg",
        "jpeg75.stack/100352.18432.jpeg",
        "jpeg75.stack/100352.20480.jpeg",
        "jpeg75.stack/100352.2048.jpeg",
        "jpeg75.stack/100352.22528.jpeg",
        "jpeg75.stack/100352.24576.jpeg",
        "jpeg75.stack/100352.26624.jpeg",
        "jpeg75.stack/100352.28672.jpeg",
        "jpeg75.stack/100352.30720.jpeg",
        "jpeg75.stack/100352.32768.jpeg",
        "jpeg75.stack/100352.34816.jpeg",
        "jpeg75.stack/100352.36864.jpeg",
        "jpeg75.stack/100352.38912.jpeg",
        "jpeg75.stack/100352.40960.jpeg",
        "jpeg75.stack/100352.4096.jpeg",
        "jpeg75.stack/100352.43008.jpeg",
        "jpeg75.stack/100352.45056.jpeg",
        "jpeg75.stack/100352.47104.jpeg",
        "jpeg75.stack/100352.49152.jpeg",
        "jpeg75.stack/100352.51200.jpeg",
        "jpeg75.stack/100352.53248.jpeg",
        "jpeg75.stack/100352.55296.jpeg",
        "jpeg75.stack/100352.57344.jpeg",
        "jpeg75.stack/100352.59392.jpeg",
        "jpeg75.stack/100352.61440.jpeg",
        "jpeg75.stack/100352.6144.jpeg",
        "jpeg75.stack/100352.63488.jpeg",
        "jpeg75.stack/100352.65536.jpeg",
        "jpeg75.stack/100352.67584.jpeg",
        "jpeg75.stack/100352.69632.jpeg",
        "jpeg75.stack/100352.71680.jpeg",
        "jpeg75.stack/100352.73728.jpeg",
        "jpeg75.stack/100352.75776.jpeg",
        "jpeg75.stack/100352.77824.jpeg",
        "jpeg75.stack/100352.79872.jpeg",
        "jpeg75.stack/100352.81920.jpeg",
        "jpeg75.stack/100352.8192.jpeg",
        "jpeg75.stack/102400.0.jpeg",
        "jpeg75.stack/102400.10240.jpeg",
        "jpeg75.stack/102400.12288.jpeg",
        "jpeg75.stack/102400.14336.jpeg",
        "jpeg75.stack/102400.16384.jpeg",
        "jpeg75.stack/102400.18432.jpeg",
        "jpeg75.stack/102400.20480.jpeg",
        "jpeg75.stack/102400.2048.jpeg",
        "jpeg75.stack/102400.22528.jpeg",
        "jpeg75.stack/102400.24576.jpeg",
        "jpeg75.stack/102400.26624.jpeg",
        "jpeg75.stack/102400.28672.jpeg",
        "jpeg75.stack/102400.30720.jpeg",
        "jpeg75.stack/102400.32768.jpeg",
        "jpeg75.stack/102400.34816.jpeg",
        "jpeg75.stack/102400.36864.jpeg",
        "jpeg75.stack/102400.38912.jpeg",
        "jpeg75.stack/102400.40960.jpeg",
        "jpeg75.stack/102400.4096.jpeg",
        "jpeg75.stack/102400.43008.jpeg",
        "jpeg75.stack/102400.45056.jpeg",
        "jpeg75.stack/102400.47104.jpeg",
        "jpeg75.stack/102400.49152.jpeg",
        "jpeg75.stack/102400.51200.jpeg",
        "jpeg75.stack/102400.53248.jpeg",
        "jpeg75.stack/102400.55296.jpeg",
        "jpeg75.stack/102400.57344.jpeg",
        "jpeg75.stack/102400.59392.jpeg",
        "jpeg75.stack/102400.61440.jpeg",
        "jpeg75.stack/102400.6144.jpeg",
        "jpeg75.stack/102400.63488.jpeg",
        "jpeg75.stack/102400.65536.jpeg",
        "jpeg75.stack/102400.67584.jpeg",
        "jpeg75.stack/102400.69632.jpeg",
        "jpeg75.stack/102400.71680.jpeg",
        "jpeg75.stack/102400.73728.jpeg",
        "jpeg75.stack/102400.75776.jpeg",
        "jpeg75.stack/102400.77824.jpeg",
        "jpeg75.stack/102400.79872.jpeg",
        "jpeg75.stack/102400.81920.jpeg",
        "jpeg75.stack/102400.8192.jpeg",
        "jpeg75.stack/10240.0.jpeg",
        "jpeg75.stack/10240.10240.jpeg",
        "jpeg75.stack/10240.12288.jpeg",
        "jpeg75.stack/10240.14336.jpeg",
        "jpeg75.stack/10240.16384.jpeg",
        "jpeg75.stack/10240.18432.jpeg",
        "jpeg75.stack/10240.20480.jpeg",
        "jpeg75.stack/10240.2048.jpeg",
        "jpeg75.stack/10240.22528.jpeg",
        "jpeg75.stack/10240.24576.jpeg",
        "jpeg75.stack/10240.26624.jpeg",
        "jpeg75.stack/10240.28672.jpeg",
        "jpeg75.stack/10240.30720.jpeg",
        "jpeg75.stack/10240.32768.jpeg",
        "jpeg75.stack/10240.34816.jpeg",
        "jpeg75.stack/10240.36864.jpeg",
        "jpeg75.stack/10240.38912.jpeg",
        "jpeg75.stack/10240.40960.jpeg",
        "jpeg75.stack/10240.4096.jpeg",
        "jpeg75.stack/10240.43008.jpeg",
        "jpeg75.stack/10240.45056.jpeg",
        "jpeg75.stack/10240.47104.jpeg",
        "jpeg75.stack/10240.49152.jpeg",
        "jpeg75.stack/10240.51200.jpeg",
        "jpeg75.stack/10240.53248.jpeg",
        "jpeg75.stack/10240.55296.jpeg",
        "jpeg75.stack/10240.57344.jpeg",
        "jpeg75.stack/10240.59392.jpeg",
        "jpeg75.stack/10240.61440.jpeg",
        "jpeg75.stack/10240.6144.jpeg",
        "jpeg75.stack/10240.63488.jpeg",
        "jpeg75.stack/10240.65536.jpeg",
        "jpeg75.stack/10240.67584.jpeg",
        "jpeg75.stack/10240.69632.jpeg",
        "jpeg75.stack/10240.71680.jpeg",
        "jpeg75.stack/10240.73728.jpeg",
        "jpeg75.stack/10240.75776.jpeg",
        "jpeg75.stack/10240.77824.jpeg",
        "jpeg75.stack/10240.79872.jpeg",
        "jpeg75.stack/10240.81920.jpeg",
        "jpeg75.stack/10240.8192.jpeg",
        "jpeg75.stack/104448.0.jpeg",
        "jpeg75.stack/104448.10240.jpeg",
        "jpeg75.stack/104448.12288.jpeg",
        "jpeg75.stack/104448.14336.jpeg",
        "jpeg75.stack/104448.16384.jpeg",
        "jpeg75.stack/104448.18432.jpeg",
        "jpeg75.stack/104448.20480.jpeg",
        "jpeg75.stack/104448.2048.jpeg",
        "jpeg75.stack/104448.22528.jpeg",
        "jpeg75.stack/104448.24576.jpeg",
        "jpeg75.stack/104448.26624.jpeg",
        "jpeg75.stack/104448.28672.jpeg",
        "jpeg75.stack/104448.30720.jpeg",
        "jpeg75.stack/104448.32768.jpeg",
        "jpeg75.stack/104448.34816.jpeg",
        "jpeg75.stack/104448.36864.jpeg",
        "jpeg75.stack/104448.38912.jpeg",
        "jpeg75.stack/104448.40960.jpeg",
        "jpeg75.stack/104448.4096.jpeg",
        "jpeg75.stack/104448.43008.jpeg",
        "jpeg75.stack/104448.45056.jpeg",
        "jpeg75.stack/104448.47104.jpeg",
        "jpeg75.stack/104448.49152.jpeg",
        "jpeg75.stack/104448.51200.jpeg",
        "jpeg75.stack/104448.53248.jpeg",
        "jpeg75.stack/104448.55296.jpeg",
        "jpeg75.stack/104448.57344.jpeg",
        "jpeg75.stack/104448.59392.jpeg",
        "jpeg75.stack/104448.61440.jpeg",
        "jpeg75.stack/104448.6144.jpeg",
        "jpeg75.stack/104448.63488.jpeg",
        "jpeg75.stack/104448.65536.jpeg",
        "jpeg75.stack/104448.67584.jpeg",
        "jpeg75.stack/104448.69632.jpeg",
        "jpeg75.stack/104448.71680.jpeg",
        "jpeg75.stack/104448.73728.jpeg",
        "jpeg75.stack/104448.75776.jpeg",
        "jpeg75.stack/104448.77824.jpeg",
        "jpeg75.stack/104448.79872.jpeg",
        "jpeg75.stack/104448.81920.jpeg",
        "jpeg75.stack/104448.8192.jpeg",
        "jpeg75.stack/106496.0.jpeg",
        "jpeg75.stack/106496.10240.jpeg",
        "jpeg75.stack/106496.12288.jpeg",
        "jpeg75.stack/106496.14336.jpeg",
        "jpeg75.stack/106496.16384.jpeg",
        "jpeg75.stack/106496.18432.jpeg",
        "jpeg75.stack/106496.20480.jpeg",
        "jpeg75.stack/106496.2048.jpeg",
        "jpeg75.stack/106496.22528.jpeg",
        "jpeg75.stack/106496.24576.jpeg",
        "jpeg75.stack/106496.26624.jpeg",
        "jpeg75.stack/106496.28672.jpeg",
        "jpeg75.stack/106496.30720.jpeg",
        "jpeg75.stack/106496.32768.jpeg",
        "jpeg75.stack/106496.34816.jpeg",
        "jpeg75.stack/106496.36864.jpeg",
        "jpeg75.stack/106496.38912.jpeg",
        "jpeg75.stack/106496.40960.jpeg",
        "jpeg75.stack/106496.4096.jpeg",
        "jpeg75.stack/106496.43008.jpeg",
        "jpeg75.stack/106496.45056.jpeg",
        "jpeg75.stack/106496.47104.jpeg",
        "jpeg75.stack/106496.49152.jpeg",
        "jpeg75.stack/106496.51200.jpeg",
        "jpeg75.stack/106496.53248.jpeg",
        "jpeg75.stack/106496.55296.jpeg",
        "jpeg75.stack/106496.57344.jpeg",
        "jpeg75.stack/106496.59392.jpeg",
        "jpeg75.stack/106496.61440.jpeg",
        "jpeg75.stack/106496.6144.jpeg",
        "jpeg75.stack/106496.63488.jpeg",
        "jpeg75.stack/106496.65536.jpeg",
        "jpeg75.stack/106496.67584.jpeg",
        "jpeg75.stack/106496.69632.jpeg",
        "jpeg75.stack/106496.71680.jpeg",
        "jpeg75.stack/106496.73728.jpeg",
        "jpeg75.stack/106496.75776.jpeg",
        "jpeg75.stack/106496.77824.jpeg",
        "jpeg75.stack/106496.79872.jpeg",
        "jpeg75.stack/106496.81920.jpeg",
        "jpeg75.stack/106496.8192.jpeg",
        "jpeg75.stack/108544.0.jpeg",
        "jpeg75.stack/108544.10240.jpeg",
        "jpeg75.stack/108544.12288.jpeg",
        "jpeg75.stack/108544.14336.jpeg",
        "jpeg75.stack/108544.16384.jpeg",
        "jpeg75.stack/108544.18432.jpeg",
        "jpeg75.stack/108544.20480.jpeg",
        "jpeg75.stack/108544.2048.jpeg",
        "jpeg75.stack/108544.22528.jpeg",
        "jpeg75.stack/108544.24576.jpeg",
        "jpeg75.stack/108544.26624.jpeg",
        "jpeg75.stack/108544.28672.jpeg",
        "jpeg75.stack/108544.30720.jpeg",
        "jpeg75.stack/108544.32768.jpeg",
        "jpeg75.stack/108544.34816.jpeg",
        "jpeg75.stack/108544.36864.jpeg",
        "jpeg75.stack/108544.38912.jpeg",
        "jpeg75.stack/108544.40960.jpeg",
        "jpeg75.stack/108544.4096.jpeg",
        "jpeg75.stack/108544.43008.jpeg",
        "jpeg75.stack/108544.45056.jpeg",
        "jpeg75.stack/108544.47104.jpeg",
        "jpeg75.stack/108544.49152.jpeg",
        "jpeg75.stack/108544.51200.jpeg",
        "jpeg75.stack/108544.53248.jpeg",
        "jpeg75.stack/108544.55296.jpeg",
        "jpeg75.stack/108544.57344.jpeg",
        "jpeg75.stack/108544.59392.jpeg",
        "jpeg75.stack/108544.61440.jpeg",
        "jpeg75.stack/108544.6144.jpeg",
        "jpeg75.stack/108544.63488.jpeg",
        "jpeg75.stack/108544.65536.jpeg",
        "jpeg75.stack/108544.67584.jpeg",
        "jpeg75.stack/108544.69632.jpeg",
        "jpeg75.stack/108544.71680.jpeg",
        "jpeg75.stack/108544.73728.jpeg",
        "jpeg75.stack/108544.75776.jpeg",
        "jpeg75.stack/108544.77824.jpeg",
        "jpeg75.stack/108544.79872.jpeg",
        "jpeg75.stack/108544.81920.jpeg",
        "jpeg75.stack/108544.8192.jpeg",
        "jpeg75.stack/12288.0.jpeg",
        "jpeg75.stack/12288.10240.jpeg",
        "jpeg75.stack/12288.12288.jpeg",
        "jpeg75.stack/12288.14336.jpeg",
        "jpeg75.stack/12288.16384.jpeg",
        "jpeg75.stack/12288.18432.jpeg",
        "jpeg75.stack/12288.20480.jpeg",
        "jpeg75.stack/12288.2048.jpeg",
        "jpeg75.stack/12288.22528.jpeg",
        "jpeg75.stack/12288.24576.jpeg",
        "jpeg75.stack/12288.26624.jpeg",
        "jpeg75.stack/12288.28672.jpeg",
        "jpeg75.stack/12288.30720.jpeg",
        "jpeg75.stack/12288.32768.jpeg",
        "jpeg75.stack/12288.34816.jpeg",
        "jpeg75.stack/12288.36864.jpeg",
        "jpeg75.stack/12288.38912.jpeg",
        "jpeg75.stack/12288.40960.jpeg",
        "jpeg75.stack/12288.4096.jpeg",
        "jpeg75.stack/12288.43008.jpeg",
        "jpeg75.stack/12288.45056.jpeg",
        "jpeg75.stack/12288.47104.jpeg",
        "jpeg75.stack/12288.49152.jpeg",
        "jpeg75.stack/12288.51200.jpeg",
        "jpeg75.stack/12288.53248.jpeg",
        "jpeg75.stack/12288.55296.jpeg",
        "jpeg75.stack/12288.57344.jpeg",
        "jpeg75.stack/12288.59392.jpeg",
        "jpeg75.stack/12288.61440.jpeg",
        "jpeg75.stack/12288.6144.jpeg",
        "jpeg75.stack/12288.63488.jpeg",
        "jpeg75.stack/12288.65536.jpeg",
        "jpeg75.stack/12288.67584.jpeg",
        "jpeg75.stack/12288.69632.jpeg",
        "jpeg75.stack/12288.71680.jpeg",
        "jpeg75.stack/12288.73728.jpeg",
        "jpeg75.stack/12288.75776.jpeg",
        "jpeg75.stack/12288.77824.jpeg",
        "jpeg75.stack/12288.79872.jpeg",
        "jpeg75.stack/12288.81920.jpeg",
        "jpeg75.stack/12288.8192.jpeg",
        "jpeg75.stack/14336.0.jpeg",
        "jpeg75.stack/14336.10240.jpeg",
        "jpeg75.stack/14336.12288.jpeg",
        "jpeg75.stack/14336.14336.jpeg",
        "jpeg75.stack/14336.16384.jpeg",
        "jpeg75.stack/14336.18432.jpeg",
        "jpeg75.stack/14336.20480.jpeg",
        "jpeg75.stack/14336.2048.jpeg",
        "jpeg75.stack/14336.22528.jpeg",
        "jpeg75.stack/14336.24576.jpeg",
        "jpeg75.stack/14336.26624.jpeg",
        "jpeg75.stack/14336.28672.jpeg",
        "jpeg75.stack/14336.30720.jpeg",
        "jpeg75.stack/14336.32768.jpeg",
        "jpeg75.stack/14336.34816.jpeg",
        "jpeg75.stack/14336.36864.jpeg",
        "jpeg75.stack/14336.38912.jpeg",
        "jpeg75.stack/14336.40960.jpeg",
        "jpeg75.stack/14336.4096.jpeg",
        "jpeg75.stack/14336.43008.jpeg",
        "jpeg75.stack/14336.45056.jpeg",
        "jpeg75.stack/14336.47104.jpeg",
        "jpeg75.stack/14336.49152.jpeg",
        "jpeg75.stack/14336.51200.jpeg",
        "jpeg75.stack/14336.53248.jpeg",
        "jpeg75.stack/14336.55296.jpeg",
        "jpeg75.stack/14336.57344.jpeg",
        "jpeg75.stack/14336.59392.jpeg",
        "jpeg75.stack/14336.61440.jpeg",
        "jpeg75.stack/14336.6144.jpeg",
        "jpeg75.stack/14336.63488.jpeg",
        "jpeg75.stack/14336.65536.jpeg",
        "jpeg75.stack/14336.67584.jpeg",
        "jpeg75.stack/14336.69632.jpeg",
        "jpeg75.stack/14336.71680.jpeg",
        "jpeg75.stack/14336.73728.jpeg",
        "jpeg75.stack/14336.75776.jpeg",
        "jpeg75.stack/14336.77824.jpeg",
        "jpeg75.stack/14336.79872.jpeg",
        "jpeg75.stack/14336.81920.jpeg",
        "jpeg75.stack/14336.8192.jpeg",
        "jpeg75.stack/16384.0.jpeg",
        "jpeg75.stack/16384.10240.jpeg",
        "jpeg75.stack/16384.12288.jpeg",
        "jpeg75.stack/16384.14336.jpeg",
        "jpeg75.stack/16384.16384.jpeg",
        "jpeg75.stack/16384.18432.jpeg",
        "jpeg75.stack/16384.20480.jpeg",
        "jpeg75.stack/16384.2048.jpeg",
        "jpeg75.stack/16384.22528.jpeg",
        "jpeg75.stack/16384.24576.jpeg",
        "jpeg75.stack/16384.26624.jpeg",
        "jpeg75.stack/16384.28672.jpeg",
        "jpeg75.stack/16384.30720.jpeg",
        "jpeg75.stack/16384.32768.jpeg",
        "jpeg75.stack/16384.34816.jpeg",
        "jpeg75.stack/16384.36864.jpeg",
        "jpeg75.stack/16384.38912.jpeg",
        "jpeg75.stack/16384.40960.jpeg",
        "jpeg75.stack/16384.4096.jpeg",
        "jpeg75.stack/16384.43008.jpeg",
        "jpeg75.stack/16384.45056.jpeg",
        "jpeg75.stack/16384.47104.jpeg",
        "jpeg75.stack/16384.49152.jpeg",
        "jpeg75.stack/16384.51200.jpeg",
        "jpeg75.stack/16384.53248.jpeg",
        "jpeg75.stack/16384.55296.jpeg",
        "jpeg75.stack/16384.57344.jpeg",
        "jpeg75.stack/16384.59392.jpeg",
        "jpeg75.stack/16384.61440.jpeg",
        "jpeg75.stack/16384.6144.jpeg",
        "jpeg75.stack/16384.63488.jpeg",
        "jpeg75.stack/16384.65536.jpeg",
        "jpeg75.stack/16384.67584.jpeg",
        "jpeg75.stack/16384.69632.jpeg",
        "jpeg75.stack/16384.71680.jpeg",
        "jpeg75.stack/16384.73728.jpeg",
        "jpeg75.stack/16384.75776.jpeg",
        "jpeg75.stack/16384.77824.jpeg",
        "jpeg75.stack/16384.79872.jpeg",
        "jpeg75.stack/16384.81920.jpeg",
        "jpeg75.stack/16384.8192.jpeg",
        "jpeg75.stack/18432.0.jpeg",
        "jpeg75.stack/18432.10240.jpeg",
        "jpeg75.stack/18432.12288.jpeg",
        "jpeg75.stack/18432.14336.jpeg",
        "jpeg75.stack/18432.16384.jpeg",
        "jpeg75.stack/18432.18432.jpeg",
        "jpeg75.stack/18432.20480.jpeg",
        "jpeg75.stack/18432.2048.jpeg",
        "jpeg75.stack/18432.22528.jpeg",
        "jpeg75.stack/18432.24576.jpeg",
        "jpeg75.stack/18432.26624.jpeg",
        "jpeg75.stack/18432.28672.jpeg",
        "jpeg75.stack/18432.30720.jpeg",
        "jpeg75.stack/18432.32768.jpeg",
        "jpeg75.stack/18432.34816.jpeg",
        "jpeg75.stack/18432.36864.jpeg",
        "jpeg75.stack/18432.38912.jpeg",
        "jpeg75.stack/18432.40960.jpeg",
        "jpeg75.stack/18432.4096.jpeg",
        "jpeg75.stack/18432.43008.jpeg",
        "jpeg75.stack/18432.45056.jpeg",
        "jpeg75.stack/18432.47104.jpeg",
        "jpeg75.stack/18432.49152.jpeg",
        "jpeg75.stack/18432.51200.jpeg",
        "jpeg75.stack/18432.53248.jpeg",
        "jpeg75.stack/18432.55296.jpeg",
        "jpeg75.stack/18432.57344.jpeg",
        "jpeg75.stack/18432.59392.jpeg",
        "jpeg75.stack/18432.61440.jpeg",
        "jpeg75.stack/18432.6144.jpeg",
        "jpeg75.stack/18432.63488.jpeg",
        "jpeg75.stack/18432.65536.jpeg",
        "jpeg75.stack/18432.67584.jpeg",
        "jpeg75.stack/18432.69632.jpeg",
        "jpeg75.stack/18432.71680.jpeg",
        "jpeg75.stack/18432.73728.jpeg",
        "jpeg75.stack/18432.75776.jpeg",
        "jpeg75.stack/18432.77824.jpeg",
        "jpeg75.stack/18432.79872.jpeg",
        "jpeg75.stack/18432.81920.jpeg",
        "jpeg75.stack/18432.8192.jpeg",
        "jpeg75.stack/20480.0.jpeg",
        "jpeg75.stack/20480.10240.jpeg",
        "jpeg75.stack/20480.12288.jpeg",
        "jpeg75.stack/20480.14336.jpeg",
        "jpeg75.stack/20480.16384.jpeg",
        "jpeg75.stack/20480.18432.jpeg",
        "jpeg75.stack/20480.20480.jpeg",
        "jpeg75.stack/20480.2048.jpeg",
        "jpeg75.stack/20480.22528.jpeg",
        "jpeg75.stack/20480.24576.jpeg",
        "jpeg75.stack/20480.26624.jpeg",
        "jpeg75.stack/20480.28672.jpeg",
        "jpeg75.stack/20480.30720.jpeg",
        "jpeg75.stack/20480.32768.jpeg",
        "jpeg75.stack/20480.34816.jpeg",
        "jpeg75.stack/20480.36864.jpeg",
        "jpeg75.stack/20480.38912.jpeg",
        "jpeg75.stack/20480.40960.jpeg",
        "jpeg75.stack/20480.4096.jpeg",
        "jpeg75.stack/20480.43008.jpeg",
        "jpeg75.stack/20480.45056.jpeg",
        "jpeg75.stack/20480.47104.jpeg",
        "jpeg75.stack/20480.49152.jpeg",
        "jpeg75.stack/20480.51200.jpeg",
        "jpeg75.stack/20480.53248.jpeg",
        "jpeg75.stack/20480.55296.jpeg",
        "jpeg75.stack/20480.57344.jpeg",
        "jpeg75.stack/20480.59392.jpeg",
        "jpeg75.stack/20480.61440.jpeg",
        "jpeg75.stack/20480.6144.jpeg",
        "jpeg75.stack/20480.63488.jpeg",
        "jpeg75.stack/20480.65536.jpeg",
        "jpeg75.stack/20480.67584.jpeg",
        "jpeg75.stack/20480.69632.jpeg",
        "jpeg75.stack/20480.71680.jpeg",
        "jpeg75.stack/20480.73728.jpeg",
        "jpeg75.stack/20480.75776.jpeg",
        "jpeg75.stack/20480.77824.jpeg",
        "jpeg75.stack/20480.79872.jpeg",
        "jpeg75.stack/20480.81920.jpeg",
        "jpeg75.stack/20480.8192.jpeg",
        "jpeg75.stack/2048.0.jpeg",
        "jpeg75.stack/2048.10240.jpeg",
        "jpeg75.stack/2048.12288.jpeg",
        "jpeg75.stack/2048.14336.jpeg",
        "jpeg75.stack/2048.16384.jpeg",
        "jpeg75.stack/2048.18432.jpeg",
        "jpeg75.stack/2048.20480.jpeg",
        "jpeg75.stack/2048.2048.jpeg",
        "jpeg75.stack/2048.22528.jpeg",
        "jpeg75.stack/2048.24576.jpeg",
        "jpeg75.stack/2048.26624.jpeg",
        "jpeg75.stack/2048.28672.jpeg",
        "jpeg75.stack/2048.30720.jpeg",
        "jpeg75.stack/2048.32768.jpeg",
        "jpeg75.stack/2048.34816.jpeg",
        "jpeg75.stack/2048.36864.jpeg",
        "jpeg75.stack/2048.38912.jpeg",
        "jpeg75.stack/2048.40960.jpeg",
        "jpeg75.stack/2048.4096.jpeg",
        "jpeg75.stack/2048.43008.jpeg",
        "jpeg75.stack/2048.45056.jpeg",
        "jpeg75.stack/2048.47104.jpeg",
        "jpeg75.stack/2048.49152.jpeg",
        "jpeg75.stack/2048.51200.jpeg",
        "jpeg75.stack/2048.53248.jpeg",
        "jpeg75.stack/2048.55296.jpeg",
        "jpeg75.stack/2048.57344.jpeg",
        "jpeg75.stack/2048.59392.jpeg",
        "jpeg75.stack/2048.61440.jpeg",
        "jpeg75.stack/2048.6144.jpeg",
        "jpeg75.stack/2048.63488.jpeg",
        "jpeg75.stack/2048.65536.jpeg",
        "jpeg75.stack/2048.67584.jpeg",
        "jpeg75.stack/2048.69632.jpeg",
        "jpeg75.stack/2048.71680.jpeg",
        "jpeg75.stack/2048.73728.jpeg",
        "jpeg75.stack/2048.75776.jpeg",
        "jpeg75.stack/2048.77824.jpeg",
        "jpeg75.stack/2048.79872.jpeg",
        "jpeg75.stack/2048.81920.jpeg",
        "jpeg75.stack/2048.8192.jpeg",
        "jpeg75.stack/22528.0.jpeg",
        "jpeg75.stack/22528.10240.jpeg",
        "jpeg75.stack/22528.12288.jpeg",
        "jpeg75.stack/22528.14336.jpeg",
        "jpeg75.stack/22528.16384.jpeg",
        "jpeg75.stack/22528.18432.jpeg",
        "jpeg75.stack/22528.20480.jpeg",
        "jpeg75.stack/22528.2048.jpeg",
        "jpeg75.stack/22528.22528.jpeg",
        "jpeg75.stack/22528.24576.jpeg",
        "jpeg75.stack/22528.26624.jpeg",
        "jpeg75.stack/22528.28672.jpeg",
        "jpeg75.stack/22528.30720.jpeg",
        "jpeg75.stack/22528.32768.jpeg",
        "jpeg75.stack/22528.34816.jpeg",
        "jpeg75.stack/22528.36864.jpeg",
        "jpeg75.stack/22528.38912.jpeg",
        "jpeg75.stack/22528.40960.jpeg",
        "jpeg75.stack/22528.4096.jpeg",
        "jpeg75.stack/22528.43008.jpeg",
        "jpeg75.stack/22528.45056.jpeg",
        "jpeg75.stack/22528.47104.jpeg",
        "jpeg75.stack/22528.49152.jpeg",
        "jpeg75.stack/22528.51200.jpeg",
        "jpeg75.stack/22528.53248.jpeg",
        "jpeg75.stack/22528.55296.jpeg",
        "jpeg75.stack/22528.57344.jpeg",
        "jpeg75.stack/22528.59392.jpeg",
        "jpeg75.stack/22528.61440.jpeg",
        "jpeg75.stack/22528.6144.jpeg",
        "jpeg75.stack/22528.63488.jpeg",
        "jpeg75.stack/22528.65536.jpeg",
        "jpeg75.stack/22528.67584.jpeg",
        "jpeg75.stack/22528.69632.jpeg",
        "jpeg75.stack/22528.71680.jpeg",
        "jpeg75.stack/22528.73728.jpeg",
        "jpeg75.stack/22528.75776.jpeg",
        "jpeg75.stack/22528.77824.jpeg",
        "jpeg75.stack/22528.79872.jpeg",
        "jpeg75.stack/22528.81920.jpeg",
        "jpeg75.stack/22528.8192.jpeg",
        "jpeg75.stack/24576.0.jpeg",
        "jpeg75.stack/24576.10240.jpeg",
        "jpeg75.stack/24576.12288.jpeg",
        "jpeg75.stack/24576.14336.jpeg",
        "jpeg75.stack/24576.16384.jpeg",
        "jpeg75.stack/24576.18432.jpeg",
        "jpeg75.stack/24576.20480.jpeg",
        "jpeg75.stack/24576.2048.jpeg",
        "jpeg75.stack/24576.22528.jpeg",
        "jpeg75.stack/24576.24576.jpeg",
        "jpeg75.stack/24576.26624.jpeg",
        "jpeg75.stack/24576.28672.jpeg",
        "jpeg75.stack/24576.30720.jpeg",
        "jpeg75.stack/24576.32768.jpeg",
        "jpeg75.stack/24576.34816.jpeg",
        "jpeg75.stack/24576.36864.jpeg",
        "jpeg75.stack/24576.38912.jpeg",
        "jpeg75.stack/24576.40960.jpeg",
        "jpeg75.stack/24576.4096.jpeg",
        "jpeg75.stack/24576.43008.jpeg",
        "jpeg75.stack/24576.45056.jpeg",
        "jpeg75.stack/24576.47104.jpeg",
        "jpeg75.stack/24576.49152.jpeg",
        "jpeg75.stack/24576.51200.jpeg",
        "jpeg75.stack/24576.53248.jpeg",
        "jpeg75.stack/24576.55296.jpeg",
        "jpeg75.stack/24576.57344.jpeg",
        "jpeg75.stack/24576.59392.jpeg",
        "jpeg75.stack/24576.61440.jpeg",
        "jpeg75.stack/24576.6144.jpeg",
        "jpeg75.stack/24576.63488.jpeg",
        "jpeg75.stack/24576.65536.jpeg",
        "jpeg75.stack/24576.67584.jpeg",
        "jpeg75.stack/24576.69632.jpeg",
        "jpeg75.stack/24576.71680.jpeg",
        "jpeg75.stack/24576.73728.jpeg",
        "jpeg75.stack/24576.75776.jpeg",
        "jpeg75.stack/24576.77824.jpeg",
        "jpeg75.stack/24576.79872.jpeg",
        "jpeg75.stack/24576.81920.jpeg",
        "jpeg75.stack/24576.8192.jpeg",
        "jpeg75.stack/26624.0.jpeg",
        "jpeg75.stack/26624.10240.jpeg",
        "jpeg75.stack/26624.12288.jpeg",
        "jpeg75.stack/26624.14336.jpeg",
        "jpeg75.stack/26624.16384.jpeg",
        "jpeg75.stack/26624.18432.jpeg",
        "jpeg75.stack/26624.20480.jpeg",
        "jpeg75.stack/26624.2048.jpeg",
        "jpeg75.stack/26624.22528.jpeg",
        "jpeg75.stack/26624.24576.jpeg",
        "jpeg75.stack/26624.26624.jpeg",
        "jpeg75.stack/26624.28672.jpeg",
        "jpeg75.stack/26624.30720.jpeg",
        "jpeg75.stack/26624.32768.jpeg",
        "jpeg75.stack/26624.34816.jpeg",
        "jpeg75.stack/26624.36864.jpeg",
        "jpeg75.stack/26624.38912.jpeg",
        "jpeg75.stack/26624.40960.jpeg",
        "jpeg75.stack/26624.4096.jpeg",
        "jpeg75.stack/26624.43008.jpeg",
        "jpeg75.stack/26624.45056.jpeg",
        "jpeg75.stack/26624.47104.jpeg",
        "jpeg75.stack/26624.49152.jpeg",
        "jpeg75.stack/26624.51200.jpeg",
        "jpeg75.stack/26624.53248.jpeg",
        "jpeg75.stack/26624.55296.jpeg",
        "jpeg75.stack/26624.57344.jpeg",
        "jpeg75.stack/26624.59392.jpeg",
        "jpeg75.stack/26624.61440.jpeg",
        "jpeg75.stack/26624.6144.jpeg",
        "jpeg75.stack/26624.63488.jpeg",
        "jpeg75.stack/26624.65536.jpeg",
        "jpeg75.stack/26624.67584.jpeg",
        "jpeg75.stack/26624.69632.jpeg",
        "jpeg75.stack/26624.71680.jpeg",
        "jpeg75.stack/26624.73728.jpeg",
        "jpeg75.stack/26624.75776.jpeg",
        "jpeg75.stack/26624.77824.jpeg",
        "jpeg75.stack/26624.79872.jpeg",
        "jpeg75.stack/26624.81920.jpeg",
        "jpeg75.stack/26624.8192.jpeg",
        "jpeg75.stack/28672.0.jpeg",
        "jpeg75.stack/28672.10240.jpeg",
        "jpeg75.stack/28672.12288.jpeg",
        "jpeg75.stack/28672.14336.jpeg",
        "jpeg75.stack/28672.16384.jpeg",
        "jpeg75.stack/28672.18432.jpeg",
        "jpeg75.stack/28672.20480.jpeg",
        "jpeg75.stack/28672.2048.jpeg",
        "jpeg75.stack/28672.22528.jpeg",
        "jpeg75.stack/28672.24576.jpeg",
        "jpeg75.stack/28672.26624.jpeg",
        "jpeg75.stack/28672.28672.jpeg",
        "jpeg75.stack/28672.30720.jpeg",
        "jpeg75.stack/28672.32768.jpeg",
        "jpeg75.stack/28672.34816.jpeg",
        "jpeg75.stack/28672.36864.jpeg",
        "jpeg75.stack/28672.38912.jpeg",
        "jpeg75.stack/28672.40960.jpeg",
        "jpeg75.stack/28672.4096.jpeg",
        "jpeg75.stack/28672.43008.jpeg",
        "jpeg75.stack/28672.45056.jpeg",
        "jpeg75.stack/28672.47104.jpeg",
        "jpeg75.stack/28672.49152.jpeg",
        "jpeg75.stack/28672.51200.jpeg",
        "jpeg75.stack/28672.53248.jpeg",
        "jpeg75.stack/28672.55296.jpeg",
        "jpeg75.stack/28672.57344.jpeg",
        "jpeg75.stack/28672.59392.jpeg",
        "jpeg75.stack/28672.61440.jpeg",
        "jpeg75.stack/28672.6144.jpeg",
        "jpeg75.stack/28672.63488.jpeg",
        "jpeg75.stack/28672.65536.jpeg",
        "jpeg75.stack/28672.67584.jpeg",
        "jpeg75.stack/28672.69632.jpeg",
        "jpeg75.stack/28672.71680.jpeg",
        "jpeg75.stack/28672.73728.jpeg",
        "jpeg75.stack/28672.75776.jpeg",
        "jpeg75.stack/28672.77824.jpeg",
        "jpeg75.stack/28672.79872.jpeg",
        "jpeg75.stack/28672.81920.jpeg",
        "jpeg75.stack/28672.8192.jpeg",
        "jpeg75.stack/30720.0.jpeg",
        "jpeg75.stack/30720.10240.jpeg",
        "jpeg75.stack/30720.12288.jpeg",
        "jpeg75.stack/30720.14336.jpeg",
        "jpeg75.stack/30720.16384.jpeg",
        "jpeg75.stack/30720.18432.jpeg",
        "jpeg75.stack/30720.20480.jpeg",
        "jpeg75.stack/30720.2048.jpeg",
        "jpeg75.stack/30720.22528.jpeg",
        "jpeg75.stack/30720.24576.jpeg",
        "jpeg75.stack/30720.26624.jpeg",
        "jpeg75.stack/30720.28672.jpeg",
        "jpeg75.stack/30720.30720.jpeg",
        "jpeg75.stack/30720.32768.jpeg",
        "jpeg75.stack/30720.34816.jpeg",
        "jpeg75.stack/30720.36864.jpeg",
        "jpeg75.stack/30720.38912.jpeg",
        "jpeg75.stack/30720.40960.jpeg",
        "jpeg75.stack/30720.4096.jpeg",
        "jpeg75.stack/30720.43008.jpeg",
        "jpeg75.stack/30720.45056.jpeg",
        "jpeg75.stack/30720.47104.jpeg",
        "jpeg75.stack/30720.49152.jpeg",
        "jpeg75.stack/30720.51200.jpeg",
        "jpeg75.stack/30720.53248.jpeg",
        "jpeg75.stack/30720.55296.jpeg",
        "jpeg75.stack/30720.57344.jpeg",
        "jpeg75.stack/30720.59392.jpeg",
        "jpeg75.stack/30720.61440.jpeg",
        "jpeg75.stack/30720.6144.jpeg",
        "jpeg75.stack/30720.63488.jpeg",
        "jpeg75.stack/30720.65536.jpeg",
        "jpeg75.stack/30720.67584.jpeg",
        "jpeg75.stack/30720.69632.jpeg",
        "jpeg75.stack/30720.71680.jpeg",
        "jpeg75.stack/30720.73728.jpeg",
        "jpeg75.stack/30720.75776.jpeg",
        "jpeg75.stack/30720.77824.jpeg",
        "jpeg75.stack/30720.79872.jpeg",
        "jpeg75.stack/30720.81920.jpeg",
        "jpeg75.stack/30720.8192.jpeg",
        "jpeg75.stack/32768.0.jpeg",
        "jpeg75.stack/32768.10240.jpeg",
        "jpeg75.stack/32768.12288.jpeg",
        "jpeg75.stack/32768.14336.jpeg",
        "jpeg75.stack/32768.16384.jpeg",
        "jpeg75.stack/32768.18432.jpeg",
        "jpeg75.stack/32768.20480.jpeg",
        "jpeg75.stack/32768.2048.jpeg",
        "jpeg75.stack/32768.22528.jpeg",
        "jpeg75.stack/32768.24576.jpeg",
        "jpeg75.stack/32768.26624.jpeg",
        "jpeg75.stack/32768.28672.jpeg",
        "jpeg75.stack/32768.30720.jpeg",
        "jpeg75.stack/32768.32768.jpeg",
        "jpeg75.stack/32768.34816.jpeg",
        "jpeg75.stack/32768.36864.jpeg",
        "jpeg75.stack/32768.38912.jpeg",
        "jpeg75.stack/32768.40960.jpeg",
        "jpeg75.stack/32768.4096.jpeg",
        "jpeg75.stack/32768.43008.jpeg",
        "jpeg75.stack/32768.45056.jpeg",
        "jpeg75.stack/32768.47104.jpeg",
        "jpeg75.stack/32768.49152.jpeg",
        "jpeg75.stack/32768.51200.jpeg",
        "jpeg75.stack/32768.53248.jpeg",
        "jpeg75.stack/32768.55296.jpeg",
        "jpeg75.stack/32768.57344.jpeg",
        "jpeg75.stack/32768.59392.jpeg",
        "jpeg75.stack/32768.61440.jpeg",
        "jpeg75.stack/32768.6144.jpeg",
        "jpeg75.stack/32768.63488.jpeg",
        "jpeg75.stack/32768.65536.jpeg",
        "jpeg75.stack/32768.67584.jpeg",
        "jpeg75.stack/32768.69632.jpeg",
        "jpeg75.stack/32768.71680.jpeg",
        "jpeg75.stack/32768.73728.jpeg",
        "jpeg75.stack/32768.75776.jpeg",
        "jpeg75.stack/32768.77824.jpeg",
        "jpeg75.stack/32768.79872.jpeg",
        "jpeg75.stack/32768.81920.jpeg",
        "jpeg75.stack/32768.8192.jpeg",
        "jpeg75.stack/34816.0.jpeg",
        "jpeg75.stack/34816.10240.jpeg",
        "jpeg75.stack/34816.12288.jpeg",
        "jpeg75.stack/34816.14336.jpeg",
        "jpeg75.stack/34816.16384.jpeg",
        "jpeg75.stack/34816.18432.jpeg",
        "jpeg75.stack/34816.20480.jpeg",
        "jpeg75.stack/34816.2048.jpeg",
        "jpeg75.stack/34816.22528.jpeg",
        "jpeg75.stack/34816.24576.jpeg",
        "jpeg75.stack/34816.26624.jpeg",
        "jpeg75.stack/34816.28672.jpeg",
        "jpeg75.stack/34816.30720.jpeg",
        "jpeg75.stack/34816.32768.jpeg",
        "jpeg75.stack/34816.34816.jpeg",
        "jpeg75.stack/34816.36864.jpeg",
        "jpeg75.stack/34816.38912.jpeg",
        "jpeg75.stack/34816.40960.jpeg",
        "jpeg75.stack/34816.4096.jpeg",
        "jpeg75.stack/34816.43008.jpeg",
        "jpeg75.stack/34816.45056.jpeg",
        "jpeg75.stack/34816.47104.jpeg",
        "jpeg75.stack/34816.49152.jpeg",
        "jpeg75.stack/34816.51200.jpeg",
        "jpeg75.stack/34816.53248.jpeg",
        "jpeg75.stack/34816.55296.jpeg",
        "jpeg75.stack/34816.57344.jpeg",
        "jpeg75.stack/34816.59392.jpeg",
        "jpeg75.stack/34816.61440.jpeg",
        "jpeg75.stack/34816.6144.jpeg",
        "jpeg75.stack/34816.63488.jpeg",
        "jpeg75.stack/34816.65536.jpeg",
        "jpeg75.stack/34816.67584.jpeg",
        "jpeg75.stack/34816.69632.jpeg",
        "jpeg75.stack/34816.71680.jpeg",
        "jpeg75.stack/34816.73728.jpeg",
        "jpeg75.stack/34816.75776.jpeg",
        "jpeg75.stack/34816.77824.jpeg",
        "jpeg75.stack/34816.79872.jpeg",
        "jpeg75.stack/34816.81920.jpeg",
        "jpeg75.stack/34816.8192.jpeg",
        "jpeg75.stack/36864.0.jpeg",
        "jpeg75.stack/36864.10240.jpeg",
        "jpeg75.stack/36864.12288.jpeg",
        "jpeg75.stack/36864.14336.jpeg",
        "jpeg75.stack/36864.16384.jpeg",
        "jpeg75.stack/36864.18432.jpeg",
        "jpeg75.stack/36864.20480.jpeg",
        "jpeg75.stack/36864.2048.jpeg",
        "jpeg75.stack/36864.22528.jpeg",
        "jpeg75.stack/36864.24576.jpeg",
        "jpeg75.stack/36864.26624.jpeg",
        "jpeg75.stack/36864.28672.jpeg",
        "jpeg75.stack/36864.30720.jpeg",
        "jpeg75.stack/36864.32768.jpeg",
        "jpeg75.stack/36864.34816.jpeg",
        "jpeg75.stack/36864.36864.jpeg",
        "jpeg75.stack/36864.38912.jpeg",
        "jpeg75.stack/36864.40960.jpeg",
        "jpeg75.stack/36864.4096.jpeg",
        "jpeg75.stack/36864.43008.jpeg",
        "jpeg75.stack/36864.45056.jpeg",
        "jpeg75.stack/36864.47104.jpeg",
        "jpeg75.stack/36864.49152.jpeg",
        "jpeg75.stack/36864.51200.jpeg",
        "jpeg75.stack/36864.53248.jpeg",
        "jpeg75.stack/36864.55296.jpeg",
        "jpeg75.stack/36864.57344.jpeg",
        "jpeg75.stack/36864.59392.jpeg",
        "jpeg75.stack/36864.61440.jpeg",
        "jpeg75.stack/36864.6144.jpeg",
        "jpeg75.stack/36864.63488.jpeg",
        "jpeg75.stack/36864.65536.jpeg",
        "jpeg75.stack/36864.67584.jpeg",
        "jpeg75.stack/36864.69632.jpeg",
        "jpeg75.stack/36864.71680.jpeg",
        "jpeg75.stack/36864.73728.jpeg",
        "jpeg75.stack/36864.75776.jpeg",
        "jpeg75.stack/36864.77824.jpeg",
        "jpeg75.stack/36864.79872.jpeg",
        "jpeg75.stack/36864.81920.jpeg",
        "jpeg75.stack/36864.8192.jpeg",
        "jpeg75.stack/38912.0.jpeg",
        "jpeg75.stack/38912.10240.jpeg",
        "jpeg75.stack/38912.12288.jpeg",
        "jpeg75.stack/38912.14336.jpeg",
        "jpeg75.stack/38912.16384.jpeg",
        "jpeg75.stack/38912.18432.jpeg",
        "jpeg75.stack/38912.20480.jpeg",
        "jpeg75.stack/38912.2048.jpeg",
        "jpeg75.stack/38912.22528.jpeg",
        "jpeg75.stack/38912.24576.jpeg",
        "jpeg75.stack/38912.26624.jpeg",
        "jpeg75.stack/38912.28672.jpeg",
        "jpeg75.stack/38912.30720.jpeg",
        "jpeg75.stack/38912.32768.jpeg",
        "jpeg75.stack/38912.34816.jpeg",
        "jpeg75.stack/38912.36864.jpeg",
        "jpeg75.stack/38912.38912.jpeg",
        "jpeg75.stack/38912.40960.jpeg",
        "jpeg75.stack/38912.4096.jpeg",
        "jpeg75.stack/38912.43008.jpeg",
        "jpeg75.stack/38912.45056.jpeg",
        "jpeg75.stack/38912.47104.jpeg",
        "jpeg75.stack/38912.49152.jpeg",
        "jpeg75.stack/38912.51200.jpeg",
        "jpeg75.stack/38912.53248.jpeg",
        "jpeg75.stack/38912.55296.jpeg",
        "jpeg75.stack/38912.57344.jpeg",
        "jpeg75.stack/38912.59392.jpeg",
        "jpeg75.stack/38912.61440.jpeg",
        "jpeg75.stack/38912.6144.jpeg",
        "jpeg75.stack/38912.63488.jpeg",
        "jpeg75.stack/38912.65536.jpeg",
        "jpeg75.stack/38912.67584.jpeg",
        "jpeg75.stack/38912.69632.jpeg",
        "jpeg75.stack/38912.71680.jpeg",
        "jpeg75.stack/38912.73728.jpeg",
        "jpeg75.stack/38912.75776.jpeg",
        "jpeg75.stack/38912.77824.jpeg",
        "jpeg75.stack/38912.79872.jpeg",
        "jpeg75.stack/38912.81920.jpeg",
        "jpeg75.stack/38912.8192.jpeg",
        "jpeg75.stack/40960.0.jpeg",
        "jpeg75.stack/40960.10240.jpeg",
        "jpeg75.stack/40960.12288.jpeg",
        "jpeg75.stack/40960.14336.jpeg",
        "jpeg75.stack/40960.16384.jpeg",
        "jpeg75.stack/40960.18432.jpeg",
        "jpeg75.stack/40960.20480.jpeg",
        "jpeg75.stack/40960.2048.jpeg",
        "jpeg75.stack/40960.22528.jpeg",
        "jpeg75.stack/40960.24576.jpeg",
        "jpeg75.stack/40960.26624.jpeg",
        "jpeg75.stack/40960.28672.jpeg",
        "jpeg75.stack/40960.30720.jpeg",
        "jpeg75.stack/40960.32768.jpeg",
        "jpeg75.stack/40960.34816.jpeg",
        "jpeg75.stack/40960.36864.jpeg",
        "jpeg75.stack/40960.38912.jpeg",
        "jpeg75.stack/40960.40960.jpeg",
        "jpeg75.stack/40960.4096.jpeg",
        "jpeg75.stack/40960.43008.jpeg",
        "jpeg75.stack/40960.45056.jpeg",
        "jpeg75.stack/40960.47104.jpeg",
        "jpeg75.stack/40960.49152.jpeg",
        "jpeg75.stack/40960.51200.jpeg",
        "jpeg75.stack/40960.53248.jpeg",
        "jpeg75.stack/40960.55296.jpeg",
        "jpeg75.stack/40960.57344.jpeg",
        "jpeg75.stack/40960.59392.jpeg",
        "jpeg75.stack/40960.61440.jpeg",
        "jpeg75.stack/40960.6144.jpeg",
        "jpeg75.stack/40960.63488.jpeg",
        "jpeg75.stack/40960.65536.jpeg",
        "jpeg75.stack/40960.67584.jpeg",
        "jpeg75.stack/40960.69632.jpeg",
        "jpeg75.stack/40960.71680.jpeg",
        "jpeg75.stack/40960.73728.jpeg",
        "jpeg75.stack/40960.75776.jpeg",
        "jpeg75.stack/40960.77824.jpeg",
        "jpeg75.stack/40960.79872.jpeg",
        "jpeg75.stack/40960.81920.jpeg",
        "jpeg75.stack/40960.8192.jpeg",
        "jpeg75.stack/4096.0.jpeg",
        "jpeg75.stack/4096.10240.jpeg",
        "jpeg75.stack/4096.12288.jpeg",
        "jpeg75.stack/4096.14336.jpeg",
        "jpeg75.stack/4096.16384.jpeg",
        "jpeg75.stack/4096.18432.jpeg",
        "jpeg75.stack/4096.20480.jpeg",
        "jpeg75.stack/4096.2048.jpeg",
        "jpeg75.stack/4096.22528.jpeg",
        "jpeg75.stack/4096.24576.jpeg",
        "jpeg75.stack/4096.26624.jpeg",
        "jpeg75.stack/4096.28672.jpeg",
        "jpeg75.stack/4096.30720.jpeg",
        "jpeg75.stack/4096.32768.jpeg",
        "jpeg75.stack/4096.34816.jpeg",
        "jpeg75.stack/4096.36864.jpeg",
        "jpeg75.stack/4096.38912.jpeg",
        "jpeg75.stack/4096.40960.jpeg",
        "jpeg75.stack/4096.4096.jpeg",
        "jpeg75.stack/4096.43008.jpeg",
        "jpeg75.stack/4096.45056.jpeg",
        "jpeg75.stack/4096.47104.jpeg",
        "jpeg75.stack/4096.49152.jpeg",
        "jpeg75.stack/4096.51200.jpeg",
        "jpeg75.stack/4096.53248.jpeg",
        "jpeg75.stack/4096.55296.jpeg",
        "jpeg75.stack/4096.57344.jpeg",
        "jpeg75.stack/4096.59392.jpeg",
        "jpeg75.stack/4096.61440.jpeg",
        "jpeg75.stack/4096.6144.jpeg",
        "jpeg75.stack/4096.63488.jpeg",
        "jpeg75.stack/4096.65536.jpeg",
        "jpeg75.stack/4096.67584.jpeg",
        "jpeg75.stack/4096.69632.jpeg",
        "jpeg75.stack/4096.71680.jpeg",
        "jpeg75.stack/4096.73728.jpeg",
        "jpeg75.stack/4096.75776.jpeg",
        "jpeg75.stack/4096.77824.jpeg",
        "jpeg75.stack/4096.79872.jpeg",
        "jpeg75.stack/4096.81920.jpeg",
        "jpeg75.stack/4096.8192.jpeg",
        "jpeg75.stack/43008.0.jpeg",
        "jpeg75.stack/43008.10240.jpeg",
        "jpeg75.stack/43008.12288.jpeg",
        "jpeg75.stack/43008.14336.jpeg",
        "jpeg75.stack/43008.16384.jpeg",
        "jpeg75.stack/43008.18432.jpeg",
        "jpeg75.stack/43008.20480.jpeg",
        "jpeg75.stack/43008.2048.jpeg",
        "jpeg75.stack/43008.22528.jpeg",
        "jpeg75.stack/43008.24576.jpeg",
        "jpeg75.stack/43008.26624.jpeg",
        "jpeg75.stack/43008.28672.jpeg",
        "jpeg75.stack/43008.30720.jpeg",
        "jpeg75.stack/43008.32768.jpeg",
        "jpeg75.stack/43008.34816.jpeg",
        "jpeg75.stack/43008.36864.jpeg",
        "jpeg75.stack/43008.38912.jpeg",
        "jpeg75.stack/43008.40960.jpeg",
        "jpeg75.stack/43008.4096.jpeg",
        "jpeg75.stack/43008.43008.jpeg",
        "jpeg75.stack/43008.45056.jpeg",
        "jpeg75.stack/43008.47104.jpeg",
        "jpeg75.stack/43008.49152.jpeg",
        "jpeg75.stack/43008.51200.jpeg",
        "jpeg75.stack/43008.53248.jpeg",
        "jpeg75.stack/43008.55296.jpeg",
        "jpeg75.stack/43008.57344.jpeg",
        "jpeg75.stack/43008.59392.jpeg",
        "jpeg75.stack/43008.61440.jpeg",
        "jpeg75.stack/43008.6144.jpeg",
        "jpeg75.stack/43008.63488.jpeg",
        "jpeg75.stack/43008.65536.jpeg",
        "jpeg75.stack/43008.67584.jpeg",
        "jpeg75.stack/43008.69632.jpeg",
        "jpeg75.stack/43008.71680.jpeg",
        "jpeg75.stack/43008.73728.jpeg",
        "jpeg75.stack/43008.75776.jpeg",
        "jpeg75.stack/43008.77824.jpeg",
        "jpeg75.stack/43008.79872.jpeg",
        "jpeg75.stack/43008.81920.jpeg",
        "jpeg75.stack/43008.8192.jpeg",
        "jpeg75.stack/45056.0.jpeg",
        "jpeg75.stack/45056.10240.jpeg",
        "jpeg75.stack/45056.12288.jpeg",
        "jpeg75.stack/45056.14336.jpeg",
        "jpeg75.stack/45056.16384.jpeg",
        "jpeg75.stack/45056.18432.jpeg",
        "jpeg75.stack/45056.20480.jpeg",
        "jpeg75.stack/45056.2048.jpeg",
        "jpeg75.stack/45056.22528.jpeg",
        "jpeg75.stack/45056.24576.jpeg",
        "jpeg75.stack/45056.26624.jpeg",
        "jpeg75.stack/45056.28672.jpeg",
        "jpeg75.stack/45056.30720.jpeg",
        "jpeg75.stack/45056.32768.jpeg",
        "jpeg75.stack/45056.34816.jpeg",
        "jpeg75.stack/45056.36864.jpeg",
        "jpeg75.stack/45056.38912.jpeg",
        "jpeg75.stack/45056.40960.jpeg",
        "jpeg75.stack/45056.4096.jpeg",
        "jpeg75.stack/45056.43008.jpeg",
        "jpeg75.stack/45056.45056.jpeg",
        "jpeg75.stack/45056.47104.jpeg",
        "jpeg75.stack/45056.49152.jpeg",
        "jpeg75.stack/45056.51200.jpeg",
        "jpeg75.stack/45056.53248.jpeg",
        "jpeg75.stack/45056.55296.jpeg",
        "jpeg75.stack/45056.57344.jpeg",
        "jpeg75.stack/45056.59392.jpeg",
        "jpeg75.stack/45056.61440.jpeg",
        "jpeg75.stack/45056.6144.jpeg",
        "jpeg75.stack/45056.63488.jpeg",
        "jpeg75.stack/45056.65536.jpeg",
        "jpeg75.stack/45056.67584.jpeg",
        "jpeg75.stack/45056.69632.jpeg",
        "jpeg75.stack/45056.71680.jpeg",
        "jpeg75.stack/45056.73728.jpeg",
        "jpeg75.stack/45056.75776.jpeg",
        "jpeg75.stack/45056.77824.jpeg",
        "jpeg75.stack/45056.79872.jpeg",
        "jpeg75.stack/45056.81920.jpeg",
        "jpeg75.stack/45056.8192.jpeg",
        "jpeg75.stack/47104.0.jpeg",
        "jpeg75.stack/47104.10240.jpeg",
        "jpeg75.stack/47104.12288.jpeg",
        "jpeg75.stack/47104.14336.jpeg",
        "jpeg75.stack/47104.16384.jpeg",
        "jpeg75.stack/47104.18432.jpeg",
        "jpeg75.stack/47104.20480.jpeg",
        "jpeg75.stack/47104.2048.jpeg",
        "jpeg75.stack/47104.22528.jpeg",
        "jpeg75.stack/47104.24576.jpeg",
        "jpeg75.stack/47104.26624.jpeg",
        "jpeg75.stack/47104.28672.jpeg",
        "jpeg75.stack/47104.30720.jpeg",
        "jpeg75.stack/47104.32768.jpeg",
        "jpeg75.stack/47104.34816.jpeg",
        "jpeg75.stack/47104.36864.jpeg",
        "jpeg75.stack/47104.38912.jpeg",
        "jpeg75.stack/47104.40960.jpeg",
        "jpeg75.stack/47104.4096.jpeg",
        "jpeg75.stack/47104.43008.jpeg",
        "jpeg75.stack/47104.45056.jpeg",
        "jpeg75.stack/47104.47104.jpeg",
        "jpeg75.stack/47104.49152.jpeg",
        "jpeg75.stack/47104.51200.jpeg",
        "jpeg75.stack/47104.53248.jpeg",
        "jpeg75.stack/47104.55296.jpeg",
        "jpeg75.stack/47104.57344.jpeg",
        "jpeg75.stack/47104.59392.jpeg",
        "jpeg75.stack/47104.61440.jpeg",
        "jpeg75.stack/47104.6144.jpeg",
        "jpeg75.stack/47104.63488.jpeg",
        "jpeg75.stack/47104.65536.jpeg",
        "jpeg75.stack/47104.67584.jpeg",
        "jpeg75.stack/47104.69632.jpeg",
        "jpeg75.stack/47104.71680.jpeg",
        "jpeg75.stack/47104.73728.jpeg",
        "jpeg75.stack/47104.75776.jpeg",
        "jpeg75.stack/47104.77824.jpeg",
        "jpeg75.stack/47104.79872.jpeg",
        "jpeg75.stack/47104.81920.jpeg",
        "jpeg75.stack/47104.8192.jpeg",
        "jpeg75.stack/49152.0.jpeg",
        "jpeg75.stack/49152.10240.jpeg",
        "jpeg75.stack/49152.12288.jpeg",
        "jpeg75.stack/49152.14336.jpeg",
        "jpeg75.stack/49152.16384.jpeg",
        "jpeg75.stack/49152.18432.jpeg",
        "jpeg75.stack/49152.20480.jpeg",
        "jpeg75.stack/49152.2048.jpeg",
        "jpeg75.stack/49152.22528.jpeg",
        "jpeg75.stack/49152.24576.jpeg",
        "jpeg75.stack/49152.26624.jpeg",
        "jpeg75.stack/49152.28672.jpeg",
        "jpeg75.stack/49152.30720.jpeg",
        "jpeg75.stack/49152.32768.jpeg",
        "jpeg75.stack/49152.34816.jpeg",
        "jpeg75.stack/49152.36864.jpeg",
        "jpeg75.stack/49152.38912.jpeg",
        "jpeg75.stack/49152.40960.jpeg",
        "jpeg75.stack/49152.4096.jpeg",
        "jpeg75.stack/49152.43008.jpeg",
        "jpeg75.stack/49152.45056.jpeg",
        "jpeg75.stack/49152.47104.jpeg",
        "jpeg75.stack/49152.49152.jpeg",
        "jpeg75.stack/49152.51200.jpeg",
        "jpeg75.stack/49152.53248.jpeg",
        "jpeg75.stack/49152.55296.jpeg",
        "jpeg75.stack/49152.57344.jpeg",
        "jpeg75.stack/49152.59392.jpeg",
        "jpeg75.stack/49152.61440.jpeg",
        "jpeg75.stack/49152.6144.jpeg",
        "jpeg75.stack/49152.63488.jpeg",
        "jpeg75.stack/49152.65536.jpeg",
        "jpeg75.stack/49152.67584.jpeg",
        "jpeg75.stack/49152.69632.jpeg",
        "jpeg75.stack/49152.71680.jpeg",
        "jpeg75.stack/49152.73728.jpeg",
        "jpeg75.stack/49152.75776.jpeg",
        "jpeg75.stack/49152.77824.jpeg",
        "jpeg75.stack/49152.79872.jpeg",
        "jpeg75.stack/49152.81920.jpeg",
        "jpeg75.stack/49152.8192.jpeg",
        "jpeg75.stack/51200.0.jpeg",
        "jpeg75.stack/51200.10240.jpeg",
        "jpeg75.stack/51200.12288.jpeg",
        "jpeg75.stack/51200.14336.jpeg",
        "jpeg75.stack/51200.16384.jpeg",
        "jpeg75.stack/51200.18432.jpeg",
        "jpeg75.stack/51200.20480.jpeg",
        "jpeg75.stack/51200.2048.jpeg",
        "jpeg75.stack/51200.22528.jpeg",
        "jpeg75.stack/51200.24576.jpeg",
        "jpeg75.stack/51200.26624.jpeg",
        "jpeg75.stack/51200.28672.jpeg",
        "jpeg75.stack/51200.30720.jpeg",
        "jpeg75.stack/51200.32768.jpeg",
        "jpeg75.stack/51200.34816.jpeg",
        "jpeg75.stack/51200.36864.jpeg",
        "jpeg75.stack/51200.38912.jpeg",
        "jpeg75.stack/51200.40960.jpeg",
        "jpeg75.stack/51200.4096.jpeg",
        "jpeg75.stack/51200.43008.jpeg",
        "jpeg75.stack/51200.45056.jpeg",
        "jpeg75.stack/51200.47104.jpeg",
        "jpeg75.stack/51200.49152.jpeg",
        "jpeg75.stack/51200.51200.jpeg",
        "jpeg75.stack/51200.53248.jpeg",
        "jpeg75.stack/51200.55296.jpeg",
        "jpeg75.stack/51200.57344.jpeg",
        "jpeg75.stack/51200.59392.jpeg",
        "jpeg75.stack/51200.61440.jpeg",
        "jpeg75.stack/51200.6144.jpeg",
        "jpeg75.stack/51200.63488.jpeg",
        "jpeg75.stack/51200.65536.jpeg",
        "jpeg75.stack/51200.67584.jpeg",
        "jpeg75.stack/51200.69632.jpeg",
        "jpeg75.stack/51200.71680.jpeg",
        "jpeg75.stack/51200.73728.jpeg",
        "jpeg75.stack/51200.75776.jpeg",
        "jpeg75.stack/51200.77824.jpeg",
        "jpeg75.stack/51200.79872.jpeg",
        "jpeg75.stack/51200.81920.jpeg",
        "jpeg75.stack/51200.8192.jpeg",
        "jpeg75.stack/53248.0.jpeg",
        "jpeg75.stack/53248.10240.jpeg",
        "jpeg75.stack/53248.12288.jpeg",
        "jpeg75.stack/53248.14336.jpeg",
        "jpeg75.stack/53248.16384.jpeg",
        "jpeg75.stack/53248.18432.jpeg",
        "jpeg75.stack/53248.20480.jpeg",
        "jpeg75.stack/53248.2048.jpeg",
        "jpeg75.stack/53248.22528.jpeg",
        "jpeg75.stack/53248.24576.jpeg",
        "jpeg75.stack/53248.26624.jpeg",
        "jpeg75.stack/53248.28672.jpeg",
        "jpeg75.stack/53248.30720.jpeg",
        "jpeg75.stack/53248.32768.jpeg",
        "jpeg75.stack/53248.34816.jpeg",
        "jpeg75.stack/53248.36864.jpeg",
        "jpeg75.stack/53248.38912.jpeg",
        "jpeg75.stack/53248.40960.jpeg",
        "jpeg75.stack/53248.4096.jpeg",
        "jpeg75.stack/53248.43008.jpeg",
        "jpeg75.stack/53248.45056.jpeg",
        "jpeg75.stack/53248.47104.jpeg",
        "jpeg75.stack/53248.49152.jpeg",
        "jpeg75.stack/53248.51200.jpeg",
        "jpeg75.stack/53248.53248.jpeg",
        "jpeg75.stack/53248.55296.jpeg",
        "jpeg75.stack/53248.57344.jpeg",
        "jpeg75.stack/53248.59392.jpeg",
        "jpeg75.stack/53248.61440.jpeg",
        "jpeg75.stack/53248.6144.jpeg",
        "jpeg75.stack/53248.63488.jpeg",
        "jpeg75.stack/53248.65536.jpeg",
        "jpeg75.stack/53248.67584.jpeg",
        "jpeg75.stack/53248.69632.jpeg",
        "jpeg75.stack/53248.71680.jpeg",
        "jpeg75.stack/53248.73728.jpeg",
        "jpeg75.stack/53248.75776.jpeg",
        "jpeg75.stack/53248.77824.jpeg",
        "jpeg75.stack/53248.79872.jpeg",
        "jpeg75.stack/53248.81920.jpeg",
        "jpeg75.stack/53248.8192.jpeg",
        "jpeg75.stack/55296.0.jpeg",
        "jpeg75.stack/55296.10240.jpeg",
        "jpeg75.stack/55296.12288.jpeg",
        "jpeg75.stack/55296.14336.jpeg",
        "jpeg75.stack/55296.16384.jpeg",
        "jpeg75.stack/55296.18432.jpeg",
        "jpeg75.stack/55296.20480.jpeg",
        "jpeg75.stack/55296.2048.jpeg",
        "jpeg75.stack/55296.22528.jpeg",
        "jpeg75.stack/55296.24576.jpeg",
        "jpeg75.stack/55296.26624.jpeg",
        "jpeg75.stack/55296.28672.jpeg",
        "jpeg75.stack/55296.30720.jpeg",
        "jpeg75.stack/55296.32768.jpeg",
        "jpeg75.stack/55296.34816.jpeg",
        "jpeg75.stack/55296.36864.jpeg",
        "jpeg75.stack/55296.38912.jpeg",
        "jpeg75.stack/55296.40960.jpeg",
        "jpeg75.stack/55296.4096.jpeg",
        "jpeg75.stack/55296.43008.jpeg",
        "jpeg75.stack/55296.45056.jpeg",
        "jpeg75.stack/55296.47104.jpeg",
        "jpeg75.stack/55296.49152.jpeg",
        "jpeg75.stack/55296.51200.jpeg",
        "jpeg75.stack/55296.53248.jpeg",
        "jpeg75.stack/55296.55296.jpeg",
        "jpeg75.stack/55296.57344.jpeg",
        "jpeg75.stack/55296.59392.jpeg",
        "jpeg75.stack/55296.61440.jpeg",
        "jpeg75.stack/55296.6144.jpeg",
        "jpeg75.stack/55296.63488.jpeg",
        "jpeg75.stack/55296.65536.jpeg",
        "jpeg75.stack/55296.67584.jpeg",
        "jpeg75.stack/55296.69632.jpeg",
        "jpeg75.stack/55296.71680.jpeg",
        "jpeg75.stack/55296.73728.jpeg",
        "jpeg75.stack/55296.75776.jpeg",
        "jpeg75.stack/55296.77824.jpeg",
        "jpeg75.stack/55296.79872.jpeg",
        "jpeg75.stack/55296.81920.jpeg",
        "jpeg75.stack/55296.8192.jpeg",
        "jpeg75.stack/57344.0.jpeg",
        "jpeg75.stack/57344.10240.jpeg",
        "jpeg75.stack/57344.12288.jpeg",
        "jpeg75.stack/57344.14336.jpeg",
        "jpeg75.stack/57344.16384.jpeg",
        "jpeg75.stack/57344.18432.jpeg",
        "jpeg75.stack/57344.20480.jpeg",
        "jpeg75.stack/57344.2048.jpeg",
        "jpeg75.stack/57344.22528.jpeg",
        "jpeg75.stack/57344.24576.jpeg",
        "jpeg75.stack/57344.26624.jpeg",
        "jpeg75.stack/57344.28672.jpeg",
        "jpeg75.stack/57344.30720.jpeg",
        "jpeg75.stack/57344.32768.jpeg",
        "jpeg75.stack/57344.34816.jpeg",
        "jpeg75.stack/57344.36864.jpeg",
        "jpeg75.stack/57344.38912.jpeg",
        "jpeg75.stack/57344.40960.jpeg",
        "jpeg75.stack/57344.4096.jpeg",
        "jpeg75.stack/57344.43008.jpeg",
        "jpeg75.stack/57344.45056.jpeg",
        "jpeg75.stack/57344.47104.jpeg",
        "jpeg75.stack/57344.49152.jpeg",
        "jpeg75.stack/57344.51200.jpeg",
        "jpeg75.stack/57344.53248.jpeg",
        "jpeg75.stack/57344.55296.jpeg",
        "jpeg75.stack/57344.57344.jpeg",
        "jpeg75.stack/57344.59392.jpeg",
        "jpeg75.stack/57344.61440.jpeg",
        "jpeg75.stack/57344.6144.jpeg",
        "jpeg75.stack/57344.63488.jpeg",
        "jpeg75.stack/57344.65536.jpeg",
        "jpeg75.stack/57344.67584.jpeg",
        "jpeg75.stack/57344.69632.jpeg",
        "jpeg75.stack/57344.71680.jpeg",
        "jpeg75.stack/57344.73728.jpeg",
        "jpeg75.stack/57344.75776.jpeg",
        "jpeg75.stack/57344.77824.jpeg",
        "jpeg75.stack/57344.79872.jpeg",
        "jpeg75.stack/57344.81920.jpeg",
        "jpeg75.stack/57344.8192.jpeg",
        "jpeg75.stack/59392.0.jpeg",
        "jpeg75.stack/59392.10240.jpeg",
        "jpeg75.stack/59392.12288.jpeg",
        "jpeg75.stack/59392.14336.jpeg",
        "jpeg75.stack/59392.16384.jpeg",
        "jpeg75.stack/59392.18432.jpeg",
        "jpeg75.stack/59392.20480.jpeg",
        "jpeg75.stack/59392.2048.jpeg",
        "jpeg75.stack/59392.22528.jpeg",
        "jpeg75.stack/59392.24576.jpeg",
        "jpeg75.stack/59392.26624.jpeg",
        "jpeg75.stack/59392.28672.jpeg",
        "jpeg75.stack/59392.30720.jpeg",
        "jpeg75.stack/59392.32768.jpeg",
        "jpeg75.stack/59392.34816.jpeg",
        "jpeg75.stack/59392.36864.jpeg",
        "jpeg75.stack/59392.38912.jpeg",
        "jpeg75.stack/59392.40960.jpeg",
        "jpeg75.stack/59392.4096.jpeg",
        "jpeg75.stack/59392.43008.jpeg",
        "jpeg75.stack/59392.45056.jpeg",
        "jpeg75.stack/59392.47104.jpeg",
        "jpeg75.stack/59392.49152.jpeg",
        "jpeg75.stack/59392.51200.jpeg",
        "jpeg75.stack/59392.53248.jpeg",
        "jpeg75.stack/59392.55296.jpeg",
        "jpeg75.stack/59392.57344.jpeg",
        "jpeg75.stack/59392.59392.jpeg",
        "jpeg75.stack/59392.61440.jpeg",
        "jpeg75.stack/59392.6144.jpeg",
        "jpeg75.stack/59392.63488.jpeg",
        "jpeg75.stack/59392.65536.jpeg",
        "jpeg75.stack/59392.67584.jpeg",
        "jpeg75.stack/59392.69632.jpeg",
        "jpeg75.stack/59392.71680.jpeg",
        "jpeg75.stack/59392.73728.jpeg",
        "jpeg75.stack/59392.75776.jpeg",
        "jpeg75.stack/59392.77824.jpeg",
        "jpeg75.stack/59392.79872.jpeg",
        "jpeg75.stack/59392.81920.jpeg",
        "jpeg75.stack/59392.8192.jpeg",
        "jpeg75.stack/61440.0.jpeg",
        "jpeg75.stack/61440.10240.jpeg",
        "jpeg75.stack/61440.12288.jpeg",
        "jpeg75.stack/61440.14336.jpeg",
        "jpeg75.stack/61440.16384.jpeg",
        "jpeg75.stack/61440.18432.jpeg",
        "jpeg75.stack/61440.20480.jpeg",
        "jpeg75.stack/61440.2048.jpeg",
        "jpeg75.stack/61440.22528.jpeg",
        "jpeg75.stack/61440.24576.jpeg",
        "jpeg75.stack/61440.26624.jpeg",
        "jpeg75.stack/61440.28672.jpeg",
        "jpeg75.stack/61440.30720.jpeg",
        "jpeg75.stack/61440.32768.jpeg",
        "jpeg75.stack/61440.34816.jpeg",
        "jpeg75.stack/61440.36864.jpeg",
        "jpeg75.stack/61440.38912.jpeg",
        "jpeg75.stack/61440.40960.jpeg",
        "jpeg75.stack/61440.4096.jpeg",
        "jpeg75.stack/61440.43008.jpeg",
        "jpeg75.stack/61440.45056.jpeg",
        "jpeg75.stack/61440.47104.jpeg",
        "jpeg75.stack/61440.49152.jpeg",
        "jpeg75.stack/61440.51200.jpeg",
        "jpeg75.stack/61440.53248.jpeg",
        "jpeg75.stack/61440.55296.jpeg",
        "jpeg75.stack/61440.57344.jpeg",
        "jpeg75.stack/61440.59392.jpeg",
        "jpeg75.stack/61440.61440.jpeg",
        "jpeg75.stack/61440.6144.jpeg",
        "jpeg75.stack/61440.63488.jpeg",
        "jpeg75.stack/61440.65536.jpeg",
        "jpeg75.stack/61440.67584.jpeg",
        "jpeg75.stack/61440.69632.jpeg",
        "jpeg75.stack/61440.71680.jpeg",
        "jpeg75.stack/61440.73728.jpeg",
        "jpeg75.stack/61440.75776.jpeg",
        "jpeg75.stack/61440.77824.jpeg",
        "jpeg75.stack/61440.79872.jpeg",
        "jpeg75.stack/61440.81920.jpeg",
        "jpeg75.stack/61440.8192.jpeg",
        "jpeg75.stack/6144.0.jpeg",
        "jpeg75.stack/6144.10240.jpeg",
        "jpeg75.stack/6144.12288.jpeg",
        "jpeg75.stack/6144.14336.jpeg",
        "jpeg75.stack/6144.16384.jpeg",
        "jpeg75.stack/6144.18432.jpeg",
        "jpeg75.stack/6144.20480.jpeg",
        "jpeg75.stack/6144.2048.jpeg",
        "jpeg75.stack/6144.22528.jpeg",
        "jpeg75.stack/6144.24576.jpeg",
        "jpeg75.stack/6144.26624.jpeg",
        "jpeg75.stack/6144.28672.jpeg",
        "jpeg75.stack/6144.30720.jpeg",
        "jpeg75.stack/6144.32768.jpeg",
        "jpeg75.stack/6144.34816.jpeg",
        "jpeg75.stack/6144.36864.jpeg",
        "jpeg75.stack/6144.38912.jpeg",
        "jpeg75.stack/6144.40960.jpeg",
        "jpeg75.stack/6144.4096.jpeg",
        "jpeg75.stack/6144.43008.jpeg",
        "jpeg75.stack/6144.45056.jpeg",
        "jpeg75.stack/6144.47104.jpeg",
        "jpeg75.stack/6144.49152.jpeg",
        "jpeg75.stack/6144.51200.jpeg",
        "jpeg75.stack/6144.53248.jpeg",
        "jpeg75.stack/6144.55296.jpeg",
        "jpeg75.stack/6144.57344.jpeg",
        "jpeg75.stack/6144.59392.jpeg",
        "jpeg75.stack/6144.61440.jpeg",
        "jpeg75.stack/6144.6144.jpeg",
        "jpeg75.stack/6144.63488.jpeg",
        "jpeg75.stack/6144.65536.jpeg",
        "jpeg75.stack/6144.67584.jpeg",
        "jpeg75.stack/6144.69632.jpeg",
        "jpeg75.stack/6144.71680.jpeg",
        "jpeg75.stack/6144.73728.jpeg",
        "jpeg75.stack/6144.75776.jpeg",
        "jpeg75.stack/6144.77824.jpeg",
        "jpeg75.stack/6144.79872.jpeg",
        "jpeg75.stack/6144.81920.jpeg",
        "jpeg75.stack/6144.8192.jpeg",
        "jpeg75.stack/63488.0.jpeg",
        "jpeg75.stack/63488.10240.jpeg",
        "jpeg75.stack/63488.12288.jpeg",
        "jpeg75.stack/63488.14336.jpeg",
        "jpeg75.stack/63488.16384.jpeg",
        "jpeg75.stack/63488.18432.jpeg",
        "jpeg75.stack/63488.20480.jpeg",
        "jpeg75.stack/63488.2048.jpeg",
        "jpeg75.stack/63488.22528.jpeg",
        "jpeg75.stack/63488.24576.jpeg",
        "jpeg75.stack/63488.26624.jpeg",
        "jpeg75.stack/63488.28672.jpeg",
        "jpeg75.stack/63488.30720.jpeg",
        "jpeg75.stack/63488.32768.jpeg",
        "jpeg75.stack/63488.34816.jpeg",
        "jpeg75.stack/63488.36864.jpeg",
        "jpeg75.stack/63488.38912.jpeg",
        "jpeg75.stack/63488.40960.jpeg",
        "jpeg75.stack/63488.4096.jpeg",
        "jpeg75.stack/63488.43008.jpeg",
        "jpeg75.stack/63488.45056.jpeg",
        "jpeg75.stack/63488.47104.jpeg",
        "jpeg75.stack/63488.49152.jpeg",
        "jpeg75.stack/63488.51200.jpeg",
        "jpeg75.stack/63488.53248.jpeg",
        "jpeg75.stack/63488.55296.jpeg",
        "jpeg75.stack/63488.57344.jpeg",
        "jpeg75.stack/63488.59392.jpeg",
        "jpeg75.stack/63488.61440.jpeg",
        "jpeg75.stack/63488.6144.jpeg",
        "jpeg75.stack/63488.63488.jpeg",
        "jpeg75.stack/63488.65536.jpeg",
        "jpeg75.stack/63488.67584.jpeg",
        "jpeg75.stack/63488.69632.jpeg",
        "jpeg75.stack/63488.71680.jpeg",
        "jpeg75.stack/63488.73728.jpeg",
        "jpeg75.stack/63488.75776.jpeg",
        "jpeg75.stack/63488.77824.jpeg",
        "jpeg75.stack/63488.79872.jpeg",
        "jpeg75.stack/63488.81920.jpeg",
        "jpeg75.stack/63488.8192.jpeg",
        "jpeg75.stack/65536.0.jpeg",
        "jpeg75.stack/65536.10240.jpeg",
        "jpeg75.stack/65536.12288.jpeg",
        "jpeg75.stack/65536.14336.jpeg",
        "jpeg75.stack/65536.16384.jpeg",
        "jpeg75.stack/65536.18432.jpeg",
        "jpeg75.stack/65536.20480.jpeg",
        "jpeg75.stack/65536.2048.jpeg",
        "jpeg75.stack/65536.22528.jpeg",
        "jpeg75.stack/65536.24576.jpeg",
        "jpeg75.stack/65536.26624.jpeg",
        "jpeg75.stack/65536.28672.jpeg",
        "jpeg75.stack/65536.30720.jpeg",
        "jpeg75.stack/65536.32768.jpeg",
        "jpeg75.stack/65536.34816.jpeg",
        "jpeg75.stack/65536.36864.jpeg",
        "jpeg75.stack/65536.38912.jpeg",
        "jpeg75.stack/65536.40960.jpeg",
        "jpeg75.stack/65536.4096.jpeg",
        "jpeg75.stack/65536.43008.jpeg",
        "jpeg75.stack/65536.45056.jpeg",
        "jpeg75.stack/65536.47104.jpeg",
        "jpeg75.stack/65536.49152.jpeg",
        "jpeg75.stack/65536.51200.jpeg",
        "jpeg75.stack/65536.53248.jpeg",
        "jpeg75.stack/65536.55296.jpeg",
        "jpeg75.stack/65536.57344.jpeg",
        "jpeg75.stack/65536.59392.jpeg",
        "jpeg75.stack/65536.61440.jpeg",
        "jpeg75.stack/65536.6144.jpeg",
        "jpeg75.stack/65536.63488.jpeg",
        "jpeg75.stack/65536.65536.jpeg",
        "jpeg75.stack/65536.67584.jpeg",
        "jpeg75.stack/65536.69632.jpeg",
        "jpeg75.stack/65536.71680.jpeg",
        "jpeg75.stack/65536.73728.jpeg",
        "jpeg75.stack/65536.75776.jpeg",
        "jpeg75.stack/65536.77824.jpeg",
        "jpeg75.stack/65536.79872.jpeg",
        "jpeg75.stack/65536.81920.jpeg",
        "jpeg75.stack/65536.8192.jpeg",
        "jpeg75.stack/67584.0.jpeg",
        "jpeg75.stack/67584.10240.jpeg",
        "jpeg75.stack/67584.12288.jpeg",
        "jpeg75.stack/67584.14336.jpeg",
        "jpeg75.stack/67584.16384.jpeg",
        "jpeg75.stack/67584.18432.jpeg",
        "jpeg75.stack/67584.20480.jpeg",
        "jpeg75.stack/67584.2048.jpeg",
        "jpeg75.stack/67584.22528.jpeg",
        "jpeg75.stack/67584.24576.jpeg",
        "jpeg75.stack/67584.26624.jpeg",
        "jpeg75.stack/67584.28672.jpeg",
        "jpeg75.stack/67584.30720.jpeg",
        "jpeg75.stack/67584.32768.jpeg",
        "jpeg75.stack/67584.34816.jpeg",
        "jpeg75.stack/67584.36864.jpeg",
        "jpeg75.stack/67584.38912.jpeg",
        "jpeg75.stack/67584.40960.jpeg",
        "jpeg75.stack/67584.4096.jpeg",
        "jpeg75.stack/67584.43008.jpeg",
        "jpeg75.stack/67584.45056.jpeg",
        "jpeg75.stack/67584.47104.jpeg",
        "jpeg75.stack/67584.49152.jpeg",
        "jpeg75.stack/67584.51200.jpeg",
        "jpeg75.stack/67584.53248.jpeg",
        "jpeg75.stack/67584.55296.jpeg",
        "jpeg75.stack/67584.57344.jpeg",
        "jpeg75.stack/67584.59392.jpeg",
        "jpeg75.stack/67584.61440.jpeg",
        "jpeg75.stack/67584.6144.jpeg",
        "jpeg75.stack/67584.63488.jpeg",
        "jpeg75.stack/67584.65536.jpeg",
        "jpeg75.stack/67584.67584.jpeg",
        "jpeg75.stack/67584.69632.jpeg",
        "jpeg75.stack/67584.71680.jpeg",
        "jpeg75.stack/67584.73728.jpeg",
        "jpeg75.stack/67584.75776.jpeg",
        "jpeg75.stack/67584.77824.jpeg",
        "jpeg75.stack/67584.79872.jpeg",
        "jpeg75.stack/67584.81920.jpeg",
        "jpeg75.stack/67584.8192.jpeg",
        "jpeg75.stack/69632.0.jpeg",
        "jpeg75.stack/69632.10240.jpeg",
        "jpeg75.stack/69632.12288.jpeg",
        "jpeg75.stack/69632.14336.jpeg",
        "jpeg75.stack/69632.16384.jpeg",
        "jpeg75.stack/69632.18432.jpeg",
        "jpeg75.stack/69632.20480.jpeg",
        "jpeg75.stack/69632.2048.jpeg",
        "jpeg75.stack/69632.22528.jpeg",
        "jpeg75.stack/69632.24576.jpeg",
        "jpeg75.stack/69632.26624.jpeg",
        "jpeg75.stack/69632.28672.jpeg",
        "jpeg75.stack/69632.30720.jpeg",
        "jpeg75.stack/69632.32768.jpeg",
        "jpeg75.stack/69632.34816.jpeg",
        "jpeg75.stack/69632.36864.jpeg",
        "jpeg75.stack/69632.38912.jpeg",
        "jpeg75.stack/69632.40960.jpeg",
        "jpeg75.stack/69632.4096.jpeg",
        "jpeg75.stack/69632.43008.jpeg",
        "jpeg75.stack/69632.45056.jpeg",
        "jpeg75.stack/69632.47104.jpeg",
        "jpeg75.stack/69632.49152.jpeg",
        "jpeg75.stack/69632.51200.jpeg",
        "jpeg75.stack/69632.53248.jpeg",
        "jpeg75.stack/69632.55296.jpeg",
        "jpeg75.stack/69632.57344.jpeg",
        "jpeg75.stack/69632.59392.jpeg",
        "jpeg75.stack/69632.61440.jpeg",
        "jpeg75.stack/69632.6144.jpeg",
        "jpeg75.stack/69632.63488.jpeg",
        "jpeg75.stack/69632.65536.jpeg",
        "jpeg75.stack/69632.67584.jpeg",
        "jpeg75.stack/69632.69632.jpeg",
        "jpeg75.stack/69632.71680.jpeg",
        "jpeg75.stack/69632.73728.jpeg",
        "jpeg75.stack/69632.75776.jpeg",
        "jpeg75.stack/69632.77824.jpeg",
        "jpeg75.stack/69632.79872.jpeg",
        "jpeg75.stack/69632.81920.jpeg",
        "jpeg75.stack/69632.8192.jpeg",
        "jpeg75.stack/71680.0.jpeg",
        "jpeg75.stack/71680.10240.jpeg",
        "jpeg75.stack/71680.12288.jpeg",
        "jpeg75.stack/71680.14336.jpeg",
        "jpeg75.stack/71680.16384.jpeg",
        "jpeg75.stack/71680.18432.jpeg",
        "jpeg75.stack/71680.20480.jpeg",
        "jpeg75.stack/71680.2048.jpeg",
        "jpeg75.stack/71680.22528.jpeg",
        "jpeg75.stack/71680.24576.jpeg",
        "jpeg75.stack/71680.26624.jpeg",
        "jpeg75.stack/71680.28672.jpeg",
        "jpeg75.stack/71680.30720.jpeg",
        "jpeg75.stack/71680.32768.jpeg",
        "jpeg75.stack/71680.34816.jpeg",
        "jpeg75.stack/71680.36864.jpeg",
        "jpeg75.stack/71680.38912.jpeg",
        "jpeg75.stack/71680.40960.jpeg",
        "jpeg75.stack/71680.4096.jpeg",
        "jpeg75.stack/71680.43008.jpeg",
        "jpeg75.stack/71680.45056.jpeg",
        "jpeg75.stack/71680.47104.jpeg",
        "jpeg75.stack/71680.49152.jpeg",
        "jpeg75.stack/71680.51200.jpeg",
        "jpeg75.stack/71680.53248.jpeg",
        "jpeg75.stack/71680.55296.jpeg",
        "jpeg75.stack/71680.57344.jpeg",
        "jpeg75.stack/71680.59392.jpeg",
        "jpeg75.stack/71680.61440.jpeg",
        "jpeg75.stack/71680.6144.jpeg",
        "jpeg75.stack/71680.63488.jpeg",
        "jpeg75.stack/71680.65536.jpeg",
        "jpeg75.stack/71680.67584.jpeg",
        "jpeg75.stack/71680.69632.jpeg",
        "jpeg75.stack/71680.71680.jpeg",
        "jpeg75.stack/71680.73728.jpeg",
        "jpeg75.stack/71680.75776.jpeg",
        "jpeg75.stack/71680.77824.jpeg",
        "jpeg75.stack/71680.79872.jpeg",
        "jpeg75.stack/71680.81920.jpeg",
        "jpeg75.stack/71680.8192.jpeg",
        "jpeg75.stack/73728.0.jpeg",
        "jpeg75.stack/73728.10240.jpeg",
        "jpeg75.stack/73728.12288.jpeg",
        "jpeg75.stack/73728.14336.jpeg",
        "jpeg75.stack/73728.16384.jpeg",
        "jpeg75.stack/73728.18432.jpeg",
        "jpeg75.stack/73728.20480.jpeg",
        "jpeg75.stack/73728.2048.jpeg",
        "jpeg75.stack/73728.22528.jpeg",
        "jpeg75.stack/73728.24576.jpeg",
        "jpeg75.stack/73728.26624.jpeg",
        "jpeg75.stack/73728.28672.jpeg",
        "jpeg75.stack/73728.30720.jpeg",
        "jpeg75.stack/73728.32768.jpeg",
        "jpeg75.stack/73728.34816.jpeg",
        "jpeg75.stack/73728.36864.jpeg",
        "jpeg75.stack/73728.38912.jpeg",
        "jpeg75.stack/73728.40960.jpeg",
        "jpeg75.stack/73728.4096.jpeg",
        "jpeg75.stack/73728.43008.jpeg",
        "jpeg75.stack/73728.45056.jpeg",
        "jpeg75.stack/73728.47104.jpeg",
        "jpeg75.stack/73728.49152.jpeg",
        "jpeg75.stack/73728.51200.jpeg",
        "jpeg75.stack/73728.53248.jpeg",
        "jpeg75.stack/73728.55296.jpeg",
        "jpeg75.stack/73728.57344.jpeg",
        "jpeg75.stack/73728.59392.jpeg",
        "jpeg75.stack/73728.61440.jpeg",
        "jpeg75.stack/73728.6144.jpeg",
        "jpeg75.stack/73728.63488.jpeg",
        "jpeg75.stack/73728.65536.jpeg",
        "jpeg75.stack/73728.67584.jpeg",
        "jpeg75.stack/73728.69632.jpeg",
        "jpeg75.stack/73728.71680.jpeg",
        "jpeg75.stack/73728.73728.jpeg",
        "jpeg75.stack/73728.75776.jpeg",
        "jpeg75.stack/73728.77824.jpeg",
        "jpeg75.stack/73728.79872.jpeg",
        "jpeg75.stack/73728.81920.jpeg",
        "jpeg75.stack/73728.8192.jpeg",
        "jpeg75.stack/75776.0.jpeg",
        "jpeg75.stack/75776.10240.jpeg",
        "jpeg75.stack/75776.12288.jpeg",
        "jpeg75.stack/75776.14336.jpeg",
        "jpeg75.stack/75776.16384.jpeg",
        "jpeg75.stack/75776.18432.jpeg",
        "jpeg75.stack/75776.20480.jpeg",
        "jpeg75.stack/75776.2048.jpeg",
        "jpeg75.stack/75776.22528.jpeg",
        "jpeg75.stack/75776.24576.jpeg",
        "jpeg75.stack/75776.26624.jpeg",
        "jpeg75.stack/75776.28672.jpeg",
        "jpeg75.stack/75776.30720.jpeg",
        "jpeg75.stack/75776.32768.jpeg",
        "jpeg75.stack/75776.34816.jpeg",
        "jpeg75.stack/75776.36864.jpeg",
        "jpeg75.stack/75776.38912.jpeg",
        "jpeg75.stack/75776.40960.jpeg",
        "jpeg75.stack/75776.4096.jpeg",
        "jpeg75.stack/75776.43008.jpeg",
        "jpeg75.stack/75776.45056.jpeg",
        "jpeg75.stack/75776.47104.jpeg",
        "jpeg75.stack/75776.49152.jpeg",
        "jpeg75.stack/75776.51200.jpeg",
        "jpeg75.stack/75776.53248.jpeg",
        "jpeg75.stack/75776.55296.jpeg",
        "jpeg75.stack/75776.57344.jpeg",
        "jpeg75.stack/75776.59392.jpeg",
        "jpeg75.stack/75776.61440.jpeg",
        "jpeg75.stack/75776.6144.jpeg",
        "jpeg75.stack/75776.63488.jpeg",
        "jpeg75.stack/75776.65536.jpeg",
        "jpeg75.stack/75776.67584.jpeg",
        "jpeg75.stack/75776.69632.jpeg",
        "jpeg75.stack/75776.71680.jpeg",
        "jpeg75.stack/75776.73728.jpeg",
        "jpeg75.stack/75776.75776.jpeg",
        "jpeg75.stack/75776.77824.jpeg",
        "jpeg75.stack/75776.79872.jpeg",
        "jpeg75.stack/75776.81920.jpeg",
        "jpeg75.stack/75776.8192.jpeg",
        "jpeg75.stack/77824.0.jpeg",
        "jpeg75.stack/77824.10240.jpeg",
        "jpeg75.stack/77824.12288.jpeg",
        "jpeg75.stack/77824.14336.jpeg",
        "jpeg75.stack/77824.16384.jpeg",
        "jpeg75.stack/77824.18432.jpeg",
        "jpeg75.stack/77824.20480.jpeg",
        "jpeg75.stack/77824.2048.jpeg",
        "jpeg75.stack/77824.22528.jpeg",
        "jpeg75.stack/77824.24576.jpeg",
        "jpeg75.stack/77824.26624.jpeg",
        "jpeg75.stack/77824.28672.jpeg",
        "jpeg75.stack/77824.30720.jpeg",
        "jpeg75.stack/77824.32768.jpeg",
        "jpeg75.stack/77824.34816.jpeg",
        "jpeg75.stack/77824.36864.jpeg",
        "jpeg75.stack/77824.38912.jpeg",
        "jpeg75.stack/77824.40960.jpeg",
        "jpeg75.stack/77824.4096.jpeg",
        "jpeg75.stack/77824.43008.jpeg",
        "jpeg75.stack/77824.45056.jpeg",
        "jpeg75.stack/77824.47104.jpeg",
        "jpeg75.stack/77824.49152.jpeg",
        "jpeg75.stack/77824.51200.jpeg",
        "jpeg75.stack/77824.53248.jpeg",
        "jpeg75.stack/77824.55296.jpeg",
        "jpeg75.stack/77824.57344.jpeg",
        "jpeg75.stack/77824.59392.jpeg",
        "jpeg75.stack/77824.61440.jpeg",
        "jpeg75.stack/77824.6144.jpeg",
        "jpeg75.stack/77824.63488.jpeg",
        "jpeg75.stack/77824.65536.jpeg",
        "jpeg75.stack/77824.67584.jpeg",
        "jpeg75.stack/77824.69632.jpeg",
        "jpeg75.stack/77824.71680.jpeg",
        "jpeg75.stack/77824.73728.jpeg",
        "jpeg75.stack/77824.75776.jpeg",
        "jpeg75.stack/77824.77824.jpeg",
        "jpeg75.stack/77824.79872.jpeg",
        "jpeg75.stack/77824.81920.jpeg",
        "jpeg75.stack/77824.8192.jpeg",
        "jpeg75.stack/79872.0.jpeg",
        "jpeg75.stack/79872.10240.jpeg",
        "jpeg75.stack/79872.12288.jpeg",
        "jpeg75.stack/79872.14336.jpeg",
        "jpeg75.stack/79872.16384.jpeg",
        "jpeg75.stack/79872.18432.jpeg",
        "jpeg75.stack/79872.20480.jpeg",
        "jpeg75.stack/79872.2048.jpeg",
        "jpeg75.stack/79872.22528.jpeg",
        "jpeg75.stack/79872.24576.jpeg",
        "jpeg75.stack/79872.26624.jpeg",
        "jpeg75.stack/79872.28672.jpeg",
        "jpeg75.stack/79872.30720.jpeg",
        "jpeg75.stack/79872.32768.jpeg",
        "jpeg75.stack/79872.34816.jpeg",
        "jpeg75.stack/79872.36864.jpeg",
        "jpeg75.stack/79872.38912.jpeg",
        "jpeg75.stack/79872.40960.jpeg",
        "jpeg75.stack/79872.4096.jpeg",
        "jpeg75.stack/79872.43008.jpeg",
        "jpeg75.stack/79872.45056.jpeg",
        "jpeg75.stack/79872.47104.jpeg",
        "jpeg75.stack/79872.49152.jpeg",
        "jpeg75.stack/79872.51200.jpeg",
        "jpeg75.stack/79872.53248.jpeg",
        "jpeg75.stack/79872.55296.jpeg",
        "jpeg75.stack/79872.57344.jpeg",
        "jpeg75.stack/79872.59392.jpeg",
        "jpeg75.stack/79872.61440.jpeg",
        "jpeg75.stack/79872.6144.jpeg",
        "jpeg75.stack/79872.63488.jpeg",
        "jpeg75.stack/79872.65536.jpeg",
        "jpeg75.stack/79872.67584.jpeg",
        "jpeg75.stack/79872.69632.jpeg",
        "jpeg75.stack/79872.71680.jpeg",
        "jpeg75.stack/79872.73728.jpeg",
        "jpeg75.stack/79872.75776.jpeg",
        "jpeg75.stack/79872.77824.jpeg",
        "jpeg75.stack/79872.79872.jpeg",
        "jpeg75.stack/79872.81920.jpeg",
        "jpeg75.stack/79872.8192.jpeg",
        "jpeg75.stack/81920.0.jpeg",
        "jpeg75.stack/81920.10240.jpeg",
        "jpeg75.stack/81920.12288.jpeg",
        "jpeg75.stack/81920.14336.jpeg",
        "jpeg75.stack/81920.16384.jpeg",
        "jpeg75.stack/81920.18432.jpeg",
        "jpeg75.stack/81920.20480.jpeg",
        "jpeg75.stack/81920.2048.jpeg",
        "jpeg75.stack/81920.22528.jpeg",
        "jpeg75.stack/81920.24576.jpeg",
        "jpeg75.stack/81920.26624.jpeg",
        "jpeg75.stack/81920.28672.jpeg",
        "jpeg75.stack/81920.30720.jpeg",
        "jpeg75.stack/81920.32768.jpeg",
        "jpeg75.stack/81920.34816.jpeg",
        "jpeg75.stack/81920.36864.jpeg",
        "jpeg75.stack/81920.38912.jpeg",
        "jpeg75.stack/81920.40960.jpeg",
        "jpeg75.stack/81920.4096.jpeg",
        "jpeg75.stack/81920.43008.jpeg",
        "jpeg75.stack/81920.45056.jpeg",
        "jpeg75.stack/81920.47104.jpeg",
        "jpeg75.stack/81920.49152.jpeg",
        "jpeg75.stack/81920.51200.jpeg",
        "jpeg75.stack/81920.53248.jpeg",
        "jpeg75.stack/81920.55296.jpeg",
        "jpeg75.stack/81920.57344.jpeg",
        "jpeg75.stack/81920.59392.jpeg",
        "jpeg75.stack/81920.61440.jpeg",
        "jpeg75.stack/81920.6144.jpeg",
        "jpeg75.stack/81920.63488.jpeg",
        "jpeg75.stack/81920.65536.jpeg",
        "jpeg75.stack/81920.67584.jpeg",
        "jpeg75.stack/81920.69632.jpeg",
        "jpeg75.stack/81920.71680.jpeg",
        "jpeg75.stack/81920.73728.jpeg",
        "jpeg75.stack/81920.75776.jpeg",
        "jpeg75.stack/81920.77824.jpeg",
        "jpeg75.stack/81920.79872.jpeg",
        "jpeg75.stack/81920.81920.jpeg",
        "jpeg75.stack/81920.8192.jpeg",
        "jpeg75.stack/8192.0.jpeg",
        "jpeg75.stack/8192.10240.jpeg",
        "jpeg75.stack/8192.12288.jpeg",
        "jpeg75.stack/8192.14336.jpeg",
        "jpeg75.stack/8192.16384.jpeg",
        "jpeg75.stack/8192.18432.jpeg",
        "jpeg75.stack/8192.20480.jpeg",
        "jpeg75.stack/8192.2048.jpeg",
        "jpeg75.stack/8192.22528.jpeg",
        "jpeg75.stack/8192.24576.jpeg",
        "jpeg75.stack/8192.26624.jpeg",
        "jpeg75.stack/8192.28672.jpeg",
        "jpeg75.stack/8192.30720.jpeg",
        "jpeg75.stack/8192.32768.jpeg",
        "jpeg75.stack/8192.34816.jpeg",
        "jpeg75.stack/8192.36864.jpeg",
        "jpeg75.stack/8192.38912.jpeg",
        "jpeg75.stack/8192.40960.jpeg",
        "jpeg75.stack/8192.4096.jpeg",
        "jpeg75.stack/8192.43008.jpeg",
        "jpeg75.stack/8192.45056.jpeg",
        "jpeg75.stack/8192.47104.jpeg",
        "jpeg75.stack/8192.49152.jpeg",
        "jpeg75.stack/8192.51200.jpeg",
        "jpeg75.stack/8192.53248.jpeg",
        "jpeg75.stack/8192.55296.jpeg",
        "jpeg75.stack/8192.57344.jpeg",
        "jpeg75.stack/8192.59392.jpeg",
        "jpeg75.stack/8192.61440.jpeg",
        "jpeg75.stack/8192.6144.jpeg",
        "jpeg75.stack/8192.63488.jpeg",
        "jpeg75.stack/8192.65536.jpeg",
        "jpeg75.stack/8192.67584.jpeg",
        "jpeg75.stack/8192.69632.jpeg",
        "jpeg75.stack/8192.71680.jpeg",
        "jpeg75.stack/8192.73728.jpeg",
        "jpeg75.stack/8192.75776.jpeg",
        "jpeg75.stack/8192.77824.jpeg",
        "jpeg75.stack/8192.79872.jpeg",
        "jpeg75.stack/8192.81920.jpeg",
        "jpeg75.stack/8192.8192.jpeg",
        "jpeg75.stack/83968.0.jpeg",
        "jpeg75.stack/83968.10240.jpeg",
        "jpeg75.stack/83968.12288.jpeg",
        "jpeg75.stack/83968.14336.jpeg",
        "jpeg75.stack/83968.16384.jpeg",
        "jpeg75.stack/83968.18432.jpeg",
        "jpeg75.stack/83968.20480.jpeg",
        "jpeg75.stack/83968.2048.jpeg",
        "jpeg75.stack/83968.22528.jpeg",
        "jpeg75.stack/83968.24576.jpeg",
        "jpeg75.stack/83968.26624.jpeg",
        "jpeg75.stack/83968.28672.jpeg",
        "jpeg75.stack/83968.30720.jpeg",
        "jpeg75.stack/83968.32768.jpeg",
        "jpeg75.stack/83968.34816.jpeg",
        "jpeg75.stack/83968.36864.jpeg",
        "jpeg75.stack/83968.38912.jpeg",
        "jpeg75.stack/83968.40960.jpeg",
        "jpeg75.stack/83968.4096.jpeg",
        "jpeg75.stack/83968.43008.jpeg",
        "jpeg75.stack/83968.45056.jpeg",
        "jpeg75.stack/83968.47104.jpeg",
        "jpeg75.stack/83968.49152.jpeg",
        "jpeg75.stack/83968.51200.jpeg",
        "jpeg75.stack/83968.53248.jpeg",
        "jpeg75.stack/83968.55296.jpeg",
        "jpeg75.stack/83968.57344.jpeg",
        "jpeg75.stack/83968.59392.jpeg",
        "jpeg75.stack/83968.61440.jpeg",
        "jpeg75.stack/83968.6144.jpeg",
        "jpeg75.stack/83968.63488.jpeg",
        "jpeg75.stack/83968.65536.jpeg",
        "jpeg75.stack/83968.67584.jpeg",
        "jpeg75.stack/83968.69632.jpeg",
        "jpeg75.stack/83968.71680.jpeg",
        "jpeg75.stack/83968.73728.jpeg",
        "jpeg75.stack/83968.75776.jpeg",
        "jpeg75.stack/83968.77824.jpeg",
        "jpeg75.stack/83968.79872.jpeg",
        "jpeg75.stack/83968.81920.jpeg",
        "jpeg75.stack/83968.8192.jpeg",
        "jpeg75.stack/86016.0.jpeg",
        "jpeg75.stack/86016.10240.jpeg",
        "jpeg75.stack/86016.12288.jpeg",
        "jpeg75.stack/86016.14336.jpeg",
        "jpeg75.stack/86016.16384.jpeg",
        "jpeg75.stack/86016.18432.jpeg",
        "jpeg75.stack/86016.20480.jpeg",
        "jpeg75.stack/86016.2048.jpeg",
        "jpeg75.stack/86016.22528.jpeg",
        "jpeg75.stack/86016.24576.jpeg",
        "jpeg75.stack/86016.26624.jpeg",
        "jpeg75.stack/86016.28672.jpeg",
        "jpeg75.stack/86016.30720.jpeg",
        "jpeg75.stack/86016.32768.jpeg",
        "jpeg75.stack/86016.34816.jpeg",
        "jpeg75.stack/86016.36864.jpeg",
        "jpeg75.stack/86016.38912.jpeg",
        "jpeg75.stack/86016.40960.jpeg",
        "jpeg75.stack/86016.4096.jpeg",
        "jpeg75.stack/86016.43008.jpeg",
        "jpeg75.stack/86016.45056.jpeg",
        "jpeg75.stack/86016.47104.jpeg",
        "jpeg75.stack/86016.49152.jpeg",
        "jpeg75.stack/86016.51200.jpeg",
        "jpeg75.stack/86016.53248.jpeg",
        "jpeg75.stack/86016.55296.jpeg",
        "jpeg75.stack/86016.57344.jpeg",
        "jpeg75.stack/86016.59392.jpeg",
        "jpeg75.stack/86016.61440.jpeg",
        "jpeg75.stack/86016.6144.jpeg",
        "jpeg75.stack/86016.63488.jpeg",
        "jpeg75.stack/86016.65536.jpeg",
        "jpeg75.stack/86016.67584.jpeg",
        "jpeg75.stack/86016.69632.jpeg",
        "jpeg75.stack/86016.71680.jpeg",
        "jpeg75.stack/86016.73728.jpeg",
        "jpeg75.stack/86016.75776.jpeg",
        "jpeg75.stack/86016.77824.jpeg",
        "jpeg75.stack/86016.79872.jpeg",
        "jpeg75.stack/86016.81920.jpeg",
        "jpeg75.stack/86016.8192.jpeg",
        "jpeg75.stack/88064.0.jpeg",
        "jpeg75.stack/88064.10240.jpeg",
        "jpeg75.stack/88064.12288.jpeg",
        "jpeg75.stack/88064.14336.jpeg",
        "jpeg75.stack/88064.16384.jpeg",
        "jpeg75.stack/88064.18432.jpeg",
        "jpeg75.stack/88064.20480.jpeg",
        "jpeg75.stack/88064.2048.jpeg",
        "jpeg75.stack/88064.22528.jpeg",
        "jpeg75.stack/88064.24576.jpeg",
        "jpeg75.stack/88064.26624.jpeg",
        "jpeg75.stack/88064.28672.jpeg",
        "jpeg75.stack/88064.30720.jpeg",
        "jpeg75.stack/88064.32768.jpeg",
        "jpeg75.stack/88064.34816.jpeg",
        "jpeg75.stack/88064.36864.jpeg",
        "jpeg75.stack/88064.38912.jpeg",
        "jpeg75.stack/88064.40960.jpeg",
        "jpeg75.stack/88064.4096.jpeg",
        "jpeg75.stack/88064.43008.jpeg",
        "jpeg75.stack/88064.45056.jpeg",
        "jpeg75.stack/88064.47104.jpeg",
        "jpeg75.stack/88064.49152.jpeg",
        "jpeg75.stack/88064.51200.jpeg",
        "jpeg75.stack/88064.53248.jpeg",
        "jpeg75.stack/88064.55296.jpeg",
        "jpeg75.stack/88064.57344.jpeg",
        "jpeg75.stack/88064.59392.jpeg",
        "jpeg75.stack/88064.61440.jpeg",
        "jpeg75.stack/88064.6144.jpeg",
        "jpeg75.stack/88064.63488.jpeg",
        "jpeg75.stack/88064.65536.jpeg",
        "jpeg75.stack/88064.67584.jpeg",
        "jpeg75.stack/88064.69632.jpeg",
        "jpeg75.stack/88064.71680.jpeg",
        "jpeg75.stack/88064.73728.jpeg",
        "jpeg75.stack/88064.75776.jpeg",
        "jpeg75.stack/88064.77824.jpeg",
        "jpeg75.stack/88064.79872.jpeg",
        "jpeg75.stack/88064.81920.jpeg",
        "jpeg75.stack/88064.8192.jpeg",
        "jpeg75.stack/90112.0.jpeg",
        "jpeg75.stack/90112.10240.jpeg",
        "jpeg75.stack/90112.12288.jpeg",
        "jpeg75.stack/90112.14336.jpeg",
        "jpeg75.stack/90112.16384.jpeg",
        "jpeg75.stack/90112.18432.jpeg",
        "jpeg75.stack/90112.20480.jpeg",
        "jpeg75.stack/90112.2048.jpeg",
        "jpeg75.stack/90112.22528.jpeg",
        "jpeg75.stack/90112.24576.jpeg",
        "jpeg75.stack/90112.26624.jpeg",
        "jpeg75.stack/90112.28672.jpeg",
        "jpeg75.stack/90112.30720.jpeg",
        "jpeg75.stack/90112.32768.jpeg",
        "jpeg75.stack/90112.34816.jpeg",
        "jpeg75.stack/90112.36864.jpeg",
        "jpeg75.stack/90112.38912.jpeg",
        "jpeg75.stack/90112.40960.jpeg",
        "jpeg75.stack/90112.4096.jpeg",
        "jpeg75.stack/90112.43008.jpeg",
        "jpeg75.stack/90112.45056.jpeg",
        "jpeg75.stack/90112.47104.jpeg",
        "jpeg75.stack/90112.49152.jpeg",
        "jpeg75.stack/90112.51200.jpeg",
        "jpeg75.stack/90112.53248.jpeg",
        "jpeg75.stack/90112.55296.jpeg",
        "jpeg75.stack/90112.57344.jpeg",
        "jpeg75.stack/90112.59392.jpeg",
        "jpeg75.stack/90112.61440.jpeg",
        "jpeg75.stack/90112.6144.jpeg",
        "jpeg75.stack/90112.63488.jpeg",
        "jpeg75.stack/90112.65536.jpeg",
        "jpeg75.stack/90112.67584.jpeg",
        "jpeg75.stack/90112.69632.jpeg",
        "jpeg75.stack/90112.71680.jpeg",
        "jpeg75.stack/90112.73728.jpeg",
        "jpeg75.stack/90112.75776.jpeg",
        "jpeg75.stack/90112.77824.jpeg",
        "jpeg75.stack/90112.79872.jpeg",
        "jpeg75.stack/90112.81920.jpeg",
        "jpeg75.stack/90112.8192.jpeg",
        "jpeg75.stack/92160.0.jpeg",
        "jpeg75.stack/92160.10240.jpeg",
        "jpeg75.stack/92160.12288.jpeg",
        "jpeg75.stack/92160.14336.jpeg",
        "jpeg75.stack/92160.16384.jpeg",
        "jpeg75.stack/92160.18432.jpeg",
        "jpeg75.stack/92160.20480.jpeg",
        "jpeg75.stack/92160.2048.jpeg",
        "jpeg75.stack/92160.22528.jpeg",
        "jpeg75.stack/92160.24576.jpeg",
        "jpeg75.stack/92160.26624.jpeg",
        "jpeg75.stack/92160.28672.jpeg",
        "jpeg75.stack/92160.30720.jpeg",
        "jpeg75.stack/92160.32768.jpeg",
        "jpeg75.stack/92160.34816.jpeg",
        "jpeg75.stack/92160.36864.jpeg",
        "jpeg75.stack/92160.38912.jpeg",
        "jpeg75.stack/92160.40960.jpeg",
        "jpeg75.stack/92160.4096.jpeg",
        "jpeg75.stack/92160.43008.jpeg",
        "jpeg75.stack/92160.45056.jpeg",
        "jpeg75.stack/92160.47104.jpeg",
        "jpeg75.stack/92160.49152.jpeg",
        "jpeg75.stack/92160.51200.jpeg",
        "jpeg75.stack/92160.53248.jpeg",
        "jpeg75.stack/92160.55296.jpeg",
        "jpeg75.stack/92160.57344.jpeg",
        "jpeg75.stack/92160.59392.jpeg",
        "jpeg75.stack/92160.61440.jpeg",
        "jpeg75.stack/92160.6144.jpeg",
        "jpeg75.stack/92160.63488.jpeg",
        "jpeg75.stack/92160.65536.jpeg",
        "jpeg75.stack/92160.67584.jpeg",
        "jpeg75.stack/92160.69632.jpeg",
        "jpeg75.stack/92160.71680.jpeg",
        "jpeg75.stack/92160.73728.jpeg",
        "jpeg75.stack/92160.75776.jpeg",
        "jpeg75.stack/92160.77824.jpeg",
        "jpeg75.stack/92160.79872.jpeg",
        "jpeg75.stack/92160.81920.jpeg",
        "jpeg75.stack/92160.8192.jpeg",
        "jpeg75.stack/94208.0.jpeg",
        "jpeg75.stack/94208.10240.jpeg",
        "jpeg75.stack/94208.12288.jpeg",
        "jpeg75.stack/94208.14336.jpeg",
        "jpeg75.stack/94208.16384.jpeg",
        "jpeg75.stack/94208.18432.jpeg",
        "jpeg75.stack/94208.20480.jpeg",
        "jpeg75.stack/94208.2048.jpeg",
        "jpeg75.stack/94208.22528.jpeg",
        "jpeg75.stack/94208.24576.jpeg",
        "jpeg75.stack/94208.26624.jpeg",
        "jpeg75.stack/94208.28672.jpeg",
        "jpeg75.stack/94208.30720.jpeg",
        "jpeg75.stack/94208.32768.jpeg",
        "jpeg75.stack/94208.34816.jpeg",
        "jpeg75.stack/94208.36864.jpeg",
        "jpeg75.stack/94208.38912.jpeg",
        "jpeg75.stack/94208.40960.jpeg",
        "jpeg75.stack/94208.4096.jpeg",
        "jpeg75.stack/94208.43008.jpeg",
        "jpeg75.stack/94208.45056.jpeg",
        "jpeg75.stack/94208.47104.jpeg",
        "jpeg75.stack/94208.49152.jpeg",
        "jpeg75.stack/94208.51200.jpeg",
        "jpeg75.stack/94208.53248.jpeg",
        "jpeg75.stack/94208.55296.jpeg",
        "jpeg75.stack/94208.57344.jpeg",
        "jpeg75.stack/94208.59392.jpeg",
        "jpeg75.stack/94208.61440.jpeg",
        "jpeg75.stack/94208.6144.jpeg",
        "jpeg75.stack/94208.63488.jpeg",
        "jpeg75.stack/94208.65536.jpeg",
        "jpeg75.stack/94208.67584.jpeg",
        "jpeg75.stack/94208.69632.jpeg",
        "jpeg75.stack/94208.71680.jpeg",
        "jpeg75.stack/94208.73728.jpeg",
        "jpeg75.stack/94208.75776.jpeg",
        "jpeg75.stack/94208.77824.jpeg",
        "jpeg75.stack/94208.79872.jpeg",
        "jpeg75.stack/94208.81920.jpeg",
        "jpeg75.stack/94208.8192.jpeg",
        "jpeg75.stack/96256.0.jpeg",
        "jpeg75.stack/96256.10240.jpeg",
        "jpeg75.stack/96256.12288.jpeg",
        "jpeg75.stack/96256.14336.jpeg",
        "jpeg75.stack/96256.16384.jpeg",
        "jpeg75.stack/96256.18432.jpeg",
        "jpeg75.stack/96256.20480.jpeg",
        "jpeg75.stack/96256.2048.jpeg",
        "jpeg75.stack/96256.22528.jpeg",
        "jpeg75.stack/96256.24576.jpeg",
        "jpeg75.stack/96256.26624.jpeg",
        "jpeg75.stack/96256.28672.jpeg",
        "jpeg75.stack/96256.30720.jpeg",
        "jpeg75.stack/96256.32768.jpeg",
        "jpeg75.stack/96256.34816.jpeg",
        "jpeg75.stack/96256.36864.jpeg",
        "jpeg75.stack/96256.38912.jpeg",
        "jpeg75.stack/96256.40960.jpeg",
        "jpeg75.stack/96256.4096.jpeg",
        "jpeg75.stack/96256.43008.jpeg",
        "jpeg75.stack/96256.45056.jpeg",
        "jpeg75.stack/96256.47104.jpeg",
        "jpeg75.stack/96256.49152.jpeg",
        "jpeg75.stack/96256.51200.jpeg",
        "jpeg75.stack/96256.53248.jpeg",
        "jpeg75.stack/96256.55296.jpeg",
        "jpeg75.stack/96256.57344.jpeg",
        "jpeg75.stack/96256.59392.jpeg",
        "jpeg75.stack/96256.61440.jpeg",
        "jpeg75.stack/96256.6144.jpeg",
        "jpeg75.stack/96256.63488.jpeg",
        "jpeg75.stack/96256.65536.jpeg",
        "jpeg75.stack/96256.67584.jpeg",
        "jpeg75.stack/96256.69632.jpeg",
        "jpeg75.stack/96256.71680.jpeg",
        "jpeg75.stack/96256.73728.jpeg",
        "jpeg75.stack/96256.75776.jpeg",
        "jpeg75.stack/96256.77824.jpeg",
        "jpeg75.stack/96256.79872.jpeg",
        "jpeg75.stack/96256.81920.jpeg",
        "jpeg75.stack/96256.8192.jpeg",
        "jpeg75.stack/98304.0.jpeg",
        "jpeg75.stack/98304.10240.jpeg",
        "jpeg75.stack/98304.12288.jpeg",
        "jpeg75.stack/98304.14336.jpeg",
        "jpeg75.stack/98304.16384.jpeg",
        "jpeg75.stack/98304.18432.jpeg",
        "jpeg75.stack/98304.20480.jpeg",
        "jpeg75.stack/98304.2048.jpeg",
        "jpeg75.stack/98304.22528.jpeg",
        "jpeg75.stack/98304.24576.jpeg",
        "jpeg75.stack/98304.26624.jpeg",
        "jpeg75.stack/98304.28672.jpeg",
        "jpeg75.stack/98304.30720.jpeg",
        "jpeg75.stack/98304.32768.jpeg",
        "jpeg75.stack/98304.34816.jpeg",
        "jpeg75.stack/98304.36864.jpeg",
        "jpeg75.stack/98304.38912.jpeg",
        "jpeg75.stack/98304.40960.jpeg",
        "jpeg75.stack/98304.4096.jpeg",
        "jpeg75.stack/98304.43008.jpeg",
        "jpeg75.stack/98304.45056.jpeg",
        "jpeg75.stack/98304.47104.jpeg",
        "jpeg75.stack/98304.49152.jpeg",
        "jpeg75.stack/98304.51200.jpeg",
        "jpeg75.stack/98304.53248.jpeg",
        "jpeg75.stack/98304.55296.jpeg",
        "jpeg75.stack/98304.57344.jpeg",
        "jpeg75.stack/98304.59392.jpeg",
        "jpeg75.stack/98304.61440.jpeg",
        "jpeg75.stack/98304.6144.jpeg",
        "jpeg75.stack/98304.63488.jpeg",
        "jpeg75.stack/98304.65536.jpeg",
        "jpeg75.stack/98304.67584.jpeg",
        "jpeg75.stack/98304.69632.jpeg",
        "jpeg75.stack/98304.71680.jpeg",
        "jpeg75.stack/98304.73728.jpeg",
        "jpeg75.stack/98304.75776.jpeg",
        "jpeg75.stack/98304.77824.jpeg",
        "jpeg75.stack/98304.79872.jpeg",
        "jpeg75.stack/98304.81920.jpeg",
        "jpeg75.stack/98304.8192.jpeg",
    ]

    # all_wsi_images = ["RGBA/default1024.zarr"]
    # all_wsi_images = ["RGBA/default2048.zarr"]
    # all_wsi_images = ["RGBA/default256.zarr"]
    # all_wsi_images = ["RGBA/default4096.zarr"]
    # all_wsi_images = ["RGBA/zstd_compressor1024.zarr"]
    # all_wsi_images = ["RGBA/zstd_compressor2048.zarr"]

    all_wsi_images = ["TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913.svs"]
    # all_wsi_images = ["default1024.zarr"]
    # all_wsi_images = ["default2048.zarr"]
    # all_wsi_images = ["default256.zarr"]
    # all_wsi_images = ["default4096.zarr"]
    # all_wsi_images = ["jpeg2k_compressor2048.zarr"]
    # all_wsi_images = ["jpeg2klevel80_compressor2048.zarr"]
    # all_wsi_images = ["kwjpeg100_compressor2048.zarr"]
    # all_wsi_images = ["kwjpeg30_compressor2048.zarr"]
    # all_wsi_images = ["kwjpeg30_compressor256.zarr"]
    # all_wsi_images = ["kwjpeg75_compressor2048.zarr"]
    # all_wsi_images = ["kwjpeg95_compressor2048.zarr"]
    # all_wsi_images = ["kwjpegVarious_compressor256.zarr"]
    # all_wsi_images = ["zstd_compressor1024.zarr"]
    # all_wsi_images = ["zstd_compressor2048.zarr"]

    print(f"Image source = {all_wsi_images}")

    # We will use a mask to determine which tiles of the slide to process.  In this example we will
    # build the mask name from the WSI file name, by inserting "-mask" and changing the file type to
    # "png", but generally any file name and any file type that we can read as an image will do.

    all_masks = [re.sub(r"^(.*)\.([^\.]*)$", r"\1-mask.png", wsi) for wsi in all_wsi_images]
    print(f"all_masks = {all_masks}")

    header = dict(
        tfdHeader(
            slides="TCGA-BH-A0BZ-01Z-00-DX1",
            filenames=all_wsi_images,
            cases="TCGA-BH-A0BZ",
            magnifications=20.0,
            read_modes="tiled",
            mask_filenames=all_masks,
        )
    )

    tiles = tf.data.Dataset.from_tensor_slices(header)

    # For the desired magnification, find the best level stored in the image file, and its associated
    # factor, width, and height.
    compute_read_parameters = ComputeReadParameters()
    tiles = tiles.map(compute_read_parameters)

    # Specify size, overlap, etc. information about the tiles that we want to analyze.
    add_tile_description = AddTileDescription()
    tiles = tiles.map(add_tile_description)

    # If there are any then read in the masks, one per slide, that specify tile selction.  If they are
    # not already then downsample (or upsample) the masks to be one pixel per tile.
    compute_resampled_mask = ComputeResampledMask()
    tiles = tiles.map(compute_resampled_mask, **dataset_map_options)

    # Split each element (e.g. each slide) into a batch of multiple rows, one per chunk to be read.
    # Note that the width `cw` or height `ch` of a row (chunk) may decreased from the requested value if
    # a chunk is near the edge of an image.  Note that it is important to call `.unbatch()` when it is
    # desired that the chunks be not batched by slide.
    compute_chunk_positions = ComputeChunkPositions()
    tiles = (
        tiles.map(compute_chunk_positions, **dataset_map_options)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .unbatch()
    )

    # For debugging purposes, do a step that does nothing
    tfd_no_op = tfdNoOp()
    tiles = tiles.map(tfd_no_op)

    # For debugging purposes, do a step that does nothing but print side effects.
    # tfd_print = tfdPrint(tf.constant(314, dtype=tf.int32))
    # tiles = tiles.map(tfd_print)

    # Read and split the chunks into the tile size we want.  Note that it is important to call
    # `.unbatch()` when it is desired that the tiles be not batched by chunk.
    read_and_split_chunk = ReadAndSplitChunk()
    tiles = (
        tiles.map(read_and_split_chunk, **dataset_map_options)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .unbatch()
    )

    # Change the element to be of the form `(tile, metadataDictionary)` rather than `alldataDictionary`.
    tiles = tiles.map(lambda elem: (elem.pop("tile"), elem), **dataset_map_options)

    # apply preprocessing to tiles - resize, float conversion, preprocessing
    tiles = tiles.map(
        lambda tile, metadata: (
            tf.cast(tf.image.resize(tile, [224, 224]), tf.float32),
            metadata,
        ),
        **dataset_map_options,
    )
    tiles = tiles.map(
        lambda tile, metadata: (
            tf.keras.applications.resnet_v2.preprocess_input(tile),
            metadata,
        ),
        **dataset_map_options,
    )

    # batch tiles
    batch_size = 128 * len(devices)

    batched_dist = strategy.experimental_distribute_dataset(tiles.batch(batch_size))
    print("read_example: %f seconds" % (time.time() - tic))
    return batched_dist


@tf.function
def predict(model, element):
    """A helper function for predict_example."""

    return model(element[0]), element[1]


def predict_example(model, batched_dist, strategy):
    """Example code for doing prediction with a tensorflow pipeline."""

    tic = time.time()

    # distributed inference, condensing distributed feature tensors, metadata dicts in lists
    feature_list = []
    metadata_list = []
    for element in batched_dist:
        f, meta = strategy.run(
            predict,
            args=(
                model,
                element,
            ),
        )
        feature_list.append(_merge_dist_tensor(strategy, f))
        metadata_list.append(_merge_dist_dict(strategy, meta))

    # merge features into single array
    features = tf.concat(feature_list, axis=0)
    del feature_list

    # merge metadata into single dict
    metadata = {}
    for key in metadata_list[0].keys():
        metadata[key] = tf.concat([meta[key] for meta in metadata_list], axis=0)
    del metadata_list

    print("predict_example: %f seconds" % (time.time() - tic))
    return features, metadata


def output_example(features, metadata):
    """Example code for writing output from a tensorflow pipeline."""

    tic = time.time()

    # write features, metadata to disk
    with h5py.File("mytestfile.hdf5", "w") as handle:
        handle.create_dataset(
            "slides",
            data=metadata["slide"].numpy(),
            dtype=h5py.string_dtype(encoding="ascii"),
        )
        handle.create_dataset("features", data=features.numpy(), dtype="float")
        handle.create_dataset("slideIdx", data=np.zeros(metadata["slide"].shape), dtype="int")
        handle.create_dataset("x_centroid", data=metadata["tx"].numpy(), dtype="float")
        handle.create_dataset("y_centroid", data=metadata["ty"].numpy(), dtype="float")
        handle.create_dataset("dataIdx", data=np.zeros(1), dtype="int")
        handle.create_dataset("wsi_mean", data=np.zeros(3), dtype="float")
        handle.create_dataset("wsi_std", data=np.zeros(3), dtype="float")

    print("output_example: %f seconds" % (time.time() - tic))


def _random_mask(shape):  # ITK order: (width, height)
    """A function that generates a random mask."""

    # Here we will make a mask with random 0 and 1 values as a demonstration.
    MaskDimension = 2
    MaskPixelType = itk.UC
    MaskImageType = itk.Image[MaskPixelType, MaskDimension]
    mask_index = itk.Index[MaskDimension]()
    mask_index.Fill(0)
    mask_size = itk.Size[MaskDimension]()
    mask_size[0] = shape[0]  # width
    mask_size[1] = shape[1]  # height
    mask_region = itk.ImageRegion[MaskDimension]()
    mask_region.SetIndex(mask_index)
    mask_region.SetSize(mask_size)
    mask_image = MaskImageType.New()
    mask_image.SetRegions(mask_region)
    mask_image.Allocate()
    random.seed("20210311", version=2)
    for y in range(mask_size[1]):
        for x in range(mask_size[0]):
            # Use numpy index order on left-hand side
            mask_image[y, x] = random.randrange(0, 2)

    return mask_image


def _convert_openslide_to_chunks(
    svs_filename,
    chunks_dirname,
    chunks_type="jpeg",
    chunk_factor=8,
    tile_size=256,
    overlap=0,
    **kwargs,
):
    """A function that writes out one image file for each chunk of a whole slide image.

    Parameters
    ----------
    svs_filename : str
        The name of the input image file name
    chunks_dirname : str
        The directory to put the output into
    chunks_type : str
        The image type of each file to be written.  E.g., "jpeg".
    chunk_factor : int
        The number of tiles (for both width and height) that make up a chunk
    tile_size : int
        The number of pixels (for both width and height) that make up a tile
    overlap : int
        The number of pixels of overlap (for both width and height) between adjacent tiles.
    **kwargs
        Optional key-value pairs to be passed to PIL.Image.save(), such as "quality=30" for jpeg writing.
    """

    assert isinstance(svs_filename, str)
    assert isinstance(chunks_dirname, str)
    assert isinstance(chunks_type, str)
    assert isinstance(chunk_factor, int)
    assert isinstance(tile_size, int)
    assert isinstance(overlap, int)
    assert chunk_factor > 0
    assert tile_size > overlap
    assert overlap >= 0
    makedirs(chunks_dirname)
    chunk_size = overlap + (tile_size - overlap) * chunk_factor
    os_obj = os.OpenSlide(svs_filename)
    level = 0
    width, height = os_obj.level_dimensions[level]
    chunk_left = range(0, width - tileSize + 1, chunk_size - overlap)
    chunk_right = [min(left + chunk_size, width) for left in chunk_left]
    chunk_top = range(0, height - tileSize + 1, chunk_size - overlap)
    chunk_bottom = [min(top + chunk_size, height) for top in chunk_top]
    for left, right in zip(chunk_left, chunk_right):
        for top, bottom in zip(chunk_top, chunkBottom):
            # Read in chunk
            chunk = np.array(os_obj.read_region((left, top), level, (right - left, bottom - top)))
            assert len(chunk.shape) == 3
            assert 3 <= chunk.shape[2] <= 4
            # Make sure that we have RGB rather than RGBA
            image = Image.fromarray(chunk[..., :3])
            # Write out chunk using {chunks_type} as file suffix to specify the type
            filename = f"{chunks_dirname}/{left}.{top}.{chunks_type}"
            # print(filename)
            image.save(filename, **kwargs)


# _convert_openslide_to_chunks("TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913.svs", "jpeg95.stack", "jpeg", quality=95)
# _convert_openslide_to_chunks("TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913.svs", "jpeg75.stack", "jpeg", quality=75)
# _convert_openslide_to_chunks("TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913.svs", "jpeg50.stack", "jpeg", quality=50)
# _convert_openslide_to_chunks("TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913.svs", "jpeg25.stack", "jpeg", quality=25)
# _convert_openslide_to_chunks("TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913.svs", "jpeg2000.stack", "jpeg2000")
# _convert_openslide_to_chunks("TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913.svs", "j2k.stack", "j2k")
# _convert_openslide_to_chunks("TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913.svs", "jpx.stack", "jpx")
# _convert_openslide_to_chunks("TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913.svs", "png.stack", "png")
