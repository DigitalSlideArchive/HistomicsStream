"""Whole-slide image file reader for TensorFlow.

The histomics_stream.dsm.chunk module supports transformations that operate on a tensorflow.data.Dataset
that has one element per chunk.  (A chunk is the unit that is read from disk.  It is smaller than the
whole slide image but larger than a tile to minimize reads for performance.)  This module defines
objects that can be supplied to the tf.data.Dataset.map() method.

"""

from napari_lazy_openslide import OpenSlideStore
from PIL import Image

# import fsspec
import numpy as np
import openslide as os
import re
import tensorflow as tf
import tifffile


class ReadAndSplitChunk:
    """A class that reads a chunk from disk and splits it into tiles.

    An instance of class histomics_stream.dsm.chunk.ReadAndSplitChunk can be supplied as an argument to
    tensorflow.dataset.map.  histomics_stream.dsm.chunk.ReadAndSplitChunk reads each chunk from disk
    (for chunks containing at least one masked tile), splits the chunk into tiles, and discards unmasked
    tiles.  The result is a dataset where each element is set of tiles batched by chunk.  Calling
    .unbatch() will transform this into an unbatched dataset of tile elements.

    """

    @tf.function
    def __call__(self, elem):
        """This method is called by tensorflow to do the work of this class."""

        zero8 = tf.constant(0, dtype=tf.uint8)
        zero32 = tf.constant(0, dtype=tf.int32)
        one8 = tf.constant(1, dtype=tf.uint8)
        one32 = tf.constant(1, dtype=tf.int32)

        left_bound = tf.maximum(zero32, elem["cw"] - elem["tw"] + one32)
        tile_left = tf.range(zero32, left_bound, elem["tw"] - elem["ow"])
        tile_right = tf.clip_by_value(tile_left + elem["tw"], zero32, elem["cw"])
        usable_mask_width = tf.size(tile_left)

        top_bound = tf.maximum(zero32, elem["ch"] - elem["th"] + one32)
        tile_top = tf.range(zero32, top_bound, elem["th"] - elem["oh"])
        tile_bottom = tf.clip_by_value(tile_top + elem["th"], zero32, elem["ch"])
        usable_mask_height = tf.size(tile_top)

        x = tf.tile(tile_left, tf.stack([tf.size(tile_top)]))
        w = tf.tile(tile_right - tile_left, tf.stack([tf.size(tile_top)]))
        y = tf.repeat(tile_top, tf.size(tile_left))
        h = tf.repeat(tile_bottom - tile_top, tf.size(tile_left))
        len = tf.size(x)
        tiles = tf.TensorArray(dtype=tf.uint8, size=len)

        mask_chunk = elem["mask_chunk"]

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

        for i in tf.range(len):
            tiles = tiles.write(
                i,
                tf.image.crop_to_bounding_box(
                    chunk,
                    tf.gather(y, i),
                    tf.gather(x, i),
                    tf.gather(h, i),
                    tf.gather(w, i),
                ),
            )

        tiles = tiles.stack()

        # Figure out which tiles we are going to keep
        where = tf.where(
            tf.reshape(
                mask_chunk[:usable_mask_height, :usable_mask_width, 0],
                [
                    len,
                ],
            )
        )
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
            # elif True:
            #     # Use fsspec to read the SVS image
            #     # This is NOT working code!!!
            #     with fsspec.open(filename) as store:
            #         source_group = zarr.open(store, mode="r")
            #         # Zarr formats other than using zarr-jpeg package have shape (height, width, colors)
            #         # using order="C".
            #         chunk = source_group[format(level.numpy())][
            #             y.numpy() : (y.numpy() + h.numpy()),
            #             x.numpy() : (x.numpy() + w.numpy()),
            #             :,
            #         ]
            # elif False:
            #     # Read the SVS image with napari_lazy_openslide.lazy_openslide.OpenSlideStore
            #     store = OpenSlideStore(filename, tilesize=2048)
            #     source_group = zarr.open(store, mode="r")
            #     # Zarr formats other than zarr-jpeg have shape (height, width, colors) using order="C".
            #     chunk = source_group[format(level.numpy())][
            #         y.numpy() : (y.numpy() + h.numpy()),
            #         x.numpy() : (x.numpy() + w.numpy()),
            #         :,
            #     ]
            # else:
            #     # Use tifffile to read the SVS image 'aszarr'
            #     # store = tifffile.imread(filename)
            #     store = tifffile.imread(filename, aszarr=True)
            #     # store = tifffile.imread(filename, aszarr=True, chunkmode="page")
            #     source_group = zarr.open(store, mode="r")
            #     chunk = source_group[format(level.numpy())][
            #         y.numpy() : (y.numpy() + h.numpy()),
            #         x.numpy() : (x.numpy() + w.numpy()),
            #         :,
            #     ]

        # elif re.compile(r"\.zarr$").search(filename):
        #     # `filename` is a directory that stores an image in Zarr format.
        #     store = zarr.DirectoryStore(filename)
        #     source_group = zarr.open(store, mode="r")
        #     # Zarr formats other than using zarr-jpeg package have shape (height, width, colors) using
        #     # order="C".
        #     chunk = source_group[format(level.numpy())][
        #         y.numpy() : (y.numpy() + h.numpy()),
        #         x.numpy() : (x.numpy() + w.numpy()),
        #         :,
        #     ]
        #     # Do chunk width and height need to be transposed to be consistent with SVS?!!!

        else:
            pil_obj = Image.open(filename)
            chunk = np.asarray(pil_obj)[
                x.numpy() : (x.numpy() + w.numpy()),
                y.numpy() : (y.numpy() + h.numpy()),
                :,
            ]

        # Do we want to support other than uint8?!!!
        return tf.convert_to_tensor(chunk[..., :3], dtype=tf.uint8)
