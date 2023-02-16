# =========================================================================
#
#   Copyright NumFOCUS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# =========================================================================

"""Whole-slide image streamer for machine learning frameworks."""

import math
import tensorflow as tf

from . import configure


class CreateTensorFlowDataset(configure.ChunkLocations):
    def __init__(self):
        self.dataset_map_options = {
            "num_parallel_calls": tf.data.experimental.AUTOTUNE,
            "deterministic": False,
        }

    def __call__(self, study_description):
        """
        From scratch, creates a tensorflow dataset with one tensorflow element per tile
        """

        # Call to superclass to find the locations for the chunks
        configure.ChunkLocations.__call__(self, study_description)

        # Start converting our description into tensors.
        study_as_tensors = {
            study_key: [tf.convert_to_tensor(study_description[study_key])]
            for study_key in study_description.keys()
            if study_key != "slides"
        }
        # print("study_as_tensors done")

        num_chunks = 0
        for slide_description in study_description["slides"].values():
            slide_as_tensors = {
                **study_as_tensors,
                **{
                    slide_key: [tf.convert_to_tensor(slide_description[slide_key])]
                    for slide_key in slide_description.keys()
                    if slide_key not in ["tiles", "chunks"]
                },
            }

            for chunk_description in slide_description["chunks"].values():
                chunk_as_tensors = {
                    **slide_as_tensors,
                    **{
                        chunk_key: [tf.convert_to_tensor(chunk_description[chunk_key])]
                        for chunk_key in chunk_description.keys()
                        if chunk_key != "tiles"
                    },
                    "tiles_top": [
                        tf.convert_to_tensor(
                            [
                                tile["tile_top"]
                                for tile in chunk_description["tiles"].values()
                            ]
                        )
                    ],
                    "tiles_left": [
                        tf.convert_to_tensor(
                            [
                                tile["tile_left"]
                                for tile in chunk_description["tiles"].values()
                            ]
                        )
                    ],
                }

                # Make a tensorflow Dataset from this chunk.
                chunk_dataset = tf.data.Dataset.from_tensor_slices(chunk_as_tensors)
                if num_chunks == 0:
                    study_dataset = chunk_dataset
                else:
                    study_dataset = study_dataset.concatenate(chunk_dataset)
                num_chunks += 1

        # We have accumulated the chunk datasets into a study_dataset where each element
        # is a chunk.  Read in the chunk pixel data and split it into tiles.
        study_dataset = study_dataset.map(
            self._read_and_split_chunk, **self.dataset_map_options
        )
        # print("_read_and_split_chunk done")
        # Change study_dataset so that each element is a tile.
        study_dataset = study_dataset.unbatch()
        # print("unbatch done")

        # Make the tile pixels easier to find in each study_dataset element.  Also, tack
        # on additional elements to the tuple so that the form is (inputs, targets,
        # sample_weights).
        study_dataset = study_dataset.map(
            lambda elem: ((elem.pop("tile_pixels"), elem), None, None),
            **self.dataset_map_options,
        )
        # print("pop done")

        return study_dataset

    @tf.function
    def _read_and_split_chunk(self, elem):
        # Get chunk's pixel data from disk and load it into chunk_as_tensor.
        # Note that if elem["factor"] differs from 1.0 then this chunk will have
        # num_rows ((chunk_bottom - chunk_top) / factor, and num_columns =
        # ((chunk_right - chunk_left) / factor.
        factor = tf.cast(elem["target_magnification"], dtype=tf.float32) / tf.cast(
            elem["returned_magnification"], dtype=tf.float32
        )
        chunk_as_tensor = tf.py_function(
            func=self._py_read_chunk,
            inp=[
                elem["chunk_top"],
                elem["chunk_left"],
                elem["chunk_bottom"],
                elem["chunk_right"],
                elem["filename"],
                elem["returned_magnification"],
                factor,
            ],
            Tout=tf.uint8,
        )
        num_tiles = tf.size(elem["tiles_top"])
        tiles = tf.TensorArray(dtype=tf.uint8, size=num_tiles)

        scaled_tile_height = tf.cast(
            tf.math.floor(
                tf.cast(elem["tile_height"], dtype=tf.float32) / factor
                + tf.convert_to_tensor(0.01, dtype=tf.float32)
            ),
            dtype=tf.int32,
        )
        scaled_tile_width = tf.cast(
            tf.math.floor(
                tf.cast(elem["tile_width"], dtype=tf.float32) / factor
                + tf.convert_to_tensor(0.01, dtype=tf.float32)
            ),
            dtype=tf.int32,
        )
        scaled_chunk_top = tf.cast(
            tf.math.floor(
                tf.cast(elem["chunk_top"], dtype=tf.float32) / factor
                + tf.convert_to_tensor(0.01, dtype=tf.float32)
            ),
            dtype=tf.int32,
        )
        scaled_chunk_left = tf.cast(
            tf.math.floor(
                tf.cast(elem["chunk_left"], dtype=tf.float32) / factor
                + tf.convert_to_tensor(0.01, dtype=tf.float32)
            ),
            dtype=tf.int32,
        )

        def condition(i, _):
            return tf.less(i, num_tiles)

        def body(i, tiles):
            return (
                i + 1,
                tiles.write(
                    i,
                    tf.image.crop_to_bounding_box(
                        chunk_as_tensor,
                        tf.cast(
                            tf.math.floor(
                                tf.cast(
                                    tf.gather(elem["tiles_top"], i), dtype=tf.float32
                                )
                                / factor
                                + tf.convert_to_tensor(0.01, dtype=tf.float32)
                            ),
                            dtype=tf.int32,
                        )
                        - scaled_chunk_top,
                        tf.cast(
                            tf.math.floor(
                                tf.cast(
                                    tf.gather(elem["tiles_left"], i), dtype=tf.float32
                                )
                                / factor
                                + tf.convert_to_tensor(0.01, dtype=tf.float32)
                            ),
                            dtype=tf.int32,
                        )
                        - scaled_chunk_left,
                        scaled_tile_height,
                        scaled_tile_width,
                    ),
                ),
            )

        _, tiles = tf.while_loop(condition, body, [0, tiles])
        tiles = tiles.stack()

        response = {}
        for key in elem.keys():
            if key not in ("tiles_top", "tiles_left"):
                response[key] = tf.repeat(elem[key], num_tiles)

        response = {
            **response,
            "tile_top": elem["tiles_top"],
            "tile_left": elem["tiles_left"],
            "tile_pixels": tiles,
        }
        return response

    def _py_read_chunk(
        self,
        chunk_top,
        chunk_left,
        chunk_bottom,
        chunk_right,
        filename,
        returned_magnification,
        factor,
    ):
        """
        Read from disk all the pixel data for a specific chunk of the
        whole slide.
        """

        filename = filename.numpy().decode("utf-8")
        chunk_top = math.floor(chunk_top.numpy() / factor.numpy() + 0.01)
        chunk_left = math.floor(chunk_left.numpy() / factor.numpy() + 0.01)
        chunk_bottom = math.floor(chunk_bottom.numpy() / factor.numpy() + 0.01)
        chunk_right = math.floor(chunk_right.numpy() / factor.numpy() + 0.01)
        returned_magnification = returned_magnification.numpy()

        # Call to the superclass to get the pixel data for this chunk
        chunk = self.read_large_image(
            filename,
            chunk_top,
            chunk_left,
            chunk_bottom,
            chunk_right,
            returned_magnification,
        )

        # Do we want to support other than RGB and/or other than uint8?!!!
        return tf.convert_to_tensor(chunk[..., :3], dtype=tf.uint8)
