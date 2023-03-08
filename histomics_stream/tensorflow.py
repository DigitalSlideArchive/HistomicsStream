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

import datetime
import math
import tensorflow as tf
from . import configure


class CreateTensorFlowDataset(configure.ChunkLocations):
    def __init__(self):
        configure.ChunkLocations.__init__(self)
        self.dataset_map_options = {
            "num_parallel_calls": tf.data.experimental.AUTOTUNE,
            "deterministic": False,
        }

    def __call__(self, study_description, num_workers=1, worker_index=0):
        """
        From scratch, creates a tensorflow dataset with one tensorflow element per tile
        """

        # Call to superclass to find the locations for the chunks
        # print(f"Build chunks: begin {datetime.datetime.now()}")
        configure.ChunkLocations.__call__(self, study_description)
        # print(f"Build chunks: end {datetime.datetime.now()}")

        # Build one record for each chunk
        _singular = {"tiles_top": "tile_top", "tiles_left": "tile_left"}
        # print(f"Build chunk_list_by_row: begin {datetime.datetime.now()}")
        chunk_list_by_row = [
            {
                **{
                    key: value
                    for key, value in study_description.items()
                    if key != "slides"
                },
                **{
                    key: value
                    for key, value in slide_description.items()
                    if key not in ("tiles", "chunks")
                },
                **{
                    key: value
                    for key, value in chunk_description.items()
                    if key != "tiles"
                },
                **{
                    key: [
                        tile_description[_singular[key]]
                        for tile_description in chunk_description["tiles"].values()
                    ]
                    for key in ("tiles_top", "tiles_left")
                },
            }
            for slide_description in study_description["slides"].values()
            for chunk_description in slide_description["chunks"].values()
        ]
        # print(f"Build chunk_list_by_row: end {datetime.datetime.now()}")

        if False:

            def gen():
                for row in chunk_list_by_row:
                    yield {
                        key: (
                            tf.constant
                            if key not in ("tiles_top", "tiles_left")
                            else tf.ragged.constant
                        )([value])
                        for key, value in row.items()
                    }

            # This approach is not used because TensorFlow is rejecting output_signature
            # = study_dataset.element_spec (and variations thereof).
            output_signature = {
                "version": tf.TensorSpec(shape=(), dtype=tf.string),
                "tile_width": tf.TensorSpec(shape=(), dtype=tf.int32),
                "tile_height": tf.TensorSpec(shape=(), dtype=tf.int32),
                "overlap_height": tf.TensorSpec(shape=(), dtype=tf.int32),
                "overlap_width": tf.TensorSpec(shape=(), dtype=tf.int32),
                "filename": tf.TensorSpec(shape=(), dtype=tf.string),
                "slide_name": tf.TensorSpec(shape=(), dtype=tf.string),
                "slide_group": tf.TensorSpec(shape=(), dtype=tf.string),
                "target_magnification": tf.TensorSpec(shape=(), dtype=tf.float32),
                "scan_magnification": tf.TensorSpec(shape=(), dtype=tf.float32),
                "read_magnification": tf.TensorSpec(shape=(), dtype=tf.float32),
                "returned_magnification": tf.TensorSpec(shape=(), dtype=tf.float64),
                "level": tf.TensorSpec(shape=(), dtype=tf.int32),
                "slide_width": tf.TensorSpec(shape=(), dtype=tf.int32),
                "slide_height": tf.TensorSpec(shape=(), dtype=tf.int32),
                "chunk_height": tf.TensorSpec(shape=(), dtype=tf.int32),
                "chunk_width": tf.TensorSpec(shape=(), dtype=tf.int32),
                "slide_height_tiles": tf.TensorSpec(shape=(), dtype=tf.int32),
                "slide_width_tiles": tf.TensorSpec(shape=(), dtype=tf.int32),
                "chunk_top": tf.TensorSpec(shape=(), dtype=tf.int32),
                "chunk_left": tf.TensorSpec(shape=(), dtype=tf.int32),
                "chunk_bottom": tf.TensorSpec(shape=(), dtype=tf.int32),
                "chunk_right": tf.TensorSpec(shape=(), dtype=tf.int32),
                "tiles_top": tf.RaggedTensorSpec(
                    tf.TensorShape([None]), tf.int32, 0, tf.int64
                ),
                "tiles_left": tf.RaggedTensorSpec(
                    tf.TensorShape([None]), tf.int32, 0, tf.int64
                ),
            }
            study_dataset = tf.data.Dataset.from_generator(gen, output_signature)
        else:
            # print(f"Build chunk_list_by_column: begin {datetime.datetime.now()}")
            chunk_list_by_column = {
                key: (
                    tf.constant
                    if key not in ("tiles_top", "tiles_left")
                    else tf.ragged.constant
                )([chunk[key] for chunk in chunk_list_by_row])
                for key in chunk_list_by_row[0].keys()
            }
            # print(f"Build chunk_list_by_column: end {datetime.datetime.now()}")

            del chunk_list_by_row
            # print(f"Build study_dataset from_tensor_slices: begin {datetime.datetime.now()}")
            study_dataset = tf.data.Dataset.from_tensor_slices(chunk_list_by_column)
            # print(f"Build study_dataset from_tensor_slices: end {datetime.datetime.now()}")
            del chunk_list_by_column

            # print(f"{study_dataset.element_spec = }")

        # Shard the dataset
        if num_workers != 1 or worker_index != 0:
            study_dataset = study_dataset.shard(num_workers, worker_index)

        # We have accumulated the chunk datasets into a study_dataset where each element
        # is a chunk.  Read in the chunk pixel data and split it into tiles.
        # print(f"Build study_dataset map: begin {datetime.datetime.now()}")
        study_dataset = study_dataset.map(
            self._read_and_split_chunk, **self.dataset_map_options
        )
        # print(f"Build study_dataset map: end {datetime.datetime.now()}")

        # Change study_dataset so that each element is a tile.
        study_dataset = study_dataset.unbatch()

        # Make the tile pixels easier to find in each study_dataset element.  Also, tack
        # on additional elements to the tuple so that the form is (inputs, targets,
        # sample_weights).
        # print(f"Build study_dataset pop: begin {datetime.datetime.now()}")
        study_dataset = study_dataset.map(
            lambda elem: ((elem.pop("tile_pixels"), elem),), **self.dataset_map_options
        )
        study_dataset = study_dataset.map(
            lambda elem: (elem, None, None), **self.dataset_map_options
        )
        # print(f"Build study_dataset pop: end {datetime.datetime.now()}")
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
            func=CreateTensorFlowDataset._py_read_chunk,
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

    @staticmethod
    def _py_read_chunk(
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
        chunk = configure.ChunkLocations.read_large_image(
            filename,
            chunk_top,
            chunk_left,
            chunk_bottom,
            chunk_right,
            returned_magnification,
        )

        # Do we want to support other than RGB and/or other than uint8?!!!
        return tf.convert_to_tensor(chunk[..., :3], dtype=tf.uint8)
