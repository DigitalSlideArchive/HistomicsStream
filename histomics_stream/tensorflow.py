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

        # print(f"Build one_chunk_per_slice: begin {datetime.datetime.now()}")
        study_keys = study_description
        slide_keys = next(iter(study_keys["slides"].values()))
        chunk_keys = next(iter(slide_keys["chunks"].values()))
        tile_keys = {"tiles_top": "tile_top", "tiles_left": "tile_left"}
        one_chunk_per_slice = {
            **{
                key: tf.constant(
                    [
                        study_description[key]
                        for slide_description in study_description["slides"].values()
                        for chunk_description in slide_description["chunks"].values()
                    ]
                )
                for key in study_keys
                if key != "slides"
            },
            **{
                key: tf.constant(
                    [
                        slide_description[key]
                        for slide_description in study_description["slides"].values()
                        for chunk_description in slide_description["chunks"].values()
                    ]
                )
                for key in slide_keys
                if key not in ("tiles", "chunks")
            },
            **{
                key: tf.constant(
                    [
                        chunk_description[key]
                        for slide_description in study_description["slides"].values()
                        for chunk_description in slide_description["chunks"].values()
                    ]
                )
                for key in chunk_keys
                if key != "tiles"
            },
            **{
                plural: tf.ragged.constant(
                    [
                        [
                            tile_description[singular]
                            for tile_description in chunk_description["tiles"].values()
                        ]
                        for slide_description in study_description["slides"].values()
                        for chunk_description in slide_description["chunks"].values()
                    ]
                )
                for plural, singular in tile_keys.items()
            },
        }
        # print(f"Build one_chunk_per_slice: end {datetime.datetime.now()}")

        # print(
        #     "Build study_dataset from_tensor_slices: begin "
        #     f"{datetime.datetime.now()}"
        # )
        study_dataset = tf.data.Dataset.from_tensor_slices(one_chunk_per_slice)
        del one_chunk_per_slice
        # print(
        #     f"Build study_dataset from_tensor_slices: end {datetime.datetime.now()}"
        # )

        # print(f"study_dataset.element_spec = {study_dataset.element_spec}")

        # Shard the dataset
        if num_workers != 1 or worker_index != 0:
            study_dataset = study_dataset.shard(num_workers, worker_index)

        # We have accumulated the chunk datasets into a study_dataset where each element
        # is a chunk.  Read in the chunk pixel data and split it into tiles.
        # print(f"Build study_dataset map: begin {datetime.datetime.now()}")
        study_dataset = study_dataset.map(
            CreateTensorFlowDataset._read_and_split_chunk, **self.dataset_map_options
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

    @staticmethod
    def _read_and_split_chunk(elem):
        # Get chunk's pixel data from disk and load it into chunk_as_tensor.
        # Note that if elem["factor"] differs from 1.0 then this chunk will have
        # num_rows ((chunk_bottom - chunk_top) / factor, and num_columns =
        # ((chunk_right - chunk_left) / factor.
        # tf.print("#_read_and_split_chunk begin")
        zero = tf.constant(0, dtype=tf.int32)
        one = tf.constant(1, dtype=tf.int32)
        epsilon = tf.constant(0.01, dtype=tf.float32)

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
                tf.cast(elem["tile_height"], dtype=tf.float32) / factor + epsilon
            ),
            dtype=tf.int32,
        )
        scaled_tile_width = tf.cast(
            tf.math.floor(
                tf.cast(elem["tile_width"], dtype=tf.float32) / factor + epsilon
            ),
            dtype=tf.int32,
        )
        scaled_chunk_top = tf.cast(
            tf.math.floor(
                tf.cast(elem["chunk_top"], dtype=tf.float32) / factor + epsilon
            ),
            dtype=tf.int32,
        )
        scaled_chunk_left = tf.cast(
            tf.math.floor(
                tf.cast(elem["chunk_left"], dtype=tf.float32) / factor + epsilon
            ),
            dtype=tf.int32,
        )

        def condition(i, _):
            return tf.less(i, num_tiles)

        def body(i, tiles):
            return (
                i + one,
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
                                + epsilon
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
                                + epsilon
                            ),
                            dtype=tf.int32,
                        )
                        - scaled_chunk_left,
                        scaled_tile_height,
                        scaled_tile_width,
                    ),
                ),
            )

        _, tiles = tf.while_loop(condition, body, [zero, tiles])
        tiles = tiles.stack()

        response = {
            **{
                key: tf.repeat(elem[key], num_tiles)
                for key in elem.keys()
                if key not in ("tiles_top", "tiles_left")
            },
            "tile_top": elem["tiles_top"],
            "tile_left": elem["tiles_left"],
            "tile_pixels": tiles,
        }

        # tf.print("#_read_and_split_chunk end")
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

        # if "_num_chunks" not in CreateTensorFlowDataset._py_read_chunk.__dict__:
        #     CreateTensorFlowDataset._py_read_chunk._num_chunks = 0
        # chunk_name = (
        #     f"#_py_read_chunk {CreateTensorFlowDataset._py_read_chunk._num_chunks:06}"
        # )
        # CreateTensorFlowDataset._py_read_chunk._num_chunks += 1

        # print(f"{chunk_name} begin {datetime.datetime.now()}")
        filename = filename.numpy().decode("utf-8")
        chunk_top = math.floor(chunk_top.numpy() / factor.numpy() + 0.01)
        chunk_left = math.floor(chunk_left.numpy() / factor.numpy() + 0.01)
        chunk_bottom = math.floor(chunk_bottom.numpy() / factor.numpy() + 0.01)
        chunk_right = math.floor(chunk_right.numpy() / factor.numpy() + 0.01)
        returned_magnification = returned_magnification.numpy()

        # print(f"{chunk_name} begin1 {datetime.datetime.now()}")
        # Call to the superclass to get the pixel data for this chunk
        chunk = configure.ChunkLocations.read_large_image(
            filename,
            chunk_top,
            chunk_left,
            chunk_bottom,
            chunk_right,
            returned_magnification,
        )
        # print(f"{chunk_name} begin2 {datetime.datetime.now()}")

        # Do we want to support other than RGB?!!!
        chunk = chunk[..., :3]
        # print(f"{chunk_name} end {datetime.datetime.now()}")
        return chunk
