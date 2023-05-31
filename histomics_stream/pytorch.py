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

import numpy as np
import torch
from . import configure

"""
See: How to load a list of numpy arrays to pytorch dataset loader?
https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader

torchvision.transforms.ToTensor transforms numpy array or PIL image to torch tensor
torchvision.transforms.LongTensor maybe stacks tensors
Subclassing torch.utils.data.Dataset maybe provides something like a tensorflow
  dataset's interface
Using a torch.utils.data.DataLoader on the torch.utils.data.Dataset subclass is maybe
  like creating the actual dataset.
"""

"""
See: A Comprehensive Guide to the DataLoader Class and Abstractions in PyTorch
https://blog.paperspace.com/dataloaders-abstractions-pytorch/
"""

"""
Notes 5/31/2023: For multi-processing, torch seems to like a single shared
torch.utils.data.IterableDataset, but one torch.utils.data.DataLoader per worker.  If we
are to avoid loading in all workers' pixel data for each worker, the Dataset should not
be loading in the pixel data, just creating the associated dictionary.  There should be
one dictionary per *chunk*, which includes its list of tiles, and the loading of the
pixel data per chunk should somehow be deferred to the DataLoader.

If we create the dataset in an eager fashion, which may be reasonable if it is not
including the pixel data, then it can instead be a (map-style rather iterable-style)
torch.utils.data.Dataset.  Especially if we compute worker_index = chunk_index %
num_workers as part of the annotation, it might be quite easy to use a DataLoader's
`num_workers` and `sampler` parameters to direct that pixel data are read only for those
chunks that belong to a given worker, at the time that the DataLoader is created.

Ultimately the goal is to have the pixel data read and predicted within a single worker
before it is grouped back together to return to the user.  If the above doesn't work for
that, alternatively to using `num_workers` in the DataLoader constructor, we might
explicitly use num_worker instances of a DataLoader, created using num_worker calls to
torch.multiprocessing.Process(target=DataLoader, args=) or similar.  These are started
in one loop and then joined in another loop.  See
https://pytorch.org/docs/stable/notes/multiprocessing.html.  We'll probably need a
torch.multiprocessing.Queue to collect outputs, similarly to but not quite the same as
https://teddykoker.com/2020/12/dataloader/.
"""


class CreateTorchDataloader(configure.ChunkLocations):
    class MyDataset(torch.utils.data.IterableDataset, configure._TilesByCommon):
        def __init__(self, study_description):
            configure._TilesByCommon.__init__(self)
            torch.utils.data.IterableDataset.__init__(self)
            """Store in self the data or pointers to it"""
            # Update keys of the dictionary from deprecated names
            self._update_dict(study_description)
            for slide_description in study_description["slides"].values():
                self._update_dict(slide_description)
                for chunk_description in slide_description["chunks"].values():
                    self._update_dict(chunk_description)
                    for tile_description in chunk_description["tiles"].values():
                        self._update_dict(tile_description)

            self.study_description = study_description

        def __iter__(self):
            """Return an iterable that yields tiles=(pixel data, annotation_dict)"""

            def my_iterable():
                """This is the iterable that we will return"""
                study_description = self.study_description
                study_dict = {
                    # !!! Is it better to have the dictionary values be length-one
                    # !!! lists, here and below?
                    # !!! Or use
                    # !!! {key: torch.from_numpy(np.array(study_description[key]))}?
                    key: study_description[key]
                    for key in study_description.keys()
                    if key != "slides"
                }
                for slide_description in study_description["slides"].values():
                    slide_dict = {
                        **study_dict,
                        **{
                            key: slide_description[key]
                            for key in slide_description.keys()
                            if key not in ["tiles", "chunks"]
                        },
                    }

                    filename = slide_dict["filename"]
                    returned_magnification = slide_dict["returned_magnification"]
                    factor = slide_dict["target_magnification"] / returned_magnification
                    scaled_tile_height = configure.ChunkLocations.scale_it(
                        slide_dict["tile_height"], factor
                    )
                    scaled_tile_width = configure.ChunkLocations.scale_it(
                        slide_dict["tile_width"], factor
                    )

                    for chunk_description in slide_description["chunks"].values():
                        chunk_dict = {
                            **slide_dict,
                            **{
                                key: chunk_description[key]
                                for key in chunk_description.keys()
                                if key != "tiles"
                            },
                        }

                        # Call to the superclass to get the pixel data for this chunk.
                        # Keep only first 3 colors.  Convert to np.uint8.
                        scaled_chunk_top = configure.ChunkLocations.scale_it(
                            chunk_dict["chunk_top"], factor
                        )
                        scaled_chunk_left = configure.ChunkLocations.scale_it(
                            chunk_dict["chunk_left"], factor
                        )
                        scaled_chunk_bottom = configure.ChunkLocations.scale_it(
                            chunk_dict["chunk_bottom"], factor
                        )
                        scaled_chunk_right = configure.ChunkLocations.scale_it(
                            chunk_dict["chunk_right"], factor
                        )

                        # Use `:3` to change RGBA (if applicable) to RGB.
                        scaled_chunk_pixels = configure.ChunkLocations.read_large_image(
                            filename,
                            scaled_chunk_top,
                            scaled_chunk_left,
                            scaled_chunk_bottom,
                            scaled_chunk_right,
                            returned_magnification,
                        )[..., :3].astype(dtype=np.float32)
                        # Color is the last/fastest dimension for images read with
                        # large_image, but channel is the first/slowest for Torch
                        # tensors.
                        scaled_chunk_pixels = np.moveaxis(scaled_chunk_pixels, -1, 0)
                        scaled_chunk_pixels = torch.from_numpy(scaled_chunk_pixels)

                        for tile_description in chunk_description["tiles"].values():
                            tile_dict = {
                                **chunk_dict,
                                **{
                                    key: tile_description[key]
                                    for key in tile_description.keys()
                                },
                            }
                            scaled_tile_top = (
                                configure.ChunkLocations.scale_it(
                                    tile_dict["tile_top"], factor
                                )
                                - scaled_chunk_top
                            )
                            scaled_tile_left = (
                                configure.ChunkLocations.scale_it(
                                    tile_dict["tile_left"], factor
                                )
                                - scaled_chunk_left
                            )
                            scaled_tile_bottom = scaled_tile_top + scaled_tile_height
                            scaled_tile_right = scaled_tile_left + scaled_tile_width
                            scaled_tile_pixels = scaled_chunk_pixels[
                                :,
                                scaled_tile_top:scaled_tile_bottom,
                                scaled_tile_left:scaled_tile_right,
                            ]

                            # Yield the pixel data as a tensor and the Python dict of
                            # associated information.  Rather than `yield
                            # scaled_tile_pixels, tile_dict` we use lists and pop() so
                            # that this iterator does not maintain a reference count for
                            # the returned objects.
                            pixels_in_list = [scaled_tile_pixels]
                            dict_in_list = [tile_dict]
                            del scaled_tile_pixels, tile_dict
                            yield pixels_in_list.pop(), dict_in_list.pop()

            """Return this generator (iterable) over the tiles"""
            return my_iterable()

    def __init__(self):
        """Set global options"""
        configure.ChunkLocations.__init__(self)
        # !!! Instead, get `batch_size` from somewhere
        self.batch_size = 1

    def __call__(self, study_description):
        """
        From scratch, creates a torch dataloader with one torch element per tile
        """
        # Call to superclass to find the locations for the chunks
        super().__call__(study_description)

        my_dataset = self.MyDataset(study_description)
        # !!! DataLoader has additional parameters that we may wish to use
        my_data_loader = torch.utils.data.DataLoader(
            my_dataset, batch_size=self.batch_size
        )

        return my_data_loader
