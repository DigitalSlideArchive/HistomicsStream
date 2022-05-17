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

import numpy as np
import torch

from . import configure


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        """Store in self the data or pointers to it"""
        pass                    # Write me!!!

    def __getitem__(self, index):
        """Return x, y, which is an (input, label) tuple"""
        pass                    # Write me!!!

    def __len__(self):
        """Return the total number of data items that will be available"""
        pass                    # Write me!!!


class CreateTorchDataset(configure.ChunkLocations):
    def __init__(self):
        pass  # Write me!!!

    def __call__(self, study_description):
        """
        From scratch, creates a torch dataset with one torch element per tile
        """
        pass  # Write me!!!
