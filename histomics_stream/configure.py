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

import copy
import itertools
import itk
import large_image
import large_image_source_tiff
import math
import numpy as np
import os
import random
import re
import scipy.interpolate


class _TilesByCommon:
    def __init__(self):
        self._key_mapping = {
            "number_pixel_columns_for_chunk": "chunk_width",
            "number_pixel_columns_for_mask": "mask_width",
            "number_pixel_columns_for_slide": "slide_width",
            "number_pixel_columns_for_tile": "tile_width",
            "number_pixel_overlap_columns_for_tile": "tile_overlap_width",
            "number_pixel_overlap_rows_for_tile": "tile_overlap_height",
            "number_pixel_rows_for_chunk": "chunk_height",
            "number_pixel_rows_for_mask": "mask_height",
            "number_pixel_rows_for_slide": "slide_height",
            "number_pixel_rows_for_tile": "tile_height",
            "number_tile_columns_for_slide": "slide_width_tiles",
            "number_tile_rows_for_slide": "slide_height_tiles",
            "tile_overlap_height": "overlap_height",
            "tile_overlap_width": "overlap_width",
        }

        self._keys_warned = set()

    # For each filename, select just upper-left corner for each tile.
    # Note that each upper-left corner is returned as (top, left), not (left, top).
    @staticmethod
    def get_tiles(study):
        return [
            (
                slide["filename"],
                [
                    (tile["tile_top"], tile["tile_left"])
                    for tile in slide["tiles"].values()
                ],
            )
            for slide in study["slides"].values()
        ]

    # Private function to map old key names to their current equivalent
    def _update_dict(self, d):
        for old_key in d.keys() & self._key_mapping.keys():
            # An old key is in use in `d`.
            new_key = self._key_mapping[old_key]
            while new_key in self._key_mapping:
                # Multiple, serial name changes
                new_key = self._key_mapping[new_key]
            if new_key in d:
                # Both the old and new key are used.
                raise ValueError(
                    f"Cannot use both {repr(old_key)} key (deprecated) "
                    f"and its replacement {repr(new_key)}"
                )
            if old_key not in self._keys_warned:
                print(
                    f"Warning: updating deprecated key {repr(old_key)} "
                    f"to new name {repr(new_key)}"
                )
                # Comment out the next line so we do have repeated warnings, in case a
                # second study comes in with deprecated keys.
                # self._keys_warned.add(old_key)
            d[new_key] = d[old_key]
            del d[old_key]


class FindResolutionForSlide(_TilesByCommon):
    """
    A class that computes read parameters for slides.

    An instance of class FindResolutionForSlide is a callable that will add level,
    target_magnification, scan_magnification, read_magnification,
    returned_magnification, slide_height, and slide_width fields to a slide dictionary.

    Parameters for the constructor
    ------------------------------

    filename : string
        The path of the image file to be read.

    target_magnification : float
        The desired objective magnification for generated tiles.  For example, a value
        of 10 corresponds to about 1 micron per pixel and a value of 20 corresponds to
        about 0.5 microns per pixel.

    magnification_source : str in ["scan", "native", "exact"]
        "scan" will produce tiles from the highest magnification avaialable. This is
        typically the slide scanner's objective magnification.

        "native" will produce tiles from the nearest available magnification equal to or
        greater than target_magnification (within a 2% tolerance). The "native" option
        is useful when you want to handle resizing of tiles to target_magnification on
        your own.

        "exact" will produce tiles using "native" option and then resize these tiles to
        match target_magnification. Resizing is handled by PIL using the Lanczos
        antialiasing filter since the resizing shrinks the tile by definition.

        For either "scan" or "native", the size of the read and returned tiles will be
        (tile_height * returned_magnification / target_magnification, tile_width *
        returned_magnification / target_magnification).  For "exact" the size of the
        returned tiles will be (tile_height, tile_width).

        This procedure sets values in the slide dictionary to capture the scan, read,
        and returned magnification of the tiles. This is helpful for example to resize
        results to the scan magnification for visualization in HistomicsUI, or to resize
        between native and target magnification when using
        "native". "scan_magnification" is the highest magnification from the source
        file; "read_magnification" is the magnification read from the source file;
        "returned_magnification" is the magnification of the returned tiles which is
        same as "read_magnification" in the case of "scan" or "native" or
        "target_magnification" in the case of "exact".
    """

    def __init__(self, study, target_magnification, magnification_source):
        """
        Sanity check the supplied parameters and store them for later use.
        """
        _TilesByCommon.__init__(self)
        # Check values.
        if not ("version" in study and study["version"] == "version-1"):
            raise ValueError('study["version"] must exist and be equal to "version-1".')
        if not (
            isinstance(target_magnification, (int, np.integer, float, np.floating))
            and 0 < target_magnification
        ):
            raise ValueError(
                f"target_magnification ({target_magnification})"
                " must be a positive number"
            )
        if not (
            isinstance(magnification_source, str)
            and magnification_source in ["scan", "native", "exact"]
        ):
            raise ValueError(
                f"magnification_source ({magnification_source})"
                " must be one of {['scan', 'native', 'exact']}."
            )

        # Save values.
        self.target_magnification = float(target_magnification)
        self.magnification_source = magnification_source

    def __call__(self, slide):
        """
        Add level, target_magnification, scan_magnification, read_magnification,
        returned_magnification, slide_height, and slide_width fields to a slide
        dictionary.
        """

        # Check values.
        if "filename" not in slide:
            raise ValueError('slide["filename"] must be already set.')
        filename = slide["filename"]

        # Do the work.
        if not re.compile(r"\.zarr$").search(filename):
            import large_image

            # read whole-slide image file and create large_image object
            ts = large_image.open(filename)

            # scan_magnification = highest available magnification from source
            scan_magnification = float(ts.getNativeMagnification()["magnification"])

            if self.magnification_source == "exact":
                # Use the tile-source level that large_image is willing to interpolate
                # for us.
                preferred_levels = [
                    ts.getLevelForMagnification(
                        self.target_magnification, rounding=False
                    )
                ]
            else:  # self.magnification_source in ["scan", "native"]
                # Use one of the tile-source levels that is stored in the image file.
                preferred_levels = list(
                    set(ts.getPreferredLevel(level) for level in range(ts.levels))
                )
                preferred_levels.sort(reverse=True)
                if self.magnification_source == "scan":
                    # Keep only the maximum tile-source level
                    preferred_levels = preferred_levels[0:1]

            estimated_magnifications = np.array(
                [
                    float(ts.getMagnificationForLevel(level)["magnification"])
                    for level in preferred_levels
                ]
            )

            # Find best tile-source level to use
            (level, returned_magnification) = self._get_level_and_magnifications(
                self.target_magnification, estimated_magnifications
            )
            # Rather than as the index into preferred_levels, change level to be the
            # value that large_image uses
            level = preferred_levels[level]

            # If large_image is resampling a native level for us, it is starting with
            # the preferred level that is the least one that is not smaller than the
            # resampled level.
            read_magnification = float(
                ts.getMagnificationForLevel(
                    min(
                        [
                            ts.getPreferredLevel(i)
                            for i in range(ts.levels)
                            if i >= level
                        ]
                    )
                )["magnification"]
            )

            slide["target_magnification"] = self.target_magnification
            slide["scan_magnification"] = scan_magnification
            slide["read_magnification"] = read_magnification
            slide["returned_magnification"] = returned_magnification

            # We don't want to walk off the right or bottom of the slide so we are
            # conservative as to how many pixels large_image will return for us.
            # 1) large_image starts with an image that is of
            #    read_magnification; we compute the dimensions for read_magnification
            #    with math.floor from the dimensions of scan_magnification (i.e.,
            #    ts.sizeX and ts.sizeY) to be conservative.
            # 2) large_image or external software may resampled from the
            #    read_magnification to the target_magnification; we compute dimensions
            #    for the target_magnification with math.floor from the
            #    read_magnification to be conservative.
            slide_height = ts.sizeY
            slide_width = ts.sizeX
            if scan_magnification != read_magnification:
                slide_height = math.floor(
                    slide_height * read_magnification / scan_magnification
                )
                slide_width = math.floor(
                    slide_width * read_magnification / scan_magnification
                )
            if read_magnification != self.target_magnification:
                slide_height = math.floor(
                    slide_height * self.target_magnification / read_magnification
                )
                slide_width = math.floor(
                    slide_width * self.target_magnification / read_magnification
                )

        else:
            import zarr
            import openslide as os

            # read whole-slide image and create zarr objects
            store = zarr.DirectoryStore(filename)
            source_group = zarr.open(store, mode="r")

            # scan_magnification = highest available magnification from source
            scan_magnification = float(
                source_group.attrs[os.PROPERTY_NAME_OBJECTIVE_POWER]
            )

            preferred_levels = list(range(0, source_group.attrs["level_downsamples"]))
            if self.magnification_source == "scan":
                preferred_levels = [np.argmin(source_group.attrs["level_downsamples"])]

            # calculate magnifications of levels
            estimated_magnifications = np.array(
                scan_magnification / source_group.attrs["level_downsamples"][level]
                for level in preferred_levels
            )

            # Find best native level to use
            (level, returned_magnification) = self._get_level_and_magnifications(
                self.target_magnification, estimated_magnifications
            )
            # Rather than as the index into preferred_levels, change level to be the
            # value that zarr uses
            level = preferred_levels[level]

            slide["target_magnification"] = self.target_magnification
            slide["scan_magnification"] = scan_magnification
            slide["read_magnification"] = returned_magnification
            slide["returned_magnification"] = returned_magnification

            # get slide slide_height, slide_width at
            # desired magnification. (Note that slide_width is before
            # slide_height)
            slide_width, slide_height = source_group[format(level)].shape[0:2]

            if (
                self.magnification_source == "exact"
                and self.target_magnification != returned_magnification
            ):
                raise ValueError(
                    f"Couldn't find magnification {self.target_magnification}X "
                    "in Zarr storage."
                )

        int_level = int(round(level))
        slide["level"] = int_level if abs(level - int_level) < 1e-4 else level
        # Note that slide size is defined by the requested magnification, which may not
        # be the same as the magnification for the selected level.  To get the slide
        # size for the magnification that we are using, these values must later be
        # multiplied by returned_magnification / target_magnification.
        slide["slide_height"] = slide_height
        slide["slide_width"] = slide_width

    @staticmethod
    def _get_level_and_magnifications(target_magnification, estimated_magnifications):
        """
        A private subroutine that computes level and magnifications.
        """
        # calculate difference with magnification levels

        magnification_tolerance = 0.02
        delta = target_magnification - estimated_magnifications

        # match to existing levels
        if (
            np.min(np.abs(np.divide(delta, target_magnification)))
            < magnification_tolerance
        ):  # match
            level = np.squeeze(np.argmin(np.abs(delta)))
        elif np.any(delta < 0):
            value = np.max(delta[delta < 0])
            level = np.squeeze(np.argwhere(delta == value)[0])
        else:  # desired magnification above base level - throw error
            raise ValueError("Cannot interpolate above scan magnification.")

        returned_magnification = estimated_magnifications[level]

        return level, returned_magnification


class TilesByGridAndMask(_TilesByCommon):
    """
    Select tiles according to a regular grid.  Optionally, restrict the list by a mask
    that is read from a file.  Optionally, further select a random subset of them.

    An instance of class TilesByGridAndMask is a callable that will select the
    coordinates of tiles to be taken from a slide.  The selected tiles will be written
    to the slide dictionary.

    Parameters for the constructor
    ------------------------------
    study : dictionary
        The study dictionary from which to read parameters about the study.
    randomly_select: int
        The number of tiles to be randomly selected from the list that would otherwise
        be written to the slide dictionary.  A value of -1 is the default and means that
        all tiles should be written.
    overlap_height
        Specifies the desired amount of vertical overlap between adjacent tiles,
        measured in pixels using the `target_magnification`.  If overlap_height is not
        supplied, it is read from the study dictionary, if available, otherwise it is
        set to zero.  Zero indicates that there is no overlap between adjacent tiles;
        they are abutting.
    overlap_width
        Specifies the desired amount of horizontal overlap between adjacent tiles,
        measured in pixels using the `target_magnification`.  If overlap_width is not
        supplied, it is read from the study dictionary, if available, otherwise it is
        set to zero.  Zero indicates that there is no overlap between adjacent tiles;
        they are abutting.
    mask_filename: string
        The path of the image file to be read and used as a mask.  The aspect ratio of
        the mask (in terms of its pixel dimensions) is expected to be about the same as
        the aspect ratio of the main image ( in terms of its grid of tiles).  A non-zero
        value in the mask indicates that the tile should be retained.  The default is
        "", which means that there is no masking.
    mask_threshold : float
        A value in [0.0, 1.0].  A tile is retained if the fraction of the tile
        overlapping non-zero pixels in the mask is at least the mask_threshold.  The
        fraction must be strictly positive when the threshold is zero; the fraction has
        to be greater than or equal to the threshold when the threshold is not zero.

    """

    def __init__(self, study, **kwargs):
        """
        Sanity check the supplied parameters and store them for later use.
        """
        _TilesByCommon.__init__(self)
        # Update keys of the dictionary from deprecated names
        self._update_dict(kwargs)
        bad_keys = kwargs.keys() - {
            "randomly_select",
            "overlap_height",
            "overlap_width",
            "mask_filename",
            "mask_threshold",
        }
        if bad_keys:
            raise ValueError(
                f"Unrecognized parameters {repr(bad_keys)} in "
                "TilesByGridAndMask.__init__"
            )

        # randomly_select defaults to select all
        randomly_select = (
            kwargs["randomly_select"] if "randomly_select" in kwargs else -1
        )
        # Defaults to no masking
        mask_filename = kwargs["mask_filename"] if "mask_filename" in kwargs else ""
        # Defaults to any overlap with the mask
        mask_threshold = kwargs["mask_threshold"] if "mask_threshold" in kwargs else 0.0

        # Update keys of the dictionary from deprecated names
        self._update_dict(study)

        # If overlap is not supplied, it is read from the study dictionary, if
        # available, otherwise it is set to zero, which is no overlap.
        overlap_height = (
            kwargs["overlap_height"]
            if "overlap_height" in kwargs
            else study["overlap_height"]
            if "overlap_height" in study
            else 0
        )
        overlap_width = (
            kwargs["overlap_width"]
            if "overlap_width" in kwargs
            else study["overlap_width"]
            if "overlap_width" in study
            else 0
        )

        # Check values.
        if not ("version" in study and study["version"] == "version-1"):
            raise ValueError('study["version"] must exist and be equal to "version-1".')
        if not (
            "tile_height" in study
            and isinstance(study["tile_height"], (int, np.integer))
            and study["tile_height"] > 0
        ):
            raise ValueError(
                'study["tile_height"]' " must exist and be a positive integer"
            )
        if not (
            "tile_width" in study
            and isinstance(study["tile_width"], (int, np.integer))
            and study["tile_width"] > 0
        ):
            raise ValueError(
                'study["tile_width"]' " must exist and be a positive integer"
            )
        if not (
            isinstance(randomly_select, (int, np.integer)) and -1 <= randomly_select
        ):
            raise ValueError(
                f"randomly_select ({randomly_select})"
                " must be a non-negative integer or -1."
            )
        if not (
            isinstance(overlap_height, (int, np.integer))
            and overlap_height < study["tile_height"]
        ):
            raise ValueError(
                f"overlap_height ({overlap_height})"
                " must be less than"
                f' tile_height ({study["tile_height"]}).'
            )
        if not (
            isinstance(overlap_width, (int, np.integer))
            and overlap_width < study["tile_width"]
        ):
            raise ValueError(
                f"overlap_width ({overlap_width})"
                " must be less than"
                f' tile_width ({study["tile_width"]}).'
            )
        if mask_filename != "":
            mask_itk = self.check_mask_filename(mask_filename)
        if not (
            isinstance(mask_threshold, (float, np.floating))
            and mask_threshold >= 0.0
            and mask_threshold <= 1.0
        ):
            raise ValueError(
                f"mask_threshold ({mask_threshold}) must be between 0 and 1 inclusive."
            )

        # Save values.  To keep garbage collection efficient don't save all of `study`.
        self.tile_height = study["tile_height"]
        self.tile_width = study["tile_width"]
        self.randomly_select = randomly_select
        self.overlap_height = overlap_height
        self.overlap_width = overlap_width
        self.mask_filename = mask_filename
        if self.mask_filename != "":
            self.mask_itk = mask_itk
        self.mask_threshold = mask_threshold
        # If the user hasn't put the overlap information into the top-level study
        # dictionary then place it there.
        if "overlap_height" not in study:
            study["overlap_height"] = self.overlap_height
        if "overlap_width" not in study:
            study["overlap_width"] = self.overlap_width
        self.studywide_overlap_height = study["overlap_height"]
        self.studywide_overlap_width = study["overlap_width"]

    def __call__(self, slide):
        """
        Select tiles according to a regular grid.  Optionally, restrict the list by a
        mask.  Optionally, select a random subset of them.
        """

        # Update keys of the dictionary from deprecated names
        self._update_dict(slide)

        # Check values.
        if "slide_height" not in slide:
            raise ValueError('slide["slide_height"] must be already set.')
        self.slide_height = slide["slide_height"]
        if "slide_width" not in slide:
            raise ValueError('slide["slide_width"] must be already set.')
        self.slide_width = slide["slide_width"]

        slide["overlap_height"] = self.overlap_height
        slide["overlap_width"] = self.overlap_width
        #
        # Do the work.
        #
        height_stride = self.tile_height - self.overlap_height
        width_stride = self.tile_width - self.overlap_width

        # Return information to the user
        slide["slide_height_tiles"] = math.floor(
            (self.slide_height - self.overlap_height) / height_stride
        )
        slide["slide_width_tiles"] = math.floor(
            (self.slide_width - self.overlap_width) / width_stride
        )

        # Find the coordinates of each tile
        top_too_large = self.slide_height - self.tile_height + 1
        left_too_large = self.slide_width - self.tile_width + 1
        top_left = np.array(
            [
                pair
                for pair in itertools.product(
                    np.arange(0, top_too_large, height_stride),
                    np.arange(0, left_too_large, width_stride),
                )
            ],
            dtype=np.int64,
        )

        if hasattr(self, "mask_itk"):
            # There is a mask that we will have to check
            (self.mask_height, self.mask_width) = self.mask_itk.shape
            # Let the user know
            slide["mask_height"] = self.mask_height
            slide["mask_width"] = self.mask_width
            slide["tiles"] = self.compute_from_mask(top_left)

        else:
            # There is no mask to check
            slide["tiles"] = {
                f"tile_{i}": {"tile_top": int(corner[0]), "tile_left": int(corner[1])}
                for i, corner in enumerate(top_left)
            }

        if 0 <= self.randomly_select < len(slide["tiles"]):
            # Choose a subset of the tiles randomly
            slide["tiles"] = dict(
                random.sample(slide["tiles"].items(), self.randomly_select)
            )

    def check_mask_filename(self, mask_filename):
        mask_itk = itk.imread(mask_filename)  # May throw exception
        if mask_itk.GetImageDimension() != 2:
            raise ValueError(
                f"The mask ({mask_filename}) should be a 2-dimensional image."
            )
        return mask_itk

    def compute_from_mask(self, top_left):
        # Check that the input and output aspect ratios are pretty close
        if (
            abs(
                math.log(
                    (self.slide_height / self.slide_width)
                    / (self.mask_height / self.mask_width)
                )
            )
            > 0.20
        ):
            raise ValueError(
                "The mask aspect ratio does not match "
                "that for the whole slide image."
            )

        # cumulative_mask[row, column] will be the number of mask_itk[r, c] (i.e.,
        # mask_itk.GetPixel((c,r))) values that are nonzero among all those with
        # both r < row and c < column; note the strict inequalities.  We have added
        # a boundary on all sides of this array -- zeros on the top and left, and a
        # duplicate row (column) on the bottom (right) -- so that we do not need to
        # do extra testing in our code at the borders.  We use int64 in case there
        # are 2^31 (~2 billion = ~ 46k by 46k) or more non-zero pixel values in our
        # mask.
        cumulative_mask = np.zeros(
            (self.mask_height + 2, self.mask_width + 2), dtype=np.int64
        )
        cumulative_mask[1 : self.mask_height + 1, 1 : self.mask_width + 1] = (
            itk.GetArrayViewFromImage(self.mask_itk).astype(bool).astype(np.int64)
        )
        cumulative_mask = np.cumsum(np.cumsum(cumulative_mask, axis=0), axis=1)

        # Define the grid for the cumulative_mask using slide (not mask!)
        # coordinates.
        grid_points = (
            np.arange(cumulative_mask.shape[0])
            * (self.slide_height / self.mask_height),
            np.arange(cumulative_mask.shape[1]) * (self.slide_width / self.mask_width),
        )

        # Tile boundaries may not line up with mask pixels, so we will need a
        # bi-linear interpolator.
        method = "linear"  # bi-linear
        interpolator = scipy.interpolate.RegularGridInterpolator(
            grid_points, cumulative_mask, method
        )
        # Find the coordinates of each tile
        top_right = top_left + np.array((0, self.tile_width))
        bottom_left = top_left + np.array((self.tile_height, 0))
        bottom_right = bottom_left + np.array((0, self.tile_width))
        # Compute the total number of mask pixels (both whole and fractional) that
        # overlap each tile.
        cumulative_by_tile = (
            interpolator(bottom_right)
            - interpolator(bottom_left)
            - interpolator(top_right)
            + interpolator(top_left)
        )
        # When the threshold is greater than zero, any `cumulative_by_tile` that is
        # greater than or equal to `threshold` is accepted.  Because we are worried
        # about rounding error, we'll use `epsilon` to let very close cases be
        # accepted.  When the threshold is exactly zero, any cumulative_by_tile that
        # is strictly greater than zero is accepted.  As `cumulative_by_tile` is,
        # `threshold` is a count of whole and fractional mask pixels.
        epsilon = 1e-6
        threshold = max(
            0.0,
            self.mask_threshold
            * (self.tile_height * self.mask_height / self.slide_height)
            * (self.tile_width * self.mask_width / self.slide_width)
            - epsilon,
        )
        return {
            f"tile_{i}": {"tile_top": int(corner[0]), "tile_left": int(corner[1])}
            for i, corner in enumerate(top_left)
            if cumulative_by_tile[i] > threshold
        }


class TilesByList(_TilesByCommon):
    """
    Select the tiles supplied by the user.  Optionally, select a random subset of them.

    An instance of class TilesByList is a callable that will select the coordinates of
    tiles to be taken from a slide.  The selected tiles will be written to the slide
    dictionary.

    Parameters for the constructor
    ------------------------------
    study : dictionary
        The study dictionary from which to read parameters about the study.
    randomly_select: int
        The number of tiles to be randomly selected from the list that would otherwise
        be written to the slide dictionary.  A value of -1 is the default and means that
        all tiles should be written.
    tiles_dictionary: dictionary
        For example, {'AB234': {'tile_top': top0, 'tile_left': left0}, 'CD43':
        {'tile_top': top1, 'tile_left': left1}, ...}.  Tiles from this list will copied
        into the slide dictionary if they are randomly selected.

    """

    def __init__(self, study, randomly_select=-1, tiles_dictionary={}):
        """
        Sanity check the supplied parameters and store them for later use.

        randomly_select defaults to "select all".

        For example,
        tiles_dictionary = {
            "AB234": {"tile_top": top0, "tile_left": left0},
            "CD43": {"tile_top": top1, "tile_left": left1},
            ...
        }
        """
        _TilesByCommon.__init__(self)

        # Update keys of the dictionary from deprecated names
        self._update_dict(study)

        # Check values
        if not ("version" in study and study["version"] == "version-1"):
            raise ValueError('study["version"] must exist and be equal to "version-1".')
        if not (
            "tile_height" in study
            and isinstance(study["tile_height"], (int, np.integer))
            and study["tile_height"] > 0
        ):
            raise ValueError(
                'study["tile_height"]' " must exist and be a positive integer"
            )
        if not (
            "tile_width" in study
            and isinstance(study["tile_width"], (int, np.integer))
            and study["tile_width"] > 0
        ):
            raise ValueError(
                'study["tile_width"]' " must exist and be a positive integer"
            )
        if not (
            isinstance(randomly_select, (int, np.integer)) and -1 <= randomly_select
        ):
            raise ValueError(
                f"randomly_select ({randomly_select})"
                " must be a non-negative integer or -1."
            )
        if not isinstance(tiles_dictionary, dict):
            raise ValueError("tiles_dictionary must be dictionary.")
        for tile_corner in tiles_dictionary.values():
            # Update keys of the dictionary from deprecated names
            self._update_dict(tile_corner)
        if not (
            all(
                [
                    isinstance(tile_corner, dict)
                    for tile_corner in tiles_dictionary.values()
                ]
            )
            and all(
                [
                    key in tile_corner.keys()
                    for tile_corner in tiles_dictionary.values()
                    for key in ("tile_top", "tile_left")
                ]
            )
            and all(
                [
                    isinstance(tile_corner[key], (int, np.integer))
                    for tile_corner in tiles_dictionary.values()
                    for key in ("tile_top", "tile_left")
                ]
            )
            and all(
                [
                    tile_corner[key] >= 0
                    for tile_corner in tiles_dictionary.values()
                    for key in ("tile_top", "tile_left")
                ]
            )
        ):
            raise ValueError(
                "tiles_dictionary must be dictionary of tiles."
                '  Each tile is a dictionary, with keys "tile_top" and "tile_left"'
                " and with values that are non-negative integers."
            )

        # Save values.  To keep garbage collection efficient don't save all of `study`,
        # just the parts that we need.
        self.tile_height = study["tile_height"]
        self.tile_width = study["tile_width"]
        self.randomly_select = randomly_select
        self.tiles_dictionary = copy.deepcopy(
            tiles_dictionary
        )  # in case user changes it later

    def __call__(self, slide):
        """
        Select the tiles supplied by the user.  Optionally, select a random subset of
        them.
        """
        slide["tiles"] = copy.deepcopy(
            self.tiles_dictionary
        )  # in case __call__ is called again.
        if 0 <= self.randomly_select < len(slide["tiles"]):
            # Choose a subset of the tiles randomly
            slide["tiles"] = dict(
                random.sample(slide["tiles"].items(), self.randomly_select)
            )


class TilesRandomly(_TilesByCommon):
    """
    Select a random subset of all possible tiles.

    An instance of class TilesRandomly is a callable that will select the coordinates of
    tiles to be taken from a slide.  The selected tiles will be written to the slide
    dictionary.

    Parameters for the constructor
    ------------------------------
    study : dictionary
        The study dictionary from which to read parameters about the study.
    randomly_select: int
        The number of tiles to be randomly selected from the slide.  The value must be
        positive.  A value of 1 is the default.

    """

    def __init__(self, study, randomly_select=1):  # Defaults to select one
        """
        Sanity check the supplied parameters and store them for later use.
        """
        _TilesByCommon.__init__(self)

        # Update keys of the dictionary from deprecated names
        self._update_dict(study)

        # Check values.
        if not ("version" in study and study["version"] == "version-1"):
            raise ValueError('study["version"] must exist and be equal to "version-1".')
        if not (
            "tile_height" in study
            and isinstance(study["tile_height"], (int, np.integer))
            and study["tile_height"] > 0
        ):
            raise ValueError(
                'study["tile_height"]' " must exist and be a positive integer"
            )
        if not (
            "tile_width" in study
            and isinstance(study["tile_width"], (int, np.integer))
            and study["tile_width"] > 0
        ):
            raise ValueError(
                'study["tile_width"]' " must exist and be a positive integer"
            )
        if not (
            isinstance(randomly_select, (int, np.integer)) and 0 <= randomly_select
        ):
            raise ValueError(
                f"randomly_select ({randomly_select})"
                " must be a non-negative integer."
            )

        # Save values.  To keep garbage collection efficient don't save all of `study`.
        self.tile_height = study["tile_height"]
        self.tile_width = study["tile_width"]
        self.randomly_select = randomly_select

    def __call__(self, slide):
        """
        Select a random subset of all possible tiles.
        """

        # Update keys of the dictionary from deprecated names
        self._update_dict(slide)

        if "slide_height" not in slide:
            raise ValueError('slide["slide_height"] must be already set.')
        if "slide_width" not in slide:
            raise ValueError('slide["slide_width"] must be already set.')

        top_too_large = slide["slide_height"] - self.tile_height + 1
        left_too_large = slide["slide_width"] - self.tile_width + 1
        slide["tiles"] = {
            f"tile_{i}": {
                "tile_top": random.randrange(0, top_too_large),
                "tile_left": random.randrange(0, left_too_large),
            }
            for i in range(self.randomly_select)
        }


class ChunkLocations(_TilesByCommon):
    def __init__(self):
        _TilesByCommon.__init__(self)
        self.no_indices = np.array((), dtype=np.int64)

    def __call__(self, study_description):
        """
        Given the list of desired tile locations, computes the locations of chunks to be
        read
        """

        # Update keys of the dictionary from deprecated names
        self._update_dict(study_description)

        if not (
            "version" in study_description
            and study_description["version"] == "version-1"
        ):
            raise ValueError(
                'study_description["version"] must exist and be equal to "version-1".'
            )
        if not (
            "tile_height" in study_description
            and isinstance(study_description["tile_height"], (int, np.integer))
            and study_description["tile_height"] > 0
        ):
            raise ValueError(
                'study_description["tile_height"]'
                " must exist and be a positive integer"
            )
        if not (
            "tile_width" in study_description
            and isinstance(study_description["tile_width"], (int, np.integer))
            and study_description["tile_width"] > 0
        ):
            raise ValueError(
                'study_description["tile_width"]'
                " must exist and be a positive integer"
            )
        for slide in study_description["slides"].values():
            # Update keys of the dictionary from deprecated names
            self._update_dict(slide)

            if not (
                "returned_magnification" in slide
                and isinstance(
                    slide["returned_magnification"],
                    (int, np.integer, float, np.floating),
                )
                and slide["returned_magnification"] > 0
            ):
                raise ValueError(
                    'slide["returned_magnification"]'
                    " must exist and be a positive number"
                )
        # Check that other necessary keys are also present!!!

        # Partition the set of tiles into chunks.
        self._designate_chunks_for_tiles(study_description)
        # cProfile.runctx(
        #     "self._designate_chunks_for_tiles(study_description)",
        #     globals=globals(),
        #     locals=locals(),
        #     sort="cumulative",
        # )

    def _designate_chunks_for_tiles(self, study_description):
        # Update keys of the dictionary from deprecated names
        self._update_dict(study_description)

        tile_height = study_description["tile_height"]
        tile_width = study_description["tile_width"]

        for slide in study_description["slides"].values():
            # Update keys of the dictionary from deprecated names
            self._update_dict(slide)

            if not (
                "chunk_height" in slide
                and isinstance(slide["chunk_height"], (int, np.integer))
                and slide["chunk_height"] > 0
            ):
                raise ValueError(
                    'slide["chunk_height"]' " must exist and be a positive integer"
                )
            if not (
                "chunk_width" in slide
                and isinstance(slide["chunk_width"], (int, np.integer))
                and slide["chunk_width"] > 0
            ):
                raise ValueError(
                    'slide["chunk_width"]' " must exist and be a positive integer"
                )
            chunk_height = slide["chunk_height"]
            chunk_width = slide["chunk_width"]

            tiles_names = list(slide["tiles"].keys())
            tiles_data = np.array(
                [
                    [
                        slide["tiles"][tile]["tile_top"],
                        slide["tiles"][tile]["tile_left"],
                    ]
                    for tile in tiles_names
                ],
                dtype=np.int64,
            )
            self.build_tree(tiles_data)
            chunks = slide["chunks"] = {}
            num_chunks = 0
            while self.get_tree() is not None:
                tile = self.get_topmost()
                chunk = chunks[f"chunk_{num_chunks}"] = {
                    "chunk_top": tiles_data[0],
                    "chunk_left": tiles_data[1],
                    "chunk_bottom": tiles_data[0] + chunk_height,
                    "chunk_right": tiles_data[1] + chunk_width,
                }
                num_chunks += 1

                mins = tile.copy()
                maxs = tile.copy()
                maxs[0] += chunk_height - tile_height + 1
                maxs[1] += chunk_width - tile_width + 1
                indices = self.find_in_range_and_delete(mins, maxs)
                tiles = chunk["tiles"] = {
                    tiles_names[i]: {
                        "tile_top": tiles_data[i][0],
                        "tile_left": tiles_data[i][1],
                    }
                    for i in indices
                }
                # Make the chunk as small as possible given the tiles that it must
                # support.  Note that this also ensures that the pixels that are read do
                # not run over the bottom or right border of the slide (assuming that
                # the tiles do not go over those borders).
                chunk["chunk_top"] = min([tile["tile_top"] for tile in tiles.values()])
                chunk["chunk_left"] = min(
                    [tile["tile_left"] for tile in tiles.values()]
                )
                chunk["chunk_bottom"] = (
                    max([tile["tile_top"] for tile in tiles.values()]) + tile_height
                )
                chunk["chunk_right"] = (
                    max([tile["tile_left"] for tile in tiles.values()]) + tile_width
                )

    @staticmethod
    def read_large_image(
        filename,
        chunk_top,
        chunk_left,
        chunk_bottom,
        chunk_right,
        returned_magnification,
    ):
        # if "_num_chunks" not in ChunkLocations.read_large_image.__dict__:
        #     ChunkLocations.read_large_image._num_chunks = 0
        # chunk_name = (
        #     f"#read_large_image {ChunkLocations.read_large_image._num_chunks:06}"
        # )
        # ChunkLocations.read_large_image._num_chunks += 1

        # print(f"{chunk_name} begin {datetime.datetime.now()}")
        ts = (
            large_image_source_tiff.open(filename)
            if os.path.splitext(filename)[1] in (".tif", ".tiff", ".svs")
            else large_image.open(filename)
        )
        chunk = ts.getRegion(
            scale=dict(magnification=returned_magnification),
            format=large_image.constants.TILE_FORMAT_NUMPY,
            region=dict(
                left=chunk_left,
                top=chunk_top,
                width=chunk_right - chunk_left,
                height=chunk_bottom - chunk_top,
                units="mag_pixels",
            ),
        )[0]
        # print(f"{chunk_name} end {datetime.datetime.now()}")
        return chunk

    @staticmethod
    def scale_it(value, factor):
        return math.floor(value / factor + 0.01)

    def build_tree(self, data):
        self.data = data
        self.tree = self._build(np.arange(self.data.shape[0]))

    def get_data(self):
        return self.data

    def get_tree(self):
        return self.tree

    def get_topmost(self):
        return self.tree["topmost"]

    def find_in_range_and_delete(self, mins, maxs):
        self.mins = mins
        self.maxs = maxs
        indices, newtree = self._find_in_range_and_delete(subtree=self.tree)
        self.tree = newtree
        return indices

    def _build(self, indices):
        # Split this subset of the data based upon its coordinate means
        subset = self.data[indices, :]
        means = np.mean(subset, axis=0)
        # Calculate the quadrant (in range(2**m)) for each point
        rants = (subset[:, 0] >= means[0]) + 0
        for col in range(1, self.data.shape[1]):
            rants = (rants * 2) + (subset[:, col] >= means[col])

        # How to process this depends upon how many quadrants are used
        occur = np.unique(rants)
        if len(occur) == 1:
            return {"means": means, "topmost": means, "indices": indices}
        else:
            recurse = {rant: self._build(indices[rants == rant]) for rant in occur}
            qvalues = list(recurse.values())
            # Find the the topmost, in dictionary order
            topmost = self._compute_topmost(qvalues)

            # Return what we have found
            return {"means": means, "topmost": topmost, "quadrants": recurse}

    @staticmethod
    def _compute_topmost(qvalues):
        topmost = qvalues[0]["topmost"]
        for k in range(1, len(qvalues)):
            test_key = qvalues[k]["topmost"]
            for c in range(len(topmost)):
                if test_key[c] != topmost[c]:
                    break
            if test_key[c] < topmost[c]:
                topmost = test_key
        return topmost

    def _find_in_range_and_delete(self, subtree):
        if "indices" in subtree:
            # Process this leaf node
            if all(subtree["means"] >= self.mins) and all(subtree["means"] < self.maxs):
                # Return these indices and remove the subtree
                return subtree["indices"], None
            else:
                # Return no indices and remove nothing from the subtree
                return self.no_indices, subtree
        else:
            # Process this internal node
            means = subtree["means"]
            recurse = dict(
                (qkey, self._find_in_range_and_delete(qvalue))
                if all(
                    (
                        self.maxs[col] > means[col]
                        if qkey & 2 ** (self.data.shape[1] - 1 - col)
                        else self.mins[col] < means[col]
                        for col in range(self.data.shape[1])
                    )
                )
                else (qkey, (self.no_indices, qvalue))
                for qkey, qvalue in subtree["quadrants"].items()
            )
            indices = np.array(
                [index for pair in recurse.values() for index in pair[0]],
                dtype=np.int64,
            )
            quadrants = {
                qkey: pair[1] for qkey, pair in recurse.items() if pair[1] is not None
            }
            if len(quadrants) == 0:
                return indices, None
            topmost = self._compute_topmost(list(quadrants.values()))
            return indices, {
                "means": subtree["means"],
                "topmost": topmost,
                "quadrants": quadrants,
            }
