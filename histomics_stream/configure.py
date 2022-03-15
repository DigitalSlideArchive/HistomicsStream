import copy
import itk
import math
import numpy as np
import random
import re


class FindResolutionForSlide:
    """
    A class that computes read parameters for slides.

    An instance of class FindResolutionForSlide is a callable that
    will add level, target_magnification, scan_magnification,
    read_magnification, returned_magnification,
    number_pixel_rows_for_slide, and number_pixel_columns_for_slide
    fields to a slide dictionary.

    Parameters for the constructor
    ------------------------------

    filename : string
        The path of the image file to be read.

    target_magnification : float
        The desired objective magnification for generated tiles.  For
        example, a value of 10 corresponds to about 1 micron per pixel
        and a value of 20 corresponds to about 0.5 microns per pixel.

    magnification_source : str in ["scan", "native", "exact"]
        "scan" will produce tiles from the highest magnification
        avaialable. This is typically the slide scanner's objective
        magnification.

        "native" will produce tiles from the nearest available
        magnification equal to or greater than target_magnification
        (within a 2% tolerance). The "native" option is useful when
        you want to handle resizing of tiles to target_magnification
        on your own.

        "exact" will produce tiles using "native" option and then
        resize these tiles to match target_magnification. Resizing is
        handled by PIL using the Lanczos antialiasing filter since the
        resizing shrinks the tile by definition.

        For either "scan" or "native", the size of the read and
        returned tiles will be (tile_height * returned_magnification /
        target_magnification, tile_width * returned_magnification /
        target_magnification).  For "exact" the size of the returned
        tiles will be (tile_height, tile_width).

        This procedure sets values in the slide dictionary to capture
        the scan, read, and returned magnification of the tiles. This
        is helpful for example to resize results to the scan
        magnification for visualization in HistomicsUI, or to resize
        between native and target magnification when using
        "native". "scan_magnification" is the highest magnification
        from the source file; "read_magnification" is the
        magnification read from the source file;
        "returned_magnification" is the magnification of the returned
        tiles which is same as "read_magnification" in the case of
        "scan" or "native" or "target_magnification" in the case of
        "exact".
    """

    def __init__(self, study, target_magnification, magnification_source):
        """
        Sanity check the supplied parameters and store them for later
        use.
        """
        # Check values.
        if not ("version" in study and study["version"] == "version-1"):
            raise ValueError('study["version"] must exist and be equal to "version-1".')
        if not (
            isinstance(target_magnification, (int, float)) and 0 < target_magnification
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
        Add level, target_magnification, scan_magnification,
        read_magnification, returned_magnification,
        number_pixel_rows_for_slide, and
        number_pixel_columns_for_slide fields to a slide dictionary.
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
            number_pixel_rows_for_slide = ts.sizeY
            number_pixel_columns_for_slide = ts.sizeX
            if scan_magnification != read_magnification:
                number_pixel_rows_for_slide = math.floor(
                    number_pixel_rows_for_slide
                    * read_magnification
                    / scan_magnification
                )
                number_pixel_columns_for_slide = math.floor(
                    number_pixel_columns_for_slide
                    * read_magnification
                    / scan_magnification
                )
            if read_magnification != self.target_magnification:
                number_pixel_rows_for_slide = math.floor(
                    number_pixel_rows_for_slide
                    * self.target_magnification
                    / read_magnification
                )
                number_pixel_columns_for_slide = math.floor(
                    number_pixel_columns_for_slide
                    * self.target_magnification
                    / read_magnification
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

            # get slide number_pixel_columns_for_slide, number_pixel_rows_for_slide at
            # desired magnification. (Note that number_pixel_rows_for_slide is before
            # number_pixel_columns_for_slide)
            number_pixel_rows_for_slide, number_pixel_columns_for_slide = source_group[
                format(level)
            ].shape[0:2]

            if (
                self.magnification_source == "exact"
                and self.target_magnification != returned_magnification
            ):
                raise ValueError(
                    f"Couldn't find magnification {self.target_magnification}X in Zarr storage."
                )

        slide["level"] = level
        # Note that slide size is defined by the requested magnification, which may not
        # be the same as the magnification for the selected level.  To get the slide
        # size for the magnification that we are using, these values must later be
        # multiplied by returned_magnification / target_magnification.
        slide["number_pixel_rows_for_slide"] = number_pixel_rows_for_slide
        slide["number_pixel_columns_for_slide"] = number_pixel_columns_for_slide

    def _get_level_and_magnifications(
        self, target_magnification, estimated_magnifications
    ):
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


class TilesByGridAndMask:
    """
    Select tiles according to a regular grid.  Optionally, restrict
    the list by a mask that is read from a file.  Optionally, further
    select a random subset of them.

    An instance of class TilesByGridAndMask is a callable that will
    select the coordinates of tiles to be taken from a slide.  The
    selected tiles will be written to the slide dictionary.

    Parameters for the constructor
    ------------------------------
    study : dictionary
        The study dictionary from which to read parameters about the
        study.
    randomly_select: int
        The number of tiles to be randomly selected from the list that
        would otherwise be written to the slide dictionary.  A value
        of -1 is the default and means that all tiles should be
        written.
    number_pixel_overlap_rows_for_tile
        Specifies the desired amount of vertical overlap (measured in
        rows of pixels) between adjacent tiles.  This defaults to 0,
        which means that there is no overlap between adjacent tiles;
        they are abutting.
    number_pixel_overlap_columns_for_tile
        Specifies the desired amount of horizontal overlap (measured
        in columns of pixels) between adjacent tiles.  This defaults
        to 0, which means that there is no overlap between adjacent
        tiles; they are abutting.
    mask_filename: string
        The path of the image file to be read and used as a mask.  The
        aspect ratio of the mask (in terms of its pixel dimensions) is
        expected to be about the same as the aspect ratio of the main
        image ( in terms of its grid of tiles).  A non-zero value in
        the mask indicates that the tile should be retained.  The
        default is "", which means that there is no masking.
    mask_threshold : float
        A value in [0.0, 1.1].  A tile is retained if the fraction of
        the tile overlapping non-zero pixels in the mask is at least
        the mask_threshold.

    """

    def __init__(
        self,
        study,
        randomly_select=-1,  # Defaults to select all
        number_pixel_overlap_rows_for_tile=0,  # Defaults to no overlap
        number_pixel_overlap_columns_for_tile=0,
        mask_filename="",  # Defaults to no masking
        mask_threshold=0.0,  # Defaults to any overlap with the mask
    ):
        """
        Sanity check the supplied parameters and store them for later
        use.
        """
        # Check values.
        if not ("version" in study and study["version"] == "version-1"):
            raise ValueError('study["version"] must exist and be equal to "version-1".')
        if not (
            "number_pixel_rows_for_tile" in study
            and isinstance(study["number_pixel_rows_for_tile"], int)
            and study["number_pixel_rows_for_tile"] > 0
        ):
            raise ValueError(
                'study["number_pixel_rows_for_tile"]'
                " must exist and be a positive integer"
            )
        if not (
            "number_pixel_columns_for_tile" in study
            and isinstance(study["number_pixel_columns_for_tile"], int)
            and study["number_pixel_columns_for_tile"] > 0
        ):
            raise ValueError(
                'study["number_pixel_columns_for_tile"]'
                " must exist and be a positive integer"
            )
        if not (isinstance(randomly_select, int) and -1 <= randomly_select):
            raise ValueError(
                f"randomly_select ({randomly_select})"
                " must be a non-negative integer or -1."
            )
        if not (
            isinstance(number_pixel_overlap_rows_for_tile, int)
            and number_pixel_overlap_rows_for_tile < study["number_pixel_rows_for_tile"]
        ):
            raise ValueError(
                f"number_pixel_overlap_rows_for_tile ({number_pixel_overlap_rows_for_tile})"
                " must be less than"
                f' number_pixel_rows_for_tile ({study["number_pixel_rows_for_tile"]}).'
            )
        if not (
            isinstance(number_pixel_overlap_columns_for_tile, int)
            and number_pixel_overlap_columns_for_tile
            < study["number_pixel_columns_for_tile"]
        ):
            raise ValueError(
                f"number_pixel_overlap_columns_for_tile ({number_pixel_overlap_columns_for_tile})"
                " must be less than"
                f' number_pixel_columns_for_tile ({study["number_pixel_columns_for_tile"]}).'
            )
        if mask_filename != "":
            mask_itk = itk.imread(mask_filename)  # May throw exception
            if mask_itk.GetImageDimension() != 2:
                raise ValueError(
                    f"The mask ({mask_filename}) should be a 2-dimensional image."
                )
        if not (
            isinstance(mask_threshold, float)
            and mask_threshold >= 0.0
            and mask_threshold <= 1.0
        ):
            raise ValueError(
                f"mask_threshold ({mask_threshold}) must be between 0 and 1 inclusive."
            )

        # Save values.  To keep garbage collection efficient don't save all of `study`.
        self.number_pixel_rows_for_tile = study["number_pixel_rows_for_tile"]
        self.number_pixel_columns_for_tile = study["number_pixel_columns_for_tile"]
        self.randomly_select = randomly_select
        self.number_pixel_overlap_rows_for_tile = number_pixel_overlap_rows_for_tile
        self.number_pixel_overlap_columns_for_tile = (
            number_pixel_overlap_columns_for_tile
        )
        self.mask_filename = mask_filename
        if self.mask_filename != "":
            self.mask_itk = mask_itk
        self.mask_threshold = mask_threshold

    def __call__(self, slide):
        """
        Select tiles according to a regular grid.  Optionally,
        restrict the list by a mask.  Optionally, select a random
        subset of them.
        """
        # Check values.
        if "number_pixel_rows_for_slide" not in slide:
            raise ValueError(
                'slide["number_pixel_rows_for_slide"] must be already set.'
            )
        self.number_pixel_rows_for_slide = slide["number_pixel_rows_for_slide"]
        if "number_pixel_columns_for_slide" not in slide:
            raise ValueError(
                'slide["number_pixel_columns_for_slide"] must be already set.'
            )

        self.number_pixel_columns_for_slide = slide["number_pixel_columns_for_slide"]
        #
        # Do the work.
        #
        row_stride = (
            self.number_pixel_rows_for_tile - self.number_pixel_overlap_rows_for_tile
        )
        column_stride = (
            self.number_pixel_columns_for_tile
            - self.number_pixel_overlap_columns_for_tile
        )

        # Return information to the user
        slide["number_tile_rows_for_slide"] = math.floor(
            (self.number_pixel_rows_for_slide - self.number_pixel_overlap_rows_for_tile)
            / row_stride
        )
        slide["number_tile_columns_for_slide"] = math.floor(
            (
                self.number_pixel_columns_for_slide
                - self.number_pixel_overlap_columns_for_tile
            )
            / column_stride
        )

        # Pre-process the mask
        has_mask = hasattr(self, "mask_itk")
        if has_mask:
            (
                self.number_pixel_rows_for_mask,
                self.number_pixel_columns_for_mask,
            ) = self.mask_itk.shape
            slide["number_pixel_rows_for_mask"] = self.number_pixel_rows_for_mask
            slide["number_pixel_columns_for_mask"] = self.number_pixel_columns_for_mask

            # Check that the input and output aspect ratios are pretty close
            if (
                abs(
                    math.log(
                        (
                            self.number_pixel_columns_for_slide
                            / self.number_pixel_columns_for_mask
                        )
                        / (
                            self.number_pixel_rows_for_slide
                            / self.number_pixel_rows_for_mask
                        )
                    )
                )
                > 0.20
            ):
                raise ValueError(
                    "The mask aspect ratio does not match that for the whole slide image."
                )

            # cumulative_mask[row, column] will be the number of mask_itk[r, c] (i.e.,
            # mask_itk.GetPixel((c,r))) values that are nonzero among all those with r <
            # row and c < column; note the strict inequalities.  We have added a
            # boundary on all sides of this array -- zeros on the top and left, and a
            # duplicate row (column) on the bottom (right) -- so that we do not need to
            # do extra testing in our code at the borders.  We use int64 in case there
            # are 2^31 (~2 billion = ~ 46k by 46k) or more non-zero pixel values in our
            # mask.
            self.cumulative_mask = np.zeros(
                (
                    self.number_pixel_rows_for_mask + 2,
                    self.number_pixel_columns_for_mask + 2,
                ),
                dtype=np.int64,
            )
            nonzero = np.vectorize(lambda x: int(x != 0))
            self.cumulative_mask[
                1 : self.number_pixel_rows_for_mask + 1,
                1 : self.number_pixel_columns_for_mask + 1,
            ] = nonzero(itk.GetArrayViewFromImage(self.mask_itk))
            self.cumulative_mask = np.cumsum(
                np.cumsum(self.cumulative_mask, axis=0), axis=1
            )

        # Look at each tile in turn
        tiles = slide["tiles"] = {}
        number_of_tiles = 0
        top_too_high = (
            self.number_pixel_rows_for_slide - self.number_pixel_rows_for_tile + 1
        )
        left_too_high = (
            self.number_pixel_columns_for_slide - self.number_pixel_columns_for_tile + 1
        )
        for top in range(0, top_too_high, row_stride):
            for left in range(0, left_too_high, column_stride):
                if not (has_mask and self.mask_rejects(top, left)):
                    tiles[f"tile_{number_of_tiles}"] = {
                        "tile_top": top,
                        "tile_left": left,
                    }
                number_of_tiles += 1  # Increment even if tile is skipped.

        # Choose a subset of the tiles randomly
        all_tile_names = tiles.keys()
        if 0 <= self.randomly_select < len(all_tile_names):
            keys_to_remove = random.sample(
                all_tile_names, len(all_tile_names) - self.randomly_select
            )
            for key in keys_to_remove:
                del tiles[key]

    def interpolate_cumulative(self, row, column):
        top = int(math.floor(row))
        left = int(math.floor(column))
        vertical_range = row - top
        horizontal_range = column - left
        response = (
            self.cumulative_mask[top, left]
            * (1.0 - vertical_range)
            * (1.0 - horizontal_range)
            + self.cumulative_mask[top + 1, left]
            * vertical_range
            * (1.0 - horizontal_range)
            + self.cumulative_mask[top, left + 1]
            * (1.0 - vertical_range)
            * horizontal_range
            + self.cumulative_mask[top + 1, left + 1]
            * vertical_range
            * horizontal_range
        )
        return response

    def mask_rejects(self, top, left):
        bottom = top + self.number_pixel_rows_for_tile
        right = left + self.number_pixel_columns_for_tile
        mask_top = (
            top * self.number_pixel_rows_for_mask / self.number_pixel_rows_for_slide
        )
        mask_bottom = (
            bottom * self.number_pixel_rows_for_mask / self.number_pixel_rows_for_slide
        )
        mask_left = (
            left
            * self.number_pixel_columns_for_mask
            / self.number_pixel_columns_for_slide
        )
        mask_right = (
            right
            * self.number_pixel_columns_for_mask
            / self.number_pixel_columns_for_slide
        )
        cumulative_top_left = self.interpolate_cumulative(mask_top, mask_left)
        cumulative_top_right = self.interpolate_cumulative(mask_top, mask_right)
        cumulative_bottom_left = self.interpolate_cumulative(mask_bottom, mask_left)
        cumulative_bottom_right = self.interpolate_cumulative(mask_bottom, mask_right)
        cumulative = (
            cumulative_bottom_right
            - cumulative_bottom_left
            - cumulative_top_right
            + cumulative_top_left
        )
        if self.mask_threshold > 0:
            score = cumulative / (
                self.mask_threshold
                * (mask_bottom - mask_top)
                * (mask_right - mask_left)
            )
            return score < 0.999999
        else:
            return cumulative < 0.000001


class TilesByList:
    """
    Select the tiles supplied by the user.  Optionally, select a
    random subset of them.

    An instance of class TilesByList is a callable that will select
    the coordinates of tiles to be taken from a slide.  The selected
    tiles will be written to the slide dictionary.

    Parameters for the constructor
    ------------------------------
    study : dictionary
        The study dictionary from which to read parameters about the
        study.
    randomly_select: int
        The number of tiles to be randomly selected from the list that
        would otherwise be written to the slide dictionary.  A value
        of -1 is the default and means that all tiles should be
        written.
    tiles_dictionary: dictionary
        For example, {'AB234': {'tile_top': top0, 'tile_left': left0},
        'CD43': {'tile_top': top1, 'tile_left': left1}, ...}.  Tiles
        from this list will copied into the slide dictionary if they
        are randomly selected.

    """

    def __init__(
        self,
        study,
        randomly_select=-1,  # Defaults to select all
        tiles_dictionary={},  # {'AB234': {'tile_top': top0, 'tile_left': left0}, 'CD43': {'tile_top': top1, 'tile_left': left1}, ...}
    ):
        """
        Sanity check the supplied parameters and store them for later
        use.
        """
        # Check values
        if not ("version" in study and study["version"] == "version-1"):
            raise ValueError('study["version"] must exist and be equal to "version-1".')
        if not (
            "number_pixel_rows_for_tile" in study
            and isinstance(study["number_pixel_rows_for_tile"], int)
            and study["number_pixel_rows_for_tile"] > 0
        ):
            raise ValueError(
                'study["number_pixel_rows_for_tile"]'
                " must exist and be a positive integer"
            )
        if not (
            "number_pixel_columns_for_tile" in study
            and isinstance(study["number_pixel_columns_for_tile"], int)
            and study["number_pixel_columns_for_tile"] > 0
        ):
            raise ValueError(
                'study["number_pixel_columns_for_tile"]'
                " must exist and be a positive integer"
            )
        if not (isinstance(randomly_select, int) and -1 <= randomly_select):
            raise ValueError(
                f"randomly_select ({randomly_select})"
                " must be a non-negative integer or -1."
            )
        if not (
            isinstance(tiles_dictionary, dict)
            and all(
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
                    isinstance(tile_corner[key], int)
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
        self.number_pixel_rows_for_tile = study["number_pixel_rows_for_tile"]
        self.number_pixel_columns_for_tile = study["number_pixel_columns_for_tile"]
        self.randomly_select = randomly_select
        self.tiles_dictionary = copy.deepcopy(
            tiles_dictionary
        )  # in case user changes it later

    def __call__(self, slide):
        """
        Select the tiles supplied by the user.  Optionally, select a
        random subset of them.
        """
        tiles = slide["tiles"] = copy.deepcopy(
            self.tiles_dictionary
        )  # in case __call__ is called again.
        all_tile_names = tiles.keys()
        if 0 <= self.randomly_select < len(all_tile_names):
            keys_to_remove = random.sample(
                all_tile_names, len(all_tile_names) - self.randomly_select
            )
            for key in keys_to_remove:
                del tiles[key]


class TilesRandomly:
    """
    Select a random subset of all possible tiles.

    An instance of class TilesRandomly is a callable that will select
    the coordinates of tiles to be taken from a slide.  The selected
    tiles will be written to the slide dictionary.

    Parameters for the constructor
    ------------------------------
    study : dictionary
        The study dictionary from which to read parameters about the
        study.
    randomly_select: int
        The number of tiles to be randomly selected from the slide.
        The value must be positive.  A value of 1 is the default.

    """

    def __init__(self, study, randomly_select=1):  # Defaults to select one
        """
        Sanity check the supplied parameters and store them for later
        use.
        """
        # Check values.
        if not ("version" in study and study["version"] == "version-1"):
            raise ValueError('study["version"] must exist and be equal to "version-1".')
        if not (
            "number_pixel_rows_for_tile" in study
            and isinstance(study["number_pixel_rows_for_tile"], int)
            and study["number_pixel_rows_for_tile"] > 0
        ):
            raise ValueError(
                'study["number_pixel_rows_for_tile"]'
                " must exist and be a positive integer"
            )
        if not (
            "number_pixel_columns_for_tile" in study
            and isinstance(study["number_pixel_columns_for_tile"], int)
            and study["number_pixel_columns_for_tile"] > 0
        ):
            raise ValueError(
                'study["number_pixel_columns_for_tile"]'
                " must exist and be a positive integer"
            )
        if not (isinstance(randomly_select, int) and 0 <= randomly_select):
            raise ValueError(
                f"randomly_select ({randomly_select})"
                " must be a non-negative integer."
            )

        # Save values.  To keep garbage collection efficient don't save all of `study`.
        self.number_pixel_rows_for_tile = study["number_pixel_rows_for_tile"]
        self.number_pixel_columns_for_tile = study["number_pixel_columns_for_tile"]
        self.randomly_select = randomly_select

    def __call__(self, slide):
        """
        Select a random subset of all possible tiles.
        """
        if "number_pixel_rows_for_slide" not in slide:
            raise ValueError(
                'slide["number_pixel_rows_for_slide"] must be already set.'
            )
        if "number_pixel_columns_for_slide" not in slide:
            raise ValueError(
                'slide["number_pixel_columns_for_slide"] must be already set.'
            )

        row_too_big = (
            slide["number_pixel_rows_for_slide"] - self.number_pixel_rows_for_tile + 1
        )
        column_too_big = (
            slide["number_pixel_columns_for_slide"]
            - self.number_pixel_columns_for_tile
            + 1
        )
        row_column_list = [
            (random.randrange(0, row_too_big), random.randrange(0, column_too_big))
            for _ in range(self.randomly_select)
        ]
        tiles = slide["tiles"] = {}
        number_of_tiles = 0
        for (row, column) in row_column_list:
            tiles[f"tile_{number_of_tiles}"] = {"tile_top": row, "tile_left": column}
            number_of_tiles += 1
