import copy
import itk
import math
import numpy as np
import random
import re


class FindResolutionForSlide:
    """A class that computes read parameters for slides.

    An instance of class FindResolutionForSlide is a callable that will add level,
    factor, number_pixel_rows_for_slide, and number_pixel_columns_for_slide fields to a
    slide dictionary.

    Parameters for the constructor
    ------------------------------
    filename : string
        The path of the image file to be read.
    desired_magnification : float
        The desired magnification to be read from the file when
        multiple resolutions are available.  For example, a value of
        10 corresponses to about 1 micron per pixel and a value of 20
        corresponds to about 0.5 microns per pixel.
    magnification_tolerance : float
        For example a value of 0.02 allows selection of an image that
        has a magnification of up to 2% different from the desired
        magnification.  If no such image exists, the image with
        greater magnification will be selected.

    """

    def __init__(self, study, desired_magnification, magnification_tolerance):
        """Sanity check the supplied parameters and store them for later use."""
        # Check values.
        if not ("version" in study and study["version"] == "version-1"):
            raise ValueError('study["version"] must exist and be equal to "version-1".')
        if not (
            isinstance(desired_magnification, (int, float))
            and 0 < desired_magnification
        ):
            raise ValueError(
                f"desired_magnification ({desired_magnification})"
                " must be a positive number"
            )
        if not (
            isinstance(magnification_tolerance, (int, float))
            and 0 <= magnification_tolerance <= 1
        ):
            raise ValueError(
                f"magnification_tolerance ({magnification_tolerance})"
                " must be a value in [0, 1]"
            )

        # Save values.
        self.desired_magnification = desired_magnification
        self.magnification_tolerance = magnification_tolerance

    def __call__(self, slide):
        """Add level, factor, number_pixel_rows_for_slide, and number_pixel_columns_for_slide
        fields to a slide dictionary.

        """

        # Check values.
        if "filename" not in slide:
            raise ValueError('slide["filename"] must be already set.')
        filename = slide["filename"]

        # Do the work.
        if re.compile(r"\.svs$").search(filename):
            import large_image

            # read whole-slide image file and create large_image object
            ts = large_image.open(filename)

            # measure objective of level 0
            objective = np.float32(ts.getNativeMagnification()["magnification"])

            if False:
                # Use the level that large_image is willing to interpolate for us.
                preferred_levels = [
                    ts.getLevelForMagnification(
                        self.desired_magnification, rounding=False
                    )
                ]
            else:
                # Use one of the levels that is stored in the image file.
                preferred_levels = list(
                    set(ts.getPreferredLevel(level) for level in range(ts.levels))
                )
                preferred_levels.sort(reverse=True)

            estimated = np.array(
                [
                    ts.getMagnificationForLevel(level)["magnification"]
                    for level in preferred_levels
                ]
            )

            # Find best native level to use and its factor
            level, factor = self._get_level_and_factor(
                self.desired_magnification, estimated, self.magnification_tolerance
            )

            number_pixel_columns_for_slide = int(
                ts.sizeX * estimated[level] // objective
            )
            number_pixel_rows_for_slide = int(ts.sizeY * estimated[level] // objective)

            # Rather than as the index into preferred_levels, change
            # level to be the value that large_image uses
            level = preferred_levels[level]

        elif re.compile(r"\.svs$").search(filename):
            import openslide as os

            # read whole-slide image file and create openslide object
            os_obj = os.OpenSlide(filename)

            # measure objective of level 0
            objective = np.float32(os_obj.properties[os.PROPERTY_NAME_OBJECTIVE_POWER])

            # calculate magnifications of levels
            estimated = np.array(objective / os_obj.level_downsamples)

            # Find best native level to use and its factor
            level, factor = self._get_level_and_factor(
                self.desired_magnification, estimated, self.magnification_tolerance
            )

            # get slide number_pixel_columns_for_slide,
            # number_pixel_rows_for_slide at desired
            # magnification. (Note number_pixel_columns_for_slide
            # before number_pixel_rows_for_slide)
            (
                number_pixel_columns_for_slide,
                number_pixel_rows_for_slide,
            ) = os_obj.level_dimensions[level]

        elif re.compile(r"\.zarr$").search(filename):
            import zarr

            # read whole-slide image and create zarr objects
            store = zarr.DirectoryStore(filename)
            source_group = zarr.open(store, mode="r")

            # measure objective of level 0
            objective = np.float32(source_group.attrs[os.PROPERTY_NAME_OBJECTIVE_POWER])

            # calculate magnifications of levels
            estimated = np.array(objective / source_group.attrs["level_downsamples"])

            # Find best native level to use and its factor
            level, factor = self._get_level_and_factor(
                self.desired_magnification, estimated, self.magnification_tolerance
            )

            # get slide number_pixel_columns_for_slide,
            # number_pixel_rows_for_slide at desired
            # magnification. (Note number_pixel_rows_for_slide before
            # number_pixel_columns_for_slide)
            number_pixel_rows_for_slide, number_pixel_columns_for_slide = source_group[
                format(level)
            ].shape[0:2]

        else:
            from PIL import Image

            # We don't know magnifications so assume reasonable values
            # for level and factor.
            level = 0
            factor = 1.0
            pil_obj = Image.open(filename)
            number_pixel_columns_for_slide, number_pixel_rows_for_slide = pil_obj.size

        slide["level"] = level
        slide["factor"] = factor
        slide["number_pixel_rows_for_slide"] = number_pixel_rows_for_slide
        slide["number_pixel_columns_for_slide"] = number_pixel_columns_for_slide

    def _get_level_and_factor(
        self, desired_magnification, estimated, magnification_tolerance
    ):
        """A private subroutine that computes level and factor."""
        # calculate difference with magnification levels
        delta = desired_magnification - estimated

        # match to existing levels
        if (
            np.min(np.abs(np.divide(delta, desired_magnification)))
            < magnification_tolerance
        ):  # match
            level = np.squeeze(np.argmin(np.abs(delta)))
            factor = 1.0
        elif np.any(delta < 0):
            value = np.max(delta[delta < 0])
            level = np.squeeze(np.argwhere(delta == value)[0])
            factor = desired_magnification / estimated[level]
        else:  # desired magnification above base level - throw error
            raise ValueError("Cannot interpolate above scan magnification.")

        return level, factor


class TilesByGridAndMask:
    """Select tiles according to a regular grid.  Optionally, restrict
    the list by a mask that is read from a file.  Optionally, further
    select a random subset of them.

    An instance of class TilesByGridAndMask is a callable that will
    select the coordinates of tiles to be taken from a slide.  The
    selected tiles will be written to the slide dictionary.

    Parameters for the constructor
    ------------------------------
    study : dictionary
        The study dictionary from which to read parameters about the study.
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

    """

    def __init__(
        self,
        study,
        randomly_select=-1,  # Defaults to select all
        number_pixel_overlap_rows_for_tile=0,  # Defaults to no overlap between adjacent tiles
        number_pixel_overlap_columns_for_tile=0,
        mask_filename="",  # Defaults to no masking
    ):
        """Sanity check the supplied parameters and store them for later use."""
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
                    f"The mask ({mask_filename})" " should be a 2-dimensional image."
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
        if mask_filename != "":
            self.mask_itk = mask_itk

    def __call__(self, slide):
        """Select tiles according to a regular grid.  Optionally, restrict the list by a mask.
        Optionally, select a random subset of them.
        """
        # Check values.
        if "number_pixel_rows_for_slide" not in slide:
            raise ValueError(
                'slide["number_pixel_rows_for_slide"] must be already set.'
            )
        if "number_pixel_columns_for_slide" not in slide:
            raise ValueError(
                'slide["number_pixel_columns_for_slide"] must be already set.'
            )

        # Do the work.
        row_stride = (
            self.number_pixel_rows_for_tile - self.number_pixel_overlap_rows_for_tile
        )
        number_tile_rows_for_slide = slide["number_tile_rows_for_slide"] = math.floor(
            (
                slide["number_pixel_rows_for_slide"]
                - self.number_pixel_overlap_rows_for_tile
            )
            / row_stride
        )
        column_stride = (
            self.number_pixel_columns_for_tile
            - self.number_pixel_overlap_columns_for_tile
        )
        number_tile_columns_for_slide = slide[
            "number_tile_columns_for_slide"
        ] = math.floor(
            (
                slide["number_pixel_columns_for_slide"]
                - self.number_pixel_overlap_columns_for_tile
            )
            / column_stride
        )
        has_mask = hasattr(self, "mask_itk")
        if has_mask:
            # We will change the resolution of the mask (if necessary), which will
            # change the number of pixels, but will not change the overall physical size
            # represented by the image nor the position of the upper left corner of its
            # upper left pixel.
            input_size = itk.size(self.mask_itk)
            output_size = [number_tile_columns_for_slide, number_tile_rows_for_slide]
            if input_size != output_size:
                # print(f"Resampling from input_size = {input_size} to output_size = {output_size}")
                # Check that the input and output aspect ratios are pretty close
                if (
                    abs(
                        math.log(
                            (output_size[0] / input_size[0])
                            / (output_size[1] / input_size[1])
                        )
                    )
                    > 0.20
                ):
                    raise ValueError(
                        "The mask aspect ratio does not match that for the number of tiles."
                    )
                input_spacing = itk.spacing(self.mask_itk)
                input_origin = itk.origin(self.mask_itk)
                image_dimension = self.mask_itk.GetImageDimension()
                output_spacing = [
                    input_spacing[d] * input_size[d] / output_size[d]
                    for d in range(image_dimension)
                ]
                output_origin = [
                    input_origin[d] + 0.5 * (output_spacing[d] - input_spacing[d])
                    for d in range(image_dimension)
                ]
                interpolator = itk.NearestNeighborInterpolateImageFunction.New(
                    self.mask_itk
                )
                resampled_mask_itk = itk.resample_image_filter(
                    self.mask_itk,
                    interpolator=interpolator,
                    size=output_size,
                    output_spacing=output_spacing,
                    output_origin=output_origin,
                )
            else:
                resampled_mask_itk = self.mask_itk

        tiles = slide["tiles"] = {}
        number_of_tiles = 0
        for row in range(number_tile_rows_for_slide):
            for column in range(number_tile_columns_for_slide):
                if not (has_mask and resampled_mask_itk[row, column] == 0):
                    tiles[f"tile_{number_of_tiles}"] = {
                        "tile_top": row * row_stride,
                        "tile_left": column * column_stride,
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


class TilesByList:
    """Select the tiles supplied by the user.  Optionally, select a
    random subset of them.

    An instance of class TilesByList is a callable that will select
    the coordinates of tiles to be taken from a slide.  The selected
    tiles will be written to the slide dictionary.

    Parameters for the constructor
    ------------------------------
    study : dictionary
        The study dictionary from which to read parameters about the study.
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
        """Sanity check the supplied parameters and store them for later use."""
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
        """Select the tiles supplied by the user.  Optionally, select a random subset of them."""
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
    """Select a random subset of all possible tiles.

    An instance of class TilesRandomly is a callable that will select
    the coordinates of tiles to be taken from a slide.  The selected
    tiles will be written to the slide dictionary.

    Parameters for the constructor
    ------------------------------
    study : dictionary
        The study dictionary from which to read parameters about the study.
    randomly_select: int
        The number of tiles to be randomly selected from the slide.
        The value must be positive.  A value of 1 is the default.

    """

    def __init__(
        self,
        study,
        randomly_select=1,  # Defaults to select one
    ):
        """Sanity check the supplied parameters and store them for later use."""
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
        """Select a random subset of all possible tiles."""
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
