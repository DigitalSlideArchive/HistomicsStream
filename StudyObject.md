# The histomcis_stream `study` Python dict

The study is defined as a hierarchy of nested Python dict objects.  The keys in the `study` dict and their corresponding values are as follows.  Those keys whose value are set directly by the user are in italics; all other values are set by calls to `histomics_stream` functions.

+ ***version***: Always equal to "version-1".
+ ***number_pixel_rows_for_tile***: how high is each tile, measured in pixels using the `target_magnification` (described below).
+ ***number_pixel_columns_for_tile***: how wide is each tile, measured in pixels using the `target_magnification` (described below).
+ ***slides***: a Python dict object containing information about the study's slides.  The distinct keys of this dictionary are set by the user for their own convenience, one per slide.
  + *user-selected key for slide*: a dictionary containing information about this slide.  The keys and values for this dictionary are:
    + ***filename***: the path to the file containing the pixel data for this slide.
    + ***slide_name***: a user-supplied name for this slide.
    + ***slide_group***: a user-supplied name for the group to which this slide belongs.
    + ***number_pixel_rows_for_chunk***: for read efficiency, how high a chunk of data that is read in one read should be, measured in pixels using the `target_magnification` (described below).
    + ***number_pixel_columns_for_chunk***: for read efficiency, how wide a chunk of data that is read in one read should be, measured in pixels using the `target_magnification` (described below).
    + **target_magnification**: The image magnification that the user wishes to use for the slide, if available given other restrictions.  A value of 10 corresponds to a pixel resolution of approximately 1 micron; magnification 40 is approximately 0.25 microns per pixel.
    + **scan_magnification**: The highest magnification directly available from the file storing the image.
    + **read_magnification**: The magnification directly read from the file storing the image.  This will be the smallest magnification directly available that is at least as large as the `target_magnification` if `magnification_source in ("exact", "native")`; it will be the `scan_magnification` if `magnification_source="scan"` is selected.
    + **returned_magnification**: The magnification of the pixel data returned by `histomics_stream`. This will be the `target_magnification` if `magnification_source="exact"` is selected; it will be `read_magnification` if `magnification_source="native"` is selected; it will be `scan_magnification` if `magnification_source="scan"` is selected.
    + **level**: the internal `large_image` level that defines the `returned_magnification`.
    + **number_pixel_rows_for_slide**: how high is the slide, measured in pixels using the `target_magnification` (described above).
    + **number_pixel_columns_for_slide**: how wide is the slide, measured in pixels using the `target_magnification` (described above).
    + **number_tile_rows_for_slide**: how many (possibly overlapping) tiles fit into the height of the slide.
    + **number_tile_columns_for_slide**: how many (possibly overlapping) tiles fit into the width of the slide.
    + **number_pixel_rows_for_mask**: if a mask is supplied this is the mask's height in its scan resolution.
    + **number_pixel_columns_for_mask**: if a mask is supplied this is the mask's width in its scan resolution.
    + **tiles**: a Python dict object containing information about the slide's tiles.  The keys of this dictionary are set by the user for their own convenience, one per tile.
      + *user-selected key for tile*: a dictionary containing information about this tile.  The keys and values for this dictionary are:
        + **tile_top**: the index of the top row of the tile, where 0 is the top row of the slide, measured in pixels using the `target_magnification` (described above).
        + **tile_left**: the index of the leftmost column of the tile, where 0 is the leftmost column of the slide, measured in pixels using the `target_magnification` (described above).
    + **chunks**: a Python dict object containing information about the slide's read chunks.  The keys of this dictionary are set by `histomics_stream` for its own convenience, one per chunk.
        + **chunk_top**: the index of the top row of the chunk, where 0 is the top row of the slide, measured in pixels using the `target_magnification` (described above).
        + **chunk_left**: the index of the leftmost column of the chunk, where 0 is the leftmost column of the slide, measured in pixels using the `target_magnification` (described above).
        + **chunk_bottom**: the index of the bottom row of the chunk, where 0 is the top row of the slide, measured in pixels using the `target_magnification` (described above).
        + **chunk_right**: the index of the rightmost column of the chunk, where 0 is the leftmost column of the slide, measured in pixels using the `target_magnification` (described above).
        + **tiles**: the tiles that that will be read together when this chunk is read; `chunk["tiles"][tile_key]` is a reference to the corresponding `slide["tiles"][tile_key]` value.
