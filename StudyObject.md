# The histomcis_stream `study` Python dict

The study is defined as a hierarchy of nested Python dict (dictionary) objects.  Further below, arguments to the `histomics_stream` function objects are described.

## Keys and values
The keys in the Python dict for the study and their corresponding values are as follows.  The format in the following list is
+ **key** (type of value): description of value

Those keys that are fixed strings are in bold.  The keys whose values are set directly by the user are in italics; all other values are set by calls to `histomics_stream` function objects.

+ ***version*** (str):
  Always equal to "version-1".
+ ***number_pixel_rows_for_tile*** (int):
  How high is each tile, measured in pixels using the `target_magnification` (described below).
+ ***number_pixel_columns_for_tile*** (int):
  How wide is each tile, measured in pixels using the `target_magnification` (described below).
+ ***slides*** (Python dict):
  Contains information about the study's slides.  The distinct keys of this Python dict are set by the user for their own convenience, one per slide.
  + *user-selected key for slide* (Python dict):
    Contains information about this slide.  The keys and values for this Python dict are:
    + ***filename*** (str):
      The path to the file containing the pixel data for this slide.
    + ***slide_name*** (str):
      A user-supplied name for this slide.
    + ***slide_group*** (str):
      A user-supplied name for the group to which this slide belongs.
    + ***number_pixel_rows_for_chunk*** (int):
      For read efficiency, how high a chunk of data that is read in one read should be, measured in pixels using the `target_magnification` (described below).
    + ***number_pixel_columns_for_chunk*** (int):
      For read efficiency, how wide a chunk of data that is read in one read should be, measured in pixels using the `target_magnification` (described below).
    + **`target_magnification`** (float):
      The image magnification that the user wishes to use for the slide, if available given other restrictions.  A value of 10 corresponds to a pixel resolution of approximately 1 micron; magnification 40 is approximately 0.25 microns per pixel.
    + **scan_magnification** (float):
      The highest magnification directly available from the file storing the image.
    + **read_magnification** (float):
      The magnification directly read from the file storing the image.  This will be the smallest magnification directly available that is at least as large as the `target_magnification` if `magnification_source in ("exact", "native")`; it will be the `scan_magnification` if `magnification_source="scan"` is selected.
    + **returned_magnification** (float):
      The magnification of the pixel data returned by `histomics_stream`.  This will be the `target_magnification` if `magnification_source="exact"` is selected; it will be `read_magnification` if `magnification_source="native"` is selected; it will be `scan_magnification` if `magnification_source="scan"` is selected.
    + **level** (float):
      the internal `large_image` level that defines the `returned_magnification`.
    + **number_pixel_rows_for_slide** (int):
      How high is the slide, measured in pixels using the `target_magnification` (described above).
    + **number_pixel_columns_for_slide** (int):
      How wide is the slide, measured in pixels using the `target_magnification` (described above).
    + **number_tile_rows_for_slide** (int):
      How many (possibly overlapping) tiles fit into the height of the slide.
    + **number_tile_columns_for_slide** (int):
      How many (possibly overlapping) tiles fit into the width of the slide.
    + **number_pixel_rows_for_mask** (int):
      If a mask is supplied this is the mask's height in its scan resolution.
    + **number_pixel_columns_for_mask** (int):
      If a mask is supplied this is the mask's width in its scan resolution.
    + **tiles** (Python dict):
      Contains information about the slide's tiles.  The keys of this Python dict are set by the user for their own convenience, one per tile.
      + *user-selected key for tile* (Python dict):
        Contains information about this tile.  The keys and values for this Python dict are:
        + **tile_top** (int):
          The index of the top row of the tile, where 0 is the top row of the slide, measured in pixels using the `target_magnification` (described above).
        + **tile_left** (int):
          The index of the leftmost column of the tile, where 0 is the leftmost column of the slide, measured in pixels using the `target_magnification` (described above).
    + **chunks** (Python dict):
      Contains information about the slide's read chunks.  The keys of this Python dict are set by `histomics_stream` for its own convenience, one per chunk.
      + key for chunk (Python dict):
        Contains information about this chunk.  The keys and values for this Python dict are:
        + **chunk_top** (int):
          The index of the top row of the chunk, where 0 is the top row of the slide, measured in pixels using the `target_magnification` (described above).
        + **chunk_left** (int):
          The index of the leftmost column of the chunk, where 0 is the leftmost column of the slide, measured in pixels using the `target_magnification` (described above).
        + **chunk_bottom** (int):
          The index of the bottom row of the chunk, where 0 is the top row of the slide, measured in pixels using the `target_magnification` (described above).
        + **chunk_right** (int):
          The index of the rightmost column of the chunk, where 0 is the leftmost column of the slide, measured in pixels using the `target_magnification` (described above).
        + **tiles** (Python dict):
          The tiles that that will be read together when this chunk is read; `chunk["tiles"][tile_key]` is a reference to the corresponding `slide["tiles"][tile_key]` value.

## Arguments for `histomics_stream` function objects.

+ ***target_magnification*** (float):
  The image magnification that the user wishes to use for the slide, if available given other restrictions.  A value of 10 corresponds to a pixel resolution of approximately 1 micron; magnification 40 is approximately 0.25 microns per pixel.

+ ***magnification_source*** (str in ["scan", "native", "exact"]):
    "scan" will produce tiles from the highest magnification avaialable.  This is typically the slide scanner's objective magnification.

    "native" will produce tiles from the nearest available magnification equal to or greater than target_magnification (within a 2% tolerance).  The "native" option is useful when you want to handle resizing of tiles to target_magnification on your own.

    "exact" will produce tiles using "native" option and then resize these tiles to match target_magnification.  Resizing is handled by PIL using the Lanczos antialiasing filter since the resizing shrinks the tile by definition.

    For either "scan" or "native", the size of the read and returned tiles will be (tile_height * returned_magnification / target_magnification, tile_width * returned_magnification / target_magnification).  For "exact" the size of the returned tiles will be (tile_height, tile_width).

    This procedure sets values in the Python dict for this slide to capture the scan, read, and returned magnification of the tiles.  This is helpful for example to resize results to the scan magnification for visualization in HistomicsUI, or to resize between native and target magnification when using "native".  "scan_magnification" is the highest magnification from the source file; "read_magnification" is the magnification read from the source file; "returned_magnification" is the magnification of the returned tiles which is same as "read_magnification" in the case of "scan" or "native" or is the same as "target_magnification" in the case of "exact".

+ ***randomly_select*** (int):
    The number of tiles to be randomly selected from the list that would otherwise be written to the Python dict for this slide.  A value of `-1` is the default and means that all tiles should be written, except that the default is `+1` for `TilesRandomly`.

+ ***number_pixel_overlap_rows_for_tile*** (int):
    Specifies the desired amount of vertical overlap between adjacent tiles, measured in pixels using the `target_magnification` (described above).  This defaults to 0, which means that there is no overlap between adjacent tiles; they are abutting.

+ ***number_pixel_overlap_columns_for_tile*** (int):
    Specifies the desired amount of horizontal overlap between adjacent tiles, measured in pixels using the `target_magnification` (described above).  This defaults to 0, which means that there is no overlap between adjacent tiles; they are abutting.

+ ***mask_filename*** (str):
    The path of the image file to be read and used as a mask.  The aspect ratio of the mask (in terms of its pixel dimensions) is expected to be about the same as the aspect ratio of the main image ( in terms of its grid of tiles).  A non-zero value in the mask indicates that the tile should be retained.  The default is "", which means that there is no masking.

+ ***mask_threshold*** (float):
    A value in [0.0, 1.1].  A tile is retained if the fraction of the tile overlapping non-zero pixels in the mask is at least the mask_threshold.

+ ***tiles_dictionary*** (Python dict):
    For example, `{'AB234': {'tile_top': top0, 'tile_left': left0}, 'CD43': {'tile_top': top1, 'tile_left': left1}, ...}`.  Tiles from this list will copied into the Python dict for this slide if they are randomly selected.
