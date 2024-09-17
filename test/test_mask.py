#!/usr/bin/env python3

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


def test_mask_threshold():
    import histomics_stream as hs
    import os
    import pooch

    wsi_path = pooch.retrieve(
        fname="TCGA-AN-A0G0-01Z-00-DX1.svs",
        url=(
            "https://drive.usercontent.google.com/download"
            "?export=download"
            "&id=19agE_0cWY582szhOVxp9h3kozRfB4CvV"
            "&confirm=t"
        ),
        known_hash="d046f952759ff6987374786768fc588740eef1e54e4e295a684f3bd356c8528f",
        path=str(pooch.os_cache("pooch")) + os.sep + "wsi",
    )
    print(f"Have {wsi_path}")

    # download binary mask image
    mask_path = pooch.retrieve(
        fname="TCGA-AN-A0G0-01Z-00-DX1.mask.png",
        url=(
            "https://drive.usercontent.google.com/download"
            "?export=download"
            "&id=17GOOHbL8Bo3933rdIui82akr7stbRfta"
            "&confirm=t"
        ),
        known_hash="bb657ead9fd3b8284db6ecc1ca8a1efa57a0e9fd73d2ea63ce6053fbd3d65171",
        path=str(pooch.os_cache("pooch")) + os.sep + "wsi",
    )
    print(f"Have {mask_path}")

    my_study = dict(
        version="version-1",
        number_pixel_columns_for_tile=5471,
        number_pixel_rows_for_tile=5743,
        overlap_width=127,
        overlap_height=101,
        slides=dict(
            Slide_0=dict(
                filename=wsi_path,
                slide_name=os.path.splitext(os.path.split(wsi_path)[1])[0],
                slide_group="test_mask_threshold",
                chunk_width=31,
                chunk_height=37,
            )
        ),
    )
    find_slide_resolution = hs.configure.FindResolutionForSlide(
        my_study, target_magnification=20, magnification_source="native"
    )
    for slide in my_study["slides"].values():
        find_slide_resolution(slide)

    tiler_thresholds = (0.00, 0.20, 0.50, 0.80, 1.00)
    tilers = [
        hs.configure.TilesByGridAndMask(
            my_study,
            mask_filename=mask_path,
            mask_threshold=threshold,
            number_pixel_overlap_rows_for_tile=101,
            number_pixel_overlap_columns_for_tile=127,
        )
        for threshold in tiler_thresholds
    ]

    def run_tiler(study, tiler):
        for slide in study["slides"].values():
            tiler(slide)
        return [
            (
                value["filename"],
                [
                    (tile["tile_top"], tile["tile_left"])
                    for tile in value["tiles"].values()
                ],
            )
            for value in study["slides"].values()
        ]

    found_tiles = [run_tiler(my_study, tiler) for tiler in tilers]

    # print(f"    expected_tiles = {repr(found_tiles)}")
    expected_tiles = [
        [
            (
                wsi_path,
                [(0, 10688), (0, 16032), (0, 21376)]
                + [(5642, 5344), (5642, 10688), (5642, 16032), (5642, 21376)]
                + [(11284, 5344), (11284, 10688), (11284, 16032), (11284, 21376)],
            )
        ],
        [
            (
                wsi_path,
                [(0, 16032), (0, 21376)]
                + [(5642, 5344), (5642, 10688), (5642, 16032), (5642, 21376)]
                + [(11284, 5344), (11284, 10688), (11284, 16032), (11284, 21376)],
            )
        ],
        [
            (
                wsi_path,
                [(0, 16032), (0, 21376)]
                + [(5642, 10688), (5642, 16032), (5642, 21376)]
                + [(11284, 10688), (11284, 16032)],
            )
        ],
        [(wsi_path, [(5642, 10688), (5642, 16032), (11284, 10688), (11284, 16032)])],
        [(wsi_path, [(5642, 16032), (11284, 16032)])],
    ]

    for i in range(len(found_tiles) - 1):
        assert set(found_tiles[i + 1][0][1]).issubset(set(found_tiles[i + 1][0][1]))
    for i in range(len(found_tiles)):
        assert found_tiles[i] == expected_tiles[i]
    print("Test succeeded")


if __name__ == "__main__":
    test_mask_threshold()
