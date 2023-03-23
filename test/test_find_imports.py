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


def test_imports_can_be_found():
    """Purpose: Test to check that each import can be found"""

    import imagecodecs  # noqa: F401
    import itk  # noqa: F401
    import numcodecs  # noqa: F401
    import numpy  # noqa: F401
    import scipy.interpolate  # noqa: F401
    import tensorflow  # noqa: F401
    import torch  # noqa: F401
    import zarr  # noqa: F401


if __name__ == "__main__":
    test_imports_can_be_found()
