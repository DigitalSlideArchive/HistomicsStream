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

import argparse
import histomics_stream as hs
import histomics_stream.pytorch
import itertools
import os
import pooch
import time
import torch
import torchvision

"""
This is a script that is used to make timings of histomics_stream.  To some extent, it
may be specific to the computer / docker image it is used with and need minor tweaks to
run on another computer.
"""

"""
# If you've just started a fresh docker container you may need some of this:
apt update ; apt install -y git emacs ; \
rm -rf /.local ; \
pip install -U pip setuptools wheel pillow ; \
pip install \
    'black[jupyter]' \
    'large_image[openslide,tiff]' \
    'monai[pillow,tqdm,ignite,gdown]' \
    'nbformat>=5.2.0' \
    'pooch' \
    'protobuf' \
    '/tf/notebooks/histomics_stream' \
    --find-links https://girder.github.io/large_image_wheels
"""


def get_data():
    start_time = time.time()
    wsi_path = pooch.retrieve(
        fname="TCGA-AN-A0G0-01Z-00-DX1.svs",
        url="https://drive.google.com/uc"
        "?export=download"
        "&id=19agE_0cWY582szhOVxp9h3kozRfB4CvV"
        "&confirm=t"
        "&uuid=6f2d51e7-9366-4e98-abc7-4f77427dd02c"
        "&at=ALgDtswlqJJw1KU7P3Z1tZNcE01I:1679111148632",
        known_hash="d046f952759ff6987374786768fc588740eef1e54e4e295a684f3bd356c8528f",
        path=str(pooch.os_cache("pooch")) + os.sep + "wsi",
    )
    print(f"Retrieved {wsi_path} in {time.time() - start_time}s", flush=True)

    # download binary mask image
    start_time = time.time()
    mask_path = pooch.retrieve(
        fname="TCGA-AN-A0G0-01Z-00-DX1.mask.png",
        url="https://drive.google.com/uc"
        "?export=download"
        "&id=17GOOHbL8Bo3933rdIui82akr7stbRfta",
        known_hash="bb657ead9fd3b8284db6ecc1ca8a1efa57a0e9fd73d2ea63ce6053fbd3d65171",
        path=str(pooch.os_cache("pooch")) + os.sep + "wsi",
    )
    print(f"Retrieved {mask_path} in {time.time() - start_time}s", flush=True)
    return wsi_path, mask_path


class WrappedModel(torch.nn.modules.module.Module):
    def __init__(self, model, preprocess_fn, *args, device="cuda", **kwargs):
        super(WrappedModel, self).__init__(*args, **kwargs)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.preprocess_fn = preprocess_fn.to(self.device)

    def forward(self, x):
        p = self.model(self.preprocess_fn(x[0].to(self.device)))
        return p, x[1]


def build_model(device="cuda"):
    start_time = time.time()
    # print(f"available_models = {repr(sorted(torchvision.models.list_models()))}")
    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    model = torchvision.models.efficientnet_v2_s(weights=weights)
    _ = model.eval()
    preprocess_fn = weights.transforms()

    unwrapped_model = model
    model = WrappedModel(unwrapped_model, preprocess_fn, device=device).to(device)

    print(f"Finished model in {time.time() - start_time}s", flush=True)
    return unwrapped_model, model


def create_study(wsi_path, mask_path, chunk_size):
    start_time = time.time()
    slide_name = os.path.splitext(os.path.split(wsi_path)[1])[0]
    slide_group = "Group 3"

    study = dict(
        version="version-1",
        tile_height=224,
        tile_width=224,
        overlap_height=0,
        overlap_width=0,
        slides=dict(
            Slide_0=dict(
                filename=wsi_path,
                slide_name=slide_name,
                slide_group=slide_group,
                chunk_height=chunk_size,
                chunk_width=chunk_size,
            )
        ),
    )

    find_slide_resolution = hs.configure.FindResolutionForSlide(
        study, target_magnification=20, magnification_source="exact"
    )
    tiles_by_grid_and_mask = hs.configure.TilesByGridAndMask(
        study, mask_filename=mask_path
    )
    # We could apply these to a subset of the slides, but we will apply it to all slides
    # in this example.
    for slide in study["slides"].values():
        find_slide_resolution(slide)
        tiles_by_grid_and_mask(slide)
    print(f"Masked study in {time.time() - start_time}s", flush=True)

    start_time = time.time()
    create_torch_dataloader = hs.pytorch.CreateTorchDataloader()
    tiles = create_torch_dataloader(study)
    print(f"#tiles = {len(create_torch_dataloader.get_tiles(study)[0][1])}")
    print(f"Chunked study in {time.time() - start_time}s", flush=True)
    return study, tiles


def show_structure(x):
    if isinstance(x, list):
        if len(x) > 0:
            return f"[{len(x)} of {show_structure(x[0])}]"
        else:
            return repr(list())
    if isinstance(x, tuple):
        if len(x) > 0:
            return f"({len(x)} of {show_structure(x[0])})"
        else:
            return repr(tuple())
    if isinstance(x, set):
        if len(x) > 0:
            return f"{{{len(x)} of {show_structure(next(iter(x)))}}}"
        else:
            return repr(set())
    if isinstance(x, dict):
        if len(x) > 0:
            return f"{{{len(x)} of {show_structure(next(iter(x.keys())))}: {show_structure(next(iter(x.values())))}}}"
        else:
            return repr(dict())
    return repr(type(x))


"""
!!! Probably we should be using torch.utils.data.DataLoader batch_size option instead of
!!! this batched() function.
"""


def batched(iterable, batch_size):
    """
    Batch data into lists of length batch_size. The last batch may be shorter:
    batched('ABCDEFG', 3) --> ABC DEF G
    """
    iterator = iter(iterable)
    # !!! Can we get rid of `list` here and a few lines below?  It is used so that we
    # !!! can detect an empty list with `while`.
    batch = list(itertools.islice(iterator, batch_size))
    while batch:
        # Yield `batch` in such a way that this iterator does not keep a reference count
        # for it.
        batch_in_list = [batch]
        del batch
        yield batch_in_list.pop()
        batch = list(itertools.islice(iterator, batch_size))


def predict_and_detach(model, item):
    predict = model(item)
    return predict[0].detach().cpu().numpy(), predict[1]


def predict(take_predictions, prediction_batch, model, tiles):
    start_time = time.time()
    if take_predictions > 0:
        tiles = itertools.islice(tiles, take_predictions)
    batched_tiles = (
        batched(tiles, prediction_batch) if prediction_batch > 0 else [tiles]
    )
    predictions = list()
    for batch in batched_tiles:
        batch_predictions = [predict_and_detach(model, item) for item in batch]
        predictions.extend(batch_predictions)
    del batch_predictions, batch
    print(f"Made predictions in {time.time() - start_time}s", flush=True)
    return predictions


def create_and_predict(
    wsi_path, mask_path, chunk_size, take_predictions, prediction_batch, model
):
    study, tiles = create_study(wsi_path, mask_path, chunk_size)
    predictions = predict(take_predictions, prediction_batch, model, tiles)
    print(f"show_structure(predictions) = {show_structure(predictions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    args = parser.parse_args()
    # device = "cuda" if True else "cpu"
    device = args.device
    print(f"***** device = {device} *****")
    take_predictions = 2**8 if True else 0

    wsi_path, mask_path = get_data()
    unwrapped_model, model = build_model(device=device)

    # for prediction_batch in [2**j for j in range(0, 6)]:
    for prediction_batch in [0]:
        for chunk_size in [1024] + [2**j for j in range(8, 14)]:
            print(
                f"***** chunk_size = {chunk_size},"
                f" prediction_batch = {prediction_batch},"
                f" take_predictions = {take_predictions} ****",
                flush=True,
            )
            create_and_predict(
                wsi_path,
                mask_path,
                chunk_size,
                take_predictions,
                prediction_batch,
                model,
            )
    print(f"***** Finished with device = {device} *****")
