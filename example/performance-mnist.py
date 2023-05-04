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

"""
This is a script that is used to make timings of histomics_stream.  To some extent, it
may be specific to the computer / docker image it is used with and need minor tweaks to
run on another computer.
"""

import histomics_stream as hs
import histomics_stream.tensorflow
import os
import pooch
import tensorflow as tf
import tensorflow_datasets as tfds
import time

"""
# If you've just started a fresh docker container you may need some of this:
rm -rf /.local ; \
pip install -U pip setuptools wheel ; \
pip install \
    'batchbald_redux' \
    'black[jupyter]' \
    'large_image[openslide,tiff]' \
    'nbformat>=5.2.0' \
    'pooch' \
    'protobuf<3.20' \
    'tensorflow_datasets' \
    'torch==1.12.1+cu113' \
    '/tf/notebooks/histomics_stream' \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
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


class WrappedModel(tf.keras.Model):
    def __init__(self, model, *args, **kwargs):
        super(WrappedModel, self).__init__(*args, **kwargs)
        self.model = model

    def call(self, element):
        # Use just red of the color image
        return (self.model(element[0][..., 0]), element[1])


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


def build_model(training_batch, epochs):
    start_time = time.time()
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    print(f"Finished tfds.load in {time.time() - start_time}s", flush=True)

    start_time = time.time()
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(training_batch)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(training_batch)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    print(f"Finished (ds_train, ds_test) in {time.time() - start_time}s", flush=True)

    start_time = time.time()
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.fit(ds_train, epochs=epochs, validation_data=ds_test)

    unwrapped_model = model
    model = WrappedModel(unwrapped_model)

    print(f"Finished model in {time.time() - start_time}s", flush=True)
    return unwrapped_model, model


def create_study(wsi_path, mask_path, chunk_size):
    start_time = time.time()
    slide_name = os.path.splitext(os.path.split(wsi_path)[1])[0]
    slide_group = "Group 3"

    study = dict(
        version="version-1",
        tile_height=28,
        tile_width=28,
        overlap_height=14,
        overlap_width=14,
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
    create_tensorflow_dataset = hs.tensorflow.CreateTensorFlowDataset()
    tiles = create_tensorflow_dataset(study, num_workers=1, worker_index=0)
    print(f"#tiles = {len(create_tensorflow_dataset.get_tiles(study)[0][1])}")
    print(f"Chunked study in {time.time() - start_time}s", flush=True)

    return study, tiles


def predict(take_predictions, prediction_batch, model, tiles):
    start_time = time.time()
    tiles = tiles.batch(prediction_batch)
    if take_predictions > 0:
        predictions = model.predict(
            tiles.take(1 + (take_predictions - 1) // prediction_batch)
        )
    else:
        predictions = model.predict(tiles)
    print(f"predictions[0].shape = {predictions[0].shape}")
    print(f"Made predictions in {time.time() - start_time}s", flush=True)
    return predictions


if True:
    gpus = [gpu.name for gpu in tf.config.list_logical_devices("GPU")]
    print(f"gpus = {repr(gpus)}")

# if __name__ == "__main__":
with tf.device(gpus[0]):
    print("***** device = GPU *****")
    training_batch = 2**7
    num_epochs = 6
    take_predictions = 2**17 if False else 0

    wsi_path, mask_path = get_data()
    unwrapped_model, model = build_model(training_batch, num_epochs)

    for prediction_batch in [2**j for j in range(5, 11)]:
        for chunk_size in [28] + [2**j for j in range(6, 14)]:
            print(
                f"***** chunk_size = {chunk_size},"
                f" prediction_batch = {prediction_batch},"
                f" take_predictions = {take_predictions} ****",
                flush=True,
            )
            study, tiles = create_study(wsi_path, mask_path, chunk_size)
            predictions = predict(take_predictions, prediction_batch, model, tiles)
    print("***** Finished with device = GPU *****")
