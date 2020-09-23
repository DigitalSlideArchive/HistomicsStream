"""Whole-slide image file reader for TensorFlow"""

__version__ = "1.0.2"

from datetime import datetime
import h5py
from matplotlib import pyplot as plt
import numpy as np
import openslide as os
import tensorflow as tf
import time


def _assign_field(element, field, value):
  #Consistent with tensorflow graphs, this function does not modify
  #any of its inputs

  #Add a field to a struct. Use with tf.data.Dataset.map.
  return {**element, field: value}


def _add_fields(elem, fields, values):
  #Consistent with tensorflow graphs, this function does not modify
  #any of its inputs

  #Add fields to struct. Use with tf.data.Dataset.map.
  element = {**elem}
  for field, value in zip(fields, values):
    element[field] = value
  return element


def _expand_tensors(elem, fields, reference):
  #Consistent with tensorflow graphs, this function does not modify
  #any of its inputs

  #Expands tensor fields in dataset element to match the length of a reference
  #field. This allows application of tf.data.Dataset.flat_map to expand from
  #a read-chunk element dataset to a tile dataset using from_tensor_slices.

  element = {**elem}
  #get reference field length
  length = tf.size(element[reference])

  #This may lead to trouble when `reference` is of size/length 0
  #because every key in `elem`, whether or not in `fields`, will end
  #up being a list of length 0.  Unlike lists with a longer length,
  #all shape information of each value (in the key-value pair) is lost
  #with a list of length 0.  Especially if it is the first batch, this
  #may cause a later `flat_map` or `unbatch` command to fail.

  #iterate over fields removing singleton dimensions and repeating
  for field in fields:
    if tf.is_tensor(element[field]):
      element[field] = tf.repeat(tf.squeeze(element[field]), length)

    else: #raise exception - assume all fields are tensor type
      raise ValueError('Field is not tensor.')

  return element


def _tile_coords(width, height, tw, th, ow, oh, fractional):
#generates tiling coordinates for reading (th, tw) tiles with (ox, oy) overlap
#from a (height, width) image.
#For example, if image with `width` fits exactly `N` tiles with width `tw` and overlap `ow` then
#  width == N * (tw-ow) + ow,
#otherwise the tiles farthest to the right will not be of full width `tw`.
# We omit these narrower tiles unless `fractional` is set to true.
#... and similarly for heights instead of widths.

  '''
  if (tf.math.greater_equal(ow, tw)):
    print("bool = %s, ow = %s, tw = %s" % (tf.math.greater_equal(ow, tw), ow, tw))
    raise ValueError('Tile width overlap must be less than tile width.')
  if (tf.math.greater_equal(oh, th)):
    print("oh = %s, th = %s" % (oh, th))
    raise ValueError('Tile height overlap must be less than tile height.')
  '''

  zero = tf.constant(0, dtype=tf.int32)
  one = tf.constant(1, dtype=tf.int32)
  #Generate list of read coordinates
  left_limit = tf.maximum(zero, width-ow if fractional else width-tw+one)
  left = tf.range(zero, left_limit, tw-ow)
  right = tf.clip_by_value(left+tw, zero, width)

  top_limit = tf.maximum(zero, height-oh if fractional else height-th+one)
  top = tf.range(zero, top_limit, th-oh)
  bottom = tf.clip_by_value(top+th, zero, height)
  l = tf.repeat(left, tf.size(top))
  w = tf.repeat(right-left, tf.size(top))
  t = tf.tile(top, tf.stack([tf.size(left)]))
  h = tf.tile(bottom-top, tf.stack([tf.size(left)]))

  return l, t, w, h


def _add_tile_coords_fields(element, tile, overlap):
    tx, ty, tw, th = _tile_coords(element['cw'], element['ch'], tile[0], tile[1], overlap[0], overlap[1], False)
    return _add_fields(element, ['tx', 'ty', 'tw', 'th', 'ow', 'oh'], [tx, ty, tw, th, overlap[0], overlap[1]])


def get_read_parameters(filename, magnification, tolerance = 1e-2):

  #read whole-slide image file and create openslide object
  os_obj = os.OpenSlide(filename)

  #measure objective of level 0
  objective = np.float32(os_obj.properties[os.PROPERTY_NAME_OBJECTIVE_POWER])

  #calculate magnifications of levels
  estimated = np.array(objective / os_obj.level_downsamples)

  #calculate difference with magnification levels
  delta = magnification - estimated

  #match to existing levels
  if np.min(np.abs(np.divide(delta, magnification))) < tolerance: #match
    level = np.squeeze(np.argmin(np.abs(delta)))
    factor = 1.0
  elif np.any(delta < 0):
    value = np.max(delta[delta < 0])
    level = np.squeeze(np.argwhere(delta == value)[0])
    factor = magnification / estimated[level]
  else: #desired magnification above base level - throw error
    raise ValueError('Cannot interpolate above scan magnification.')

  #get slide width, height at desired magnification
  width, height = os_obj.level_dimensions[level]

  return level, factor, width, height


def read_region(filename, level, x, y, w, h):

  #open slide
  os_obj = os.OpenSlide(filename.numpy())

  #read chunk and convert to tensor
  chunk = os_obj.read_region((x.numpy(), y.numpy()),
                             level.numpy(),
                             (w.numpy(), h.numpy()))

  return tf.convert_to_tensor(np.array(chunk)[...,:3], dtype=tf.uint8)


def tf_read_region(filename, level, x, y, w, h):
  return tf.py_function(func=read_region,
                        inp=[filename, level, x, y, w, h],
                        Tout=tf.uint8)


def _read_chunk(element):

  #read chunk and convert to tensor
  chunk = tf_read_region(element['filename'],
                         tf.cast(element['level'], dtype=tf.int32),
                         element['cx'], element['cy'],
                         element['cw'], element['ch'])

  #split read chunk into tiles using a loop.
  #this avoids copying 'chunk' with 'map_fn' or 'tf.image.generate_glimpse'
  tiles = tf.TensorArray(dtype=tf.uint8, size=tf.size(element['tx']))
  condition = lambda i, _: tf.less(i, tf.size(element['tx']))
  body = lambda i, tiles: (i+1,
                           tiles.write(i, tf.image.crop_to_bounding_box(chunk,
                                                        tf.gather(element['ty'], i),
                                                        tf.gather(element['tx'], i),
                                                        tf.gather(element['th'], i),
                                                        tf.gather(element['tw'], i))))
  _, tiles = tf.while_loop(condition, body, [0, tiles])
  tiles = tiles.stack()

  #add tile tensor to element
  element['tiles'] = tiles

  #return dataset element dict
  return element


def _merge_dist_tensor(strategy, distributed, axis=0):
    #check if input is type roduced by distributed.Strategy.run
    if isinstance(distributed, tf.python.distribute.values.PerReplica):
        return tf.concat(strategy.experimental_local_results(distributed), axis=axis)
    else:
        raise ValueError('Input to _merge_dist_tensor not a distributed PerReplica tensor.')


def _merge_dist_dict(strategy, distributed, axis=0):
    #check if input is type roduced by distributed.Strategy.run
    if isinstance(distributed, dict):
        for key in distributed.keys():
            distributed[key] = _merge_dist_tensor(strategy, distributed[key], axis=axis)
        return distributed
    else:
        raise ValueError('Input to _merge_dist_tensor not a dict.')


def tiled(filename, slide='', case='', magnification=20.0, tile=(256, 256),
          overlap=(0, 0), chunkFactor=(4, 4), mask=None):
  """Generates a tf.data.Dataset where each element contains RGB pixel data
  generated by a regular-grid tiling of the slide. This function generates an
  intermediate dataset of 'read chunks' that contain many tiles, and that
  defines the actual reads made from the whole-slide image file. Functions are
  applied to tile these reads and stack them into a new dataset containing the
  tiles. This function is used to stream tiles to downstream preprocessing,
  inference, or training steps to hide read times with these compute-intensive
  operations.

  Related functions enable these tiles to be filtered and grouped by read chunk,
  slide, or case to enable complex logic for controlling downstream operations
  and for aggregating and organizing their outputs.

  Each element in the tile dataset is a dict where the tile is stored in
  element['tiles']. Metadata stored in other fields includes:

    slide - name of slide without file extension
    filename - filename and path of whole-slide image file.
    case - string describing case that slide is associated with.
    magnification - float describing objective magnification of tiles.
    cx, cy - location of upper-left corner of read chunk in pixels at native
      scan magnification. Used for calculating global coordinates.
    cw, ch - parameter width and height of read chunk in pixels.
    level - read level used to access pixel data from file.
    factor - resizing factor applied to read chunk images read from `level`. If
      < 1 then `magnification` is not available in the file and must be computed
      from a high magnification.
    tw, th - parameter tile width and height in pixels.
    tx, ty - local coordates of tile in read chunk.
    ow, oh - parameter overlap width and height in pixels.
    read_mode - string containing.

  Parameters
  ----------
  filename: string
    Path and filename of the slide readable by openslide.
  slide: string
    Slide name (used for operations that gather tiles by slide). Default ''.
  case: string
    Case identifier (used for gather operations on tiles). Default ''.
  magnification: float
    Desired objective magnification of pixel data tiles extracted from
    `filename`. Default 20.0.
  tile: (int, int)
    Tuple containing width and height respectively of extracted tiles. Default
    (256, 256).
  overlap: (int, int)
    Tuple containing horizontal and vertical overlap of tiles. Default (0, 0).
  chunkFactor: (int, int)
    Number of overlapping tiles that fit in a chunk in each direction. Critical for performance.
    These chunks are tiled and the tiles stacked to generate the output dataset.
    Default (4, 4).
  mask: array_like
    Boolean mask of slide indicating where tiles should be extracted. Acceptable
    types include _____.

  Returns
  -------
  tiled: tf.data.Dataset
    A dataset where each element contains an tuple of (RGB tile, metadata). The
    tile and metadata are separated in the tuple to satisfy unpacking rules.

  See Also
  --------
  dense: A function for generating tiles at arbitrary locations from a single
    read.
  """

  level, factor, width, height = get_read_parameters(filename, magnification)

  #**** error if magnification > native ****

  chunk = (overlap[0] + chunkFactor[0] * (tile[0] - overlap[0]), overlap[1] + chunkFactor[1] * (tile[1] - overlap[1]))
  #generate list of read coordinates (global frame, upper-left corner)
  cx, cy, cw, ch = _tile_coords(width, height,
                        chunk[0], chunk[1],
                        overlap[0], overlap[1], True)

  #generate read chunk dataset
  read = tf.data.Dataset.from_tensor_slices({'cx': cx, 'cy': cy, 'cw': cw, 'ch': ch})

  #add general parameters
  read = read.map(lambda elem: _add_fields(elem, ['slide', 'case', 'filename',
                                                  'magnification', 'read_mode'],
                                          [slide, case, filename,
                                           magnification, 'tiled']),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

  #add chunk parameters
  read = read.map(lambda elem:
                  _add_fields(elem, ['level', 'factor'],
                              [level, 1.0]),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

  #**** error if mismatch between tile size and read chunk size ****

  #generate list of tile coordinates (local frame, upper-left corner)
  read = read.map(lambda elem:
                  _add_tile_coords_fields(elem, tile, overlap),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

  #apply this function to the read chunk dataset
  read = read.map(_read_chunk, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  #expand dimensions to match number of tiles
  read = read.map(lambda element:
                  _expand_tensors(element, ['slide', 'filename', 'case',
                                            'magnification', 'cx', 'cy',
                                            'cw', 'ch', 'level', 'factor',
                                            'ow', 'oh',
                                            'read_mode'], 'tx'),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

  #flatten dataset to make each element one tile
  tiled = read.flat_map(lambda element:
                        tf.data.Dataset.from_tensor_slices(element))

  #split tile, metadata into tuple
  tiled = tiled.map(lambda element: (element.pop('tiles'), element))

  return tiled

def strategyExample():
  #find available GPUs
  devices=[gpu.name.replace('/physical_device:', '/').lower() for gpu in tf.config.experimental.list_physical_devices('GPU')]

  #define strategy
  strategy = tf.distribute.MirroredStrategy(devices=devices)

  #generate network model
  with strategy.scope():
      model = tf.keras.applications.VGG16(include_top=True, weights='imagenet')
      model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('fc1').output)

  return devices, strategy, model


def readExample(chunkFactor = 8):
  tic = time.time()
  #using numpy_function allows mapping of keras functions to dataset elements

  devices, strategy, model = strategyExample()

  #set slide and read parameters
  filename='TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913.svs'
  slide='TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913'
  case='TCGA-BH-A0BZ'
  magnification=20.0
  tile=(tf.constant(256, dtype=tf.int32),
        tf.constant(256, dtype=tf.int32))
  chunkFactor=(tf.constant(chunkFactor, dtype=tf.int32),
               tf.constant(chunkFactor, dtype=tf.int32))
  overlap=(tf.constant(0, dtype=tf.int32),
           tf.constant(0, dtype=tf.int32))
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  batch_size = 128*len(devices)

  #generate tiles dataset
  tiles = tiled(filename, slide, case, magnification,
                tile, overlap, chunkFactor, mask=None)

  #batch tiles
  batched = tiles.batch(batch_size)

  #apply preprocessing to batched tiles - resize, float conversion, preprocessing
  batched = batched.map(lambda tile, metadata:
                        (tf.cast(tf.image.resize(tile, [224, 224]), tf.float32),
                         metadata),
                        num_parallel_calls=AUTOTUNE)
  batched = batched.map(lambda tile, metadata:
                        (tf.keras.applications.resnet_v2.preprocess_input(tile),
                         metadata),
                        num_parallel_calls=AUTOTUNE)

  #set prefetch
  batched_dist = strategy.experimental_distribute_dataset(batched)
  print('Read and resize dataset: %f seconds' % (time.time() - tic))
  return model, batched_dist, strategy


#wrap prediction function in graph
@tf.function
def predict(model, element):
    return model(element[0]), element[1]


def predictExample(model, batched_dist, strategy):
  tic = time.time()

  #distributed inference, condensing distributed feature tensors, metadata dicts in lists
  feature_list = []
  metadata_list = []
  for element in batched_dist:
      f, meta = strategy.run(predict, args=(model,element,))
      feature_list.append(_merge_dist_tensor(strategy, f))
      metadata_list.append(_merge_dist_dict(strategy, meta))

  #merge features into single array
  features = tf.concat(feature_list, axis=0)
  del feature_list

  #merge metadata into single dict
  metadata = {}
  for key in metadata_list[0].keys():
      metadata[key] = tf.concat([meta[key] for meta in metadata_list], axis=0)
  del metadata_list

  #map tile coordinates from chunk frame to global slide frame
  metadata['tx'] = metadata['tx'] + metadata['cx']
  metadata['ty'] = metadata['ty'] + metadata['cy']

  print('Predict distributed dataset: %f seconds' % (time.time() - tic))
  return features, metadata


def outputExample(features, metadata):
  tic = time.time()

  #write features, metadata to disk
  with h5py.File('mytestfile.hdf5', 'w') as handle:
      handle.create_dataset('slides', data=metadata['slide'].numpy(),
                            dtype=h5py.string_dtype(encoding='ascii'))
      handle.create_dataset('features', data=features.numpy(), dtype='float')
      handle.create_dataset('slideIdx', data=np.zeros(metadata['slide'].shape), dtype='int')
      handle.create_dataset('x_centroid', data=metadata['tx'].numpy(), dtype='float')
      handle.create_dataset('y_centroid', data=metadata['ty'].numpy(), dtype='float')
      handle.create_dataset('dataIdx', data=np.zeros(1), dtype='int')
      handle.create_dataset('wsi_mean', data=np.zeros(3), dtype='float')
      handle.create_dataset('wsi_std', data=np.zeros(3), dtype='float')

  print('Writing h5 data: %f seconds' % (time.time() - tic))

  #write superpixel boundaries to disk
