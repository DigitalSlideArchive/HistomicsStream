"""Whole-slide image file reader for TensorFlow"""

__version__ = "1.0.5"

from datetime import datetime
import h5py
from matplotlib import pyplot as plt
import numpy as np
import openslide as os
import tensorflow as tf
import time


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


def strategyExample():
  tic = time.time()
  #find available GPUs
  devices=[gpu.name.replace('/physical_device:', '/').lower() for gpu in tf.config.experimental.list_physical_devices('GPU')]

  #define strategy
  strategy = tf.distribute.MirroredStrategy(devices=devices)

  #generate network model
  with strategy.scope():
      model = tf.keras.applications.VGG16(include_top=True, weights='imagenet')
      model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('fc1').output)

  print('strategyExample: %f seconds' % (time.time() - tic))
  return devices, strategy, model


def readExample(devices, strategy, model):
  tic = time.time()

  AUTOTUNE = {'num_parallel_calls': tf.data.experimental.AUTOTUNE}
  # Each key=value pair in the dictionary should have a value that is
  # a list of the same length.  Taken together, the Nth entry from
  # each list comprise a dictionary that is the Nth element in the
  # dataset.
  header = {
    'slide': ['TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913'],
    'case': ['TCGA-BH-A0BZ'],
    'filename': ['TCGA-BH-A0BZ-01Z-00-DX1.45EB3E93-A871-49C6-9EAE-90D98AE01913.svs'],
    'magnification': [20.0],
    'read_mode': ['tiled'],
  }
  tiles = tf.data.Dataset.from_tensor_slices(header)

  # For the desired magnification, find the best level stored in the
  # image file, and its associated factor, width, and height.
  tiles = tiles.map(tf_ComputeReadParameters)

  # We are going to use a regularly spaced tiling for each of our
  # dataset elements.  To begin, set the desired tile width, height,
  # width overlap, and height overlap for each element, and indicate
  # whether we want tiles even if they are fractional (aka of
  # truncated size) due to partially falling off the edge of the
  # image.  (These fractional tiles can be problematic because
  # tensorflow likes its shapes to be uniform.)
  tileWidth = tf.constant(256, dtype=tf.int32)
  tileHeight = tileWidth
  overlapWidth = tf.constant(0, dtype=tf.int32)
  overlapHeight = overlapWidth
  # (chunkWidthFactor, chunkHeightFactor) indicates how many
  # (overlapping) tiles are read at a time.
  chunkWidthFactor = tf.constant(8, dtype=tf.int32)
  chunkHeightFactor = chunkWidthFactor
  fractional = tf.constant(False, dtype=tf.bool)
  newFields = {'tw': tileWidth, 'th': tileHeight,
               'ow': overlapWidth, 'oh': overlapHeight,
               'cwf': chunkWidthFactor, 'chf': chunkHeightFactor,
               'fractional': fractional}
  tiles = tiles.map(lambda elem: {**elem, **newFields})

  # Split each element (e.g. each slide) into a batch of multiple
  # rows, one per chunk to be read.  Note that the width `cw` or
  # height `ch` of a row (chunk) may decreased from the requested
  # value if a chunk is near the edge of an image.  Note that it is
  # important to call `.unbatch()` when it is desired that the chunks
  # be not batched by slide.
  tiles = tiles.map(tf_ComputeChunkPositions, **AUTOTUNE).unbatch()

  # Read and split the chunks into the tile size we want.  Note that
  # it is important to call `.unbatch()` when it is desired that the
  # tiles be not batched by chunk.
  tiles = tiles.map(tf_ReadAndSplitChunk, **AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE).unbatch()

  # Change the element to be of the form `(tile, metadataDictionary)`
  # rather than `alldataDictionary`.
  tiles = tiles.map(lambda elem: (elem.pop('tile'), elem), **AUTOTUNE)

  batch_size = 128*len(devices)

  #batch tiles
  batched = tiles.batch(batch_size)

  #apply preprocessing to batched tiles - resize, float conversion, preprocessing
  batched = batched.map(lambda tile, metadata:
                        (tf.cast(tf.image.resize(tile, [224, 224]), tf.float32),
                         metadata), **AUTOTUNE)
  batched = batched.map(lambda tile, metadata:
                        (tf.keras.applications.resnet_v2.preprocess_input(tile),
                         metadata), **AUTOTUNE)

  batched_dist = strategy.experimental_distribute_dataset(batched)
  print('readExample: %f seconds' % (time.time() - tic))
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

  print('predictExample: %f seconds' % (time.time() - tic))
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

  print('outputExample: %f seconds' % (time.time() - tic))

  #write superpixel boundaries to disk


def py_ComputeReadParameters(filenameIn, magnificationIn, toleranceIn):
  filename = filenameIn.numpy()
  magnification = magnificationIn.numpy()
  tolerance = toleranceIn.numpy()

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


def tf_ComputeReadParameters(elem, tolerance = tf.constant(0.02, dtype=tf.float32)):
  level, factor, width, height = tf.py_function(
    func=py_ComputeReadParameters,
    inp=[elem['filename'], elem['magnification'], tolerance],
    Tout=(tf.int32, tf.float32, tf.int32, tf.int32))
  return {**elem, 'level': level, 'factor': factor, 'width': width, 'height': height }


def tf_ComputeChunkPositions(elem):
  zero = tf.constant(0, dtype=tf.int32)
  one = tf.constant(1, dtype=tf.int32)
  chunkWidth = elem['cwf'] * (elem['tw'] - elem['ow']) + elem['ow']
  chunkHeight = elem['chf'] * (elem['th'] - elem['oh']) + elem['oh']

  ## The left side of a tile cannot be as large as left_bound.  Also,
  ## the left side of a chunk cannot be as large as left_bound because
  ## chunks contain a whole number of tiles.
  left_bound = tf.maximum(zero,
                          elem['width'] - elem['ow'] if elem['fractional'] else elem['width'] - elem['tw'] + one)
  chunkLeft = tf.range(zero, left_bound, chunkWidth - elem['ow'])
  chunkRight = tf.clip_by_value(chunkLeft + chunkWidth, zero, elem['width'])

  top_bound = tf.maximum(zero,
                         elem['height'] - elem['oh'] if elem['fractional'] else elem['height'] - elem['th'] + one)
  chunkTop = tf.range(zero, top_bound, chunkHeight - elem['oh'])
  chunkBottom = tf.clip_by_value(chunkTop + chunkHeight, zero, elem['height'])

  x = tf.tile(chunkLeft, tf.stack([tf.size(chunkTop)]))
  w = tf.tile(chunkRight-chunkLeft, tf.stack([tf.size(chunkTop)]))
  y = tf.repeat(chunkTop, tf.size(chunkLeft))
  h = tf.repeat(chunkBottom-chunkTop, tf.size(chunkLeft))
  len = tf.size(x)

  response = {}
  for key in elem.keys():
    response[key] = tf.repeat(elem[key], len)
  return {**response, 'cx': x, 'cy': y, 'cw': w, 'ch': h}


def py_ReadChunk(filename, level, x, y, w, h):
  #open slide
  os_obj = os.OpenSlide(filename.numpy())

  #read chunk and convert to tensor
  chunk = os_obj.read_region((x.numpy(), y.numpy()),
                             level.numpy(),
                             (w.numpy(), h.numpy()))

  return tf.convert_to_tensor(np.array(chunk)[...,:3], dtype=tf.uint8)


def tf_ReadAndSplitChunk(elem):
  zero = tf.constant(0, dtype=tf.int32)
  one = tf.constant(1, dtype=tf.int32)
  left_bound = tf.maximum(zero,
                          elem['cw'] - elem['ow'] if elem['fractional'] else elem['cw'] - elem['tw'] + one)
  tileLeft = tf.range(zero, left_bound, elem['tw'] - elem['ow'])
  tileRight = tf.clip_by_value(tileLeft + elem['tw'], zero, elem['cw'])

  top_bound = tf.maximum(zero,
                         elem['ch'] - elem['oh'] if elem['fractional'] else elem['ch'] - elem['th'] + one)
  tileTop = tf.range(zero, top_bound, elem['th'] - elem['oh'])
  tileBottom = tf.clip_by_value(tileTop + elem['th'], zero, elem['ch'])

  x = tf.tile(tileLeft, tf.stack([tf.size(tileTop)]))
  w = tf.tile(tileRight-tileLeft, tf.stack([tf.size(tileTop)]))
  y = tf.repeat(tileTop, tf.size(tileLeft))
  h = tf.repeat(tileBottom-tileTop, tf.size(tileLeft))
  len = tf.size(x)

  chunk = tf.py_function(
            func=py_ReadChunk,
            inp=[elem['filename'], elem['level'], elem['cx'], elem['cy'], elem['cw'], elem['ch']],
            Tout=tf.uint8)

  tiles = tf.TensorArray(dtype=tf.uint8, size=len)
  condition = lambda i, _: tf.less(i, len)
  body = lambda i, tiles: (
    i+1, tiles.write(
      i, tf.image.crop_to_bounding_box(chunk,
                                       tf.gather(y, i),
                                       tf.gather(x, i),
                                       tf.gather(h, i),
                                       tf.gather(w, i))))
  _, tiles = tf.while_loop(condition, body, [0, tiles])
  tiles = tiles.stack()
  del chunk

  response = {}
  for key in elem.keys():
    response[key] = tf.repeat(elem[key], len)

  return {**response, 'tx': elem['cx']+x, 'ty': elem['cy']+y, 'tw': w, 'th': h, 'tile': tiles}
