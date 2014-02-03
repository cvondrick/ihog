dirpath = '/Users/vondrick/datasets/VOC2011/JPEGImages'
outdir = 'data/'
dim = 256
skip = 16
layers = ['pool5_cudanet_out', 'fc6_neuron_cudanet_out', 'fc7_neuron_cudanet_out']




from decaf.scripts.imagenet import DecafNet
import scipy.io
import Image
import numpy
import os
import multiprocessing
import sys
import random

def process(filename):
  net = DecafNet('imagenet_pretrained/imagenet.decafnet.epoch90', 'imagenet_pretrained/imagenet.decafnet.meta')
  filepath = os.path.join(dirpath, filename)
  print 'reading {0}'.format(filepath)
  im = Image.open(filepath)
  data = numpy.asarray(im)

  crops = []
  locations = []
  featdim = None
  features = {}
  for layer in layers:
    features[layer] = []
  features['scores'] = []

  for i in range(0, data.shape[0] - dim, skip):
    for j in range(0, data.shape[1] - dim, skip):
      crop = data[i:i+dim, j:j+dim, :];
      scores = net.classify(crop, center_only = True)

      crops.append(numpy.expand_dims(crop, 0))
      locations.append((i, j))

      for layer in layers:
        feature = net.feature(layer)
        features[layer].append(feature)
      features['scores'].append(scores)

  if len(crops) == 0:
    print "found no crops for {0}".format(filepath)
    return

  cropmatrix = numpy.vstack(crops)
  locationmatrix = numpy.vstack(locations)

  payload = {'images': cropmatrix, 
             'locations': locationmatrix,
             'imsize': im.size,
             'cropdim': [dim, dim, 3],
             'layers': layers}
  for layer in features:
    payload[layer] = numpy.vstack(features[layer])

  outpath = os.path.join(outdir, '{0}.mat'.format(filename))
  print 'saving {0}'.format(outpath)
  scipy.io.savemat(outpath, payload, oned_as = 'row')

filenames = [filename for filename in os.listdir(dirpath) if filename.endswith('.jpg')]
random.shuffle(filenames)

pool = multiprocessing.Pool(processes = 3)
pool.map(process, filenames)

#map(process, filenames)
