import tensorflow as tf
import math
import os
import numpy as np
from scipy import misc
import cPickle as pickle
import glob
import scipy.optimize


def load_cifar_XandY(data_dir):
    print('Loading cifar10...')
    with open(data_dir+'/data_batch_1', 'rb') as fo:
        train1 = pickle.load(fo)
        train1X = np.reshape(train1['data'],[10000,3,32,32])
        train1Y = np.array(train1['labels'])
    with open(data_dir+'/data_batch_2', 'rb') as fo:
        train2 = pickle.load(fo)
        train2X = np.reshape(train2['data'],[10000,3,32,32])
        train2Y = np.array(train2['labels'])

    with open(data_dir+'/data_batch_3', 'rb') as fo:
        train3 = pickle.load(fo)
        train3X = np.reshape(train3['data'],[10000,3,32,32])
        train3Y = np.array(train3['labels'])

    with open(data_dir+'/data_batch_4', 'rb') as fo:
        train4 = pickle.load(fo)
        train4X = np.reshape(train4['data'],[10000,3,32,32])
        train4Y = np.array(train4['labels'])

    with open(data_dir+'/data_batch_5', 'rb') as fo:
        train5 = pickle.load(fo)
        train5X = np.reshape(train5['data'],[10000,3,32,32])
        train5Y = np.array(train5['labels'])

    with open(data_dir+'/test_batch', 'rb') as fo:
        test = pickle.load(fo)
        testX = np.reshape(test['data'],[10000,3,32,32])
        testY = np.array(test['labels'])


    trX = np.concatenate((train1X,train2X,train3X,train4X,train5X),axis=0).astype(np.float)
    trX = np.swapaxes(trX, 1, 3)
    trX = np.swapaxes(trX, 1, 2)
    trY = np.concatenate((train1Y,train2Y,train3Y,train4Y,train5Y),axis=0).astype(np.int)
    teX = np.swapaxes(testX, 1, 3)
    teX = np.swapaxes(teX, 1, 2).astype(np.float)
    teY = testY.astype(np.int)

    print('Done.')
    return trX/255., trY, teX/255., teY

def generateTargetReps(n, z):
    # Use Marsaglias algorithm to generate targets on z-unit-sphere
    samples = np.random.normal(0, 1, [n, z]).astype(np.float32)
    radiuses = np.expand_dims(np.sqrt(np.sum(np.square(samples),axis=1)),1)
    reps = samples/radiuses
    return reps

def calc_optimal_target_permutation(reps, targets):
    # Compute cost matrix
    cost_matrix = np.zeros([reps.shape[0],targets.shape[0]])
    for i in range(reps.shape[0]):
        cost_matrix[:,i] = np.sum(np.square(reps-targets[i,:]),axis=1)

    _, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    # Permute
    targets[range(reps.shape[0])] = targets[col_ind]
    return targets

def shuffle_together(array1, array2):
    assert(array1.shape[0]==array2.shape[0])
    randomize = np.arange(array1.shape[0])
    np.random.shuffle(randomize)
    array1 = array1[randomize]
    array2 = array2[randomize]
    return array1, array2


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  print(images.shape)
  img = np.zeros((h * size[0], w * size[1], 3))
  for idx, image in enumerate(images[0:size[0]*size[1], :, :, :]):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img


def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
  return (images+1.)/2.

def imsave(images, size, path):
    arr = merge(images, size)
    print(arr.shape)
    return misc.imsave(path,arr)
