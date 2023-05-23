import os
import sys
import torch
import numpy as np

import torch.nn.functional as F


def load_mnist(root, flatten = False, valid_size = 5000, image_size = 28, normalize = True):
    """taken from https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py"""
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source = 'http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, os.path.join(root, filename))

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    os.makedirs(root, exist_ok=True)

    def load_mnist_images(filename):
        if not os.path.exists(os.path.join(root, filename)):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(os.path.join(root, filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        if image_size != 28:
            data = F.interpolate(torch.from_numpy(data.astype(np.float32)),
                                 (image_size, image_size),
                                 mode="bilinear").numpy()
        if normalize:
            return data.astype(np.float32) * 2 / 255 - 1
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(os.path.join(root, filename)):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(os.path.join(root, filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 5000 training examples for validation.
    if valid_size == 0:
        X_val, y_val = np.empty((0, 1, image_size, image_size)), np.empty(0)
    else:
        X_train, X_val = X_train[:-valid_size], X_train[-valid_size:]
        y_train, y_val = y_train[:-valid_size], y_train[-valid_size:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        if valid_size > 0:
            X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


class MNISTDataloader:
    def __init__(self, inputs, targets, batchsize, shuffle = False):
        assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets
        self.batchsize = batchsize
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(len(self.inputs))
        for start_idx in range(0, len(self.inputs) - self.batchsize + 1, self.batchsize):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + self.batchsize]
            else:
                excerpt = slice(start_idx, start_idx + self.batchsize)
            yield (torch.from_numpy(self.inputs[excerpt].astype(np.float32)),
                   torch.from_numpy(self.targets[excerpt].astype(np.int64)))

    def __len__(self):
        return len(self.inputs) // self.batchsize
