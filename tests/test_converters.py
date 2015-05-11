import argparse
import gzip
import os
import struct
import tarfile

import h5py
import numpy
from numpy.testing import assert_equal, assert_raises
from six.moves import range, zip, cPickle

from fuel.converters.base import fill_hdf5_file
from fuel.converters import binarized_mnist, cifar10, mnist


class TestFillHDF5File(object):
    def setUp(self):
        self.h5file = h5py.File(
            'file.hdf5', mode='w', driver='core', backing_store=False)
        self.train_features = numpy.arange(
            16, dtype='uint8').reshape((4, 2, 2))
        self.test_features = numpy.arange(
            8, dtype='uint8').reshape((2, 2, 2)) + 3
        self.train_targets = numpy.arange(
            4, dtype='float32').reshape((4, 1))
        self.test_targets = numpy.arange(
            2, dtype='float32').reshape((2, 1)) + 3

    def tearDown(self):
        self.h5file.close()

    def test_data(self):
        fill_hdf5_file(
            self.h5file,
            (('train', 'features', self.train_features, '.'),
             ('train', 'targets', self.train_targets),
             ('test', 'features', self.test_features),
             ('test', 'targets', self.test_targets)))
        assert_equal(self.h5file['features'],
                     numpy.vstack([self.train_features, self.test_features]))
        assert_equal(self.h5file['targets'],
                     numpy.vstack([self.train_targets, self.test_targets]))

    def test_dtype(self):
        fill_hdf5_file(
            self.h5file,
            (('train', 'features', self.train_features),
             ('train', 'targets', self.train_targets),
             ('test', 'features', self.test_features),
             ('test', 'targets', self.test_targets)))
        assert_equal(str(self.h5file['features'].dtype), 'uint8')
        assert_equal(str(self.h5file['targets'].dtype), 'float32')

    def test_multiple_length_error(self):
        train_targets = numpy.arange(8, dtype='float32').reshape((8, 1))
        assert_raises(ValueError, fill_hdf5_file, self.h5file,
                      (('train', 'features', self.train_features),
                       ('train', 'targets', train_targets)))

    def test_multiple_dtype_error(self):
        test_features = numpy.arange(
            8, dtype='float32').reshape((2, 2, 2)) + 3
        assert_raises(
            ValueError, fill_hdf5_file, self.h5file,
            (('train', 'features', self.train_features),
             ('test', 'features', test_features)))

    def test_multiple_shape_error(self):
        test_features = numpy.arange(
            16, dtype='uint8').reshape((2, 4, 2)) + 3
        assert_raises(
            ValueError, fill_hdf5_file, self.h5file,
            (('train', 'features', self.train_features),
             ('test', 'features', test_features)))


class TestMNIST(object):
    def setUp(self):
        MNIST_IMAGE_MAGIC = 2051
        MNIST_LABEL_MAGIC = 2049
        numpy.random.seed(9 + 5 + 2015)
        self.train_features_mock = numpy.random.randint(
            0, 256, (10, 1, 28, 28)).astype('uint8')
        self.train_targets_mock = numpy.random.randint(
            0, 10, (10, 1)).astype('uint8')
        self.test_features_mock = numpy.random.randint(
            0, 256, (10, 1, 28, 28)).astype('uint8')
        self.test_targets_mock = numpy.random.randint(
            0, 10, (10, 1)).astype('uint8')
        with gzip.open('train-images-idx3-ubyte.gz', 'wb') as f:
            f.write(struct.pack('>iiii', *(MNIST_IMAGE_MAGIC, 10, 28, 28)))
            f.write(numpy.getbuffer(self.train_features_mock.flatten()))
        with gzip.open('train-labels-idx1-ubyte.gz', 'wb') as f:
            f.write(struct.pack('>ii', *(MNIST_LABEL_MAGIC, 10)))
            f.write(numpy.getbuffer(self.train_targets_mock.flatten()))
        with gzip.open('t10k-images-idx3-ubyte.gz', 'wb') as f:
            f.write(struct.pack('>iiii', *(MNIST_IMAGE_MAGIC, 10, 28, 28)))
            f.write(numpy.getbuffer(self.test_features_mock.flatten()))
        with gzip.open('t10k-labels-idx1-ubyte.gz', 'wb') as f:
            f.write(struct.pack('>ii', *(MNIST_LABEL_MAGIC, 10)))
            f.write(numpy.getbuffer(self.test_targets_mock.flatten()))

    def tearDown(self):
        os.remove('train-images-idx3-ubyte.gz')
        os.remove('train-labels-idx1-ubyte.gz')
        os.remove('t10k-images-idx3-ubyte.gz')
        os.remove('t10k-labels-idx1-ubyte.gz')
        os.remove('mock_mnist.hdf5')

    def test_converter(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        subparser = subparsers.add_parser('mnist')
        subparser.set_defaults(directory='./', output_file='mock_mnist.hdf5')
        mnist.fill_subparser(subparser)
        args = parser.parse_args(['mnist'])
        args.func(args)
        h5file = h5py.File('mock_mnist.hdf5', mode='r')
        assert_equal(
            h5file['features'][...],
            numpy.vstack(
                [self.train_features_mock, self.test_features_mock]))
        assert_equal(
            h5file['targets'][...],
            numpy.vstack([self.train_targets_mock, self.test_targets_mock]))
        assert_equal(str(h5file['features'].dtype), 'uint8')
        assert_equal(str(h5file['targets'].dtype), 'uint8')
        assert_equal(tuple(dim.label for dim in h5file['features'].dims),
                     ('batch', 'channel', 'height', 'width'))
        assert_equal(tuple(dim.label for dim in h5file['targets'].dims),
                     ('batch', 'index'))


class TestBinarizedMNIST(object):
    def setUp(self):
        numpy.random.seed(9 + 5 + 2015)
        self.train_mock = numpy.random.randint(0, 2, (5, 784))
        self.valid_mock = numpy.random.randint(0, 2, (5, 784))
        self.test_mock = numpy.random.randint(0, 2, (5, 784))
        numpy.savetxt('binarized_mnist_train.amat', self.train_mock)
        numpy.savetxt('binarized_mnist_valid.amat', self.valid_mock)
        numpy.savetxt('binarized_mnist_test.amat', self.test_mock)

    def tearDown(self):
        os.remove('binarized_mnist_train.amat')
        os.remove('binarized_mnist_valid.amat')
        os.remove('binarized_mnist_test.amat')
        os.remove('mock_binarized_mnist.hdf5')

    def test_converter(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        subparser = subparsers.add_parser('binarized_mnist')
        subparser.set_defaults(
            directory='./', output_file='mock_binarized_mnist.hdf5')
        binarized_mnist.fill_subparser(subparser)
        args = parser.parse_args(['binarized_mnist'])
        args.func(args)
        h5file = h5py.File('mock_binarized_mnist.hdf5', mode='r')
        assert_equal(h5file['features'][...],
                     numpy.vstack([self.train_mock, self.valid_mock,
                                   self.test_mock]).reshape((-1, 1, 28, 28)))
        assert_equal(str(h5file['features'].dtype), 'uint8')
        assert_equal(tuple(dim.label for dim in h5file['features'].dims),
                     ('batch', 'channel', 'height', 'width'))


class TestCIFAR10(object):
    def setUp(self):
        numpy.random.seed(9 + 5 + 2015)
        self.train_features_mock = [
            numpy.random.randint(0, 256, (10, 3, 32, 32)).astype('uint8')
            for i in range(5)]
        self.train_targets_mock = [
            numpy.random.randint(0, 10, (10,)).astype('uint8')
            for i in range(5)]
        self.test_features_mock = numpy.random.randint(
            0, 256, (10, 3, 32, 32)).astype('uint8')
        self.test_targets_mock = numpy.random.randint(
            0, 10, (10,)).astype('uint8')
        os.mkdir('cifar-10-batches-py')
        for i, (x, y) in enumerate(zip(self.train_features_mock,
                                       self.train_targets_mock)):
            filename = 'cifar-10-batches-py/data_batch_{}'.format(i + 1)
            with open(filename, 'wb') as f:
                cPickle.dump({'data': x, 'labels': y}, f)
        with open('cifar-10-batches-py/test_batch', 'wb') as f:
            cPickle.dump({'data': self.test_features_mock,
                          'labels': self.test_targets_mock},
                         f)
        tar_file = tarfile.open('cifar-10-python.tar.gz', 'w:gz')
        tar_file.add('cifar-10-batches-py')
        tar_file.close()

    def tearDown(self):
        for i in range(1, 6):
            os.remove('cifar-10-batches-py/data_batch_{}'.format(i))
        os.remove('cifar-10-batches-py/test_batch')
        os.rmdir('cifar-10-batches-py')
        os.remove('cifar-10-python.tar.gz')
        os.remove('mock_cifar10.hdf5')

    def test_converter(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        subparser = subparsers.add_parser('cifar10')
        subparser.set_defaults(directory='./', output_file='mock_cifar10.hdf5')
        cifar10.fill_subparser(subparser)
        args = parser.parse_args(['cifar10'])
        args.func(args)
        h5file = h5py.File('mock_cifar10.hdf5', mode='r')
        assert_equal(
            h5file['features'][...],
            numpy.vstack(
                self.train_features_mock + [self.test_features_mock]))
        assert_equal(
            h5file['targets'][...],
            numpy.hstack(self.train_targets_mock + [self.test_targets_mock]))
        assert_equal(str(h5file['features'].dtype), 'uint8')
        assert_equal(str(h5file['targets'].dtype), 'uint8')
        assert_equal(tuple(dim.label for dim in h5file['features'].dims),
                     ('batch', 'channel', 'height', 'width'))
        assert_equal(h5file['targets'].dims[0].label, 'batch')
