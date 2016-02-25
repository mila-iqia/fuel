from collections import OrderedDict
from io import BytesIO
import numpy
from numpy.testing import assert_raises, assert_equal
from PIL import Image
from picklable_itertools.extras import partition_all
from six.moves import zip
from fuel import config
from fuel.datasets.base import IndexableDataset
from fuel.schemes import ShuffledScheme, SequentialExampleScheme
from fuel.streams import DataStream
from fuel.transformers.image import (ImagesFromBytes,
                                     MinimumImageDimensions,
                                     RandomFixedSizeCrop,
                                     Random2DRotation)


def reorder_axes(shp):
    if len(shp) == 3:
        shp = (shp[-1],) + shp[:-1]
    elif len(shp) == 2:
        shp = (1,) + shp
    return shp


class ImageTestingMixin(object):
    def common_setup(self):
        ex_scheme = SequentialExampleScheme(self.dataset.num_examples)
        self.example_stream = DataStream(self.dataset,
                                         iteration_scheme=ex_scheme)
        self.batch_size = 2
        scheme = ShuffledScheme(self.dataset.num_examples,
                                batch_size=self.batch_size)
        self.batch_stream = DataStream(self.dataset, iteration_scheme=scheme)


class TestImagesFromBytes(ImageTestingMixin):
    def setUp(self):
        rng = numpy.random.RandomState(config.default_seed)
        self.shapes = [
            (10, 12, 3),
            (9, 8, 4),
            (12, 14, 3),
            (4, 7),
            (9, 8, 4),
            (7, 9, 3)
        ]
        pil1 = Image.fromarray(rng.random_integers(0, 255,
                                                   size=self.shapes[0])
                               .astype('uint8'), mode='RGB')
        pil2 = Image.fromarray(rng.random_integers(0, 255,
                                                   size=self.shapes[1])
                               .astype('uint8'), mode='CMYK')
        pil3 = Image.fromarray(rng.random_integers(0, 255,
                                                   size=self.shapes[2])
                               .astype('uint8'), mode='RGB')
        pil4 = Image.fromarray(rng.random_integers(0, 255,
                                                   size=self.shapes[3])
                               .astype('uint8'), mode='L')
        pil5 = Image.fromarray(rng.random_integers(0, 255,
                                                   size=self.shapes[4])
                               .astype('uint8'), mode='RGBA')
        pil6 = Image.fromarray(rng.random_integers(0, 255,
                                                   size=self.shapes[5])
                               .astype('uint8'), mode='YCbCr')
        source1 = [pil1, pil2, pil3]
        source2 = [pil4, pil5, pil6]
        bytesio1 = [BytesIO() for _ in range(3)]
        bytesio2 = [BytesIO() for _ in range(3)]
        formats1 = ['PNG', 'JPEG', 'BMP']
        formats2 = ['GIF', 'PNG', 'JPEG']
        for s, b, f in zip(source1, bytesio1, formats1):
            s.save(b, format=f)
        for s, b, f in zip(source2, bytesio2, formats2):
            s.save(b, format=f)
        self.dataset = IndexableDataset(
            OrderedDict([('source1', [b.getvalue() for b in bytesio1]),
                         ('source2', [b.getvalue() for b in bytesio2])]),
            axis_labels={'source1': ('batch', 'bytes'),
                         'source2': ('batch', 'bytes')})
        self.common_setup()

    def test_images_from_bytes_example_stream(self):
        stream = ImagesFromBytes(self.example_stream,
                                 which_sources=('source1', 'source2'),
                                 color_mode=None)
        s1, s2 = list(zip(*list(stream.get_epoch_iterator())))
        s1_shape = set(s.shape for s in s1)
        s2_shape = set(s.shape for s in s2)
        actual_s1 = set(reorder_axes(s) for s in self.shapes[:3])
        actual_s2 = set(reorder_axes(s) for s in self.shapes[3:])
        assert actual_s1 == s1_shape
        assert actual_s2 == s2_shape

    def test_images_from_bytes_batch_stream(self):
        stream = ImagesFromBytes(self.batch_stream,
                                 which_sources=('source1', 'source2'),
                                 color_mode=None)
        s1, s2 = list(zip(*list(stream.get_epoch_iterator())))
        s1 = sum(s1, [])
        s2 = sum(s2, [])
        s1_shape = set(s.shape for s in s1)
        s2_shape = set(s.shape for s in s2)
        actual_s1 = set(reorder_axes(s) for s in self.shapes[:3])
        actual_s2 = set(reorder_axes(s) for s in self.shapes[3:])
        assert actual_s1 == s1_shape
        assert actual_s2 == s2_shape

    def test_images_from_bytes_example_stream_convert_rgb(self):
        stream = ImagesFromBytes(self.example_stream,
                                 which_sources=('source1'),
                                 color_mode='RGB')
        s1, s2 = list(zip(*list(stream.get_epoch_iterator())))
        actual_s1_gen = (reorder_axes(s) for s in self.shapes[:3])
        actual_s1 = set((3,) + s[1:] for s in actual_s1_gen)
        s1_shape = set(s.shape for s in s1)
        assert actual_s1 == s1_shape

    def test_images_from_bytes_example_stream_convert_l(self):
        stream = ImagesFromBytes(self.example_stream,
                                 which_sources=('source2'),
                                 color_mode='L')
        s1, s2 = list(zip(*list(stream.get_epoch_iterator())))
        actual_s2_gen = (reorder_axes(s) for s in self.shapes[3:])
        actual_s2 = set((1,) + s[1:] for s in actual_s2_gen)
        s2_shape = set(s.shape for s in s2)
        assert actual_s2 == s2_shape

    def test_axis_labels(self):
        stream = ImagesFromBytes(self.example_stream,
                                 which_sources=('source2',))
        assert stream.axis_labels['source1'] == ('bytes',)
        assert stream.axis_labels['source2'] == ('channel', 'height',
                                                 'width')
        bstream = ImagesFromBytes(self.batch_stream,
                                  which_sources=('source1',))
        assert bstream.axis_labels['source1'] == ('batch', 'channel', 'height',
                                                  'width')
        assert bstream.axis_labels['source2'] == ('batch', 'bytes')

    def test_bytes_type_exception(self):
        stream = ImagesFromBytes(self.example_stream,
                                 which_sources=('source2',))
        assert_raises(TypeError, stream.transform_source_example, 54321,
                      'source2')


class TestMinimumDimensions(ImageTestingMixin):
    def setUp(self):
        rng = numpy.random.RandomState(config.default_seed)
        source1 = []
        source2 = []
        source3 = []
        self.shapes = [(5, 9), (4, 6), (4, 3), (6, 4), (2, 5), (4, 8), (8, 3)]
        for i, shape in enumerate(self.shapes):
            source1.append(rng.normal(size=shape))
            source2.append(rng.normal(size=shape[::-1]))
            source3.append(rng.random_integers(0, 255, size=(3,) + shape)
                           .astype('uint8'))
        self.dataset = IndexableDataset(OrderedDict([('source1', source1),
                                                     ('source2', source2),
                                                     ('source3', source3)]),
                                        axis_labels={'source1':
                                                     ('batch', 'channel',
                                                      'height', 'width'),
                                                     'source3':
                                                     ('batch', 'channel',
                                                      'height', 'width')})
        self.common_setup()

    def test_minimum_dimensions_example_stream(self):
        stream = MinimumImageDimensions(self.example_stream, (4, 5),
                                        which_sources=('source1',
                                                       'source3'))
        it = stream.get_epoch_iterator()
        for example, shp in zip(it, self.shapes):
            assert example[0].shape[0] >= 4 and example[0].shape[1] >= 5
            assert (example[1].shape[1] == shp[0] and
                    example[1].shape[0] == shp[1])
            assert example[2].shape[0] == 3
            assert example[2].shape[1] >= 4 and example[2].shape[2] >= 5

    def test_minimum_dimensions_batch_stream(self):
        stream = MinimumImageDimensions(self.batch_stream, (4, 5),
                                        which_sources=('source1',))
        it = stream.get_epoch_iterator()
        for batch, shapes in zip(it, partition_all(self.batch_size,
                                                   self.shapes)):
            assert (example.shape[0] >= 4 and example.shape[1] >= 5
                    for example in batch[0])
            assert (example.shape[1] == shp[0] and
                    example.shape[0] == shp[1]
                    for example, shp in zip(batch[1], shapes))

    def test_axes_exception(self):
        stream = MinimumImageDimensions(self.example_stream, (4, 5),
                                        which_sources=('source1',))
        assert_raises(NotImplementedError,
                      stream.transform_source_example,
                      numpy.empty((2, 3, 4, 2)),
                      'source1')

    def test_resample_exception(self):
        assert_raises(ValueError,
                      MinimumImageDimensions, self.example_stream, (4, 5),
                      resample='notarealresamplingmode')


class TestFixedSizeRandomCrop(ImageTestingMixin):
    def setUp(self):
        source1 = numpy.zeros((9, 3, 7, 5), dtype='uint8')
        source1[:] = numpy.arange(3 * 7 * 5, dtype='uint8').reshape(3, 7, 5)
        shapes = [(5, 9), (6, 8), (5, 6), (5, 5), (6, 4), (7, 4),
                  (9, 4), (5, 6), (6, 5)]
        source2 = []
        biggest = 0
        num_channels = 2
        for shp in shapes:
            biggest = max(biggest, shp[0] * shp[1] * 2)
            ex = numpy.arange(shp[0] * shp[1] * num_channels).reshape(
                (num_channels,) + shp).astype('uint8')
            source2.append(ex)
        self.source2_biggest = biggest
        axis_labels = {'source1': ('batch', 'channel', 'height', 'width'),
                       'source2': ('batch', 'channel', 'height', 'width')}
        self.dataset = IndexableDataset(OrderedDict([('source1', source1),
                                                     ('source2', source2)]),
                                        axis_labels=axis_labels)
        self.common_setup()

    def test_ndarray_batch_source(self):
        # Make sure that with enough epochs we sample everything.
        stream = RandomFixedSizeCrop(self.batch_stream, (5, 4),
                                     which_sources=('source1',))
        seen_indices = numpy.array([], dtype='uint8')
        for i in range(30):
            for batch in stream.get_epoch_iterator():
                assert batch[0].shape[1:] == (3, 5, 4)
                assert batch[0].shape[0] in (1, 2)
                seen_indices = numpy.union1d(seen_indices, batch[0].flatten())
            if 3 * 7 * 5 == len(seen_indices):
                break
        else:
            assert False

    def test_list_batch_source(self):
        # Make sure that with enough epochs we sample everything.
        stream = RandomFixedSizeCrop(self.batch_stream, (5, 4),
                                     which_sources=('source2',))
        seen_indices = numpy.array([], dtype='uint8')
        for i in range(30):
            for batch in stream.get_epoch_iterator():
                for example in batch[1]:
                    assert example.shape == (2, 5, 4)
                    seen_indices = numpy.union1d(seen_indices,
                                                 example.flatten())
                assert len(batch[1]) in (1, 2)
            if self.source2_biggest == len(seen_indices):
                break
        else:
            assert False

    def test_format_exceptions(self):
        estream = RandomFixedSizeCrop(self.example_stream, (5, 4),
                                      which_sources=('source2',))
        bstream = RandomFixedSizeCrop(self.batch_stream, (5, 4),
                                      which_sources=('source2',))
        assert_raises(ValueError, estream.transform_source_example,
                      numpy.empty((5, 6)), 'source2')
        assert_raises(ValueError, bstream.transform_source_batch,
                      [numpy.empty((7, 6))], 'source2')
        assert_raises(ValueError, bstream.transform_source_batch,
                      [numpy.empty((8, 6))], 'source2')

    def test_window_too_big_exceptions(self):
        stream = RandomFixedSizeCrop(self.example_stream, (5, 4),
                                     which_sources=('source2',))

        assert_raises(ValueError, stream.transform_source_example,
                      numpy.empty((3, 4, 2)), 'source2')

        bstream = RandomFixedSizeCrop(self.batch_stream, (5, 4),
                                      which_sources=('source1',))

        assert_raises(ValueError, bstream.transform_source_batch,
                      numpy.empty((5, 3, 4, 2)), 'source1')


class TestRandom2DRotation(ImageTestingMixin):
    def setUp(self):
        source1 = numpy.zeros((2, 3, 4, 5), dtype='uint8')
        source1[:] = numpy.arange(3 * 4 * 5, dtype='uint8').reshape((3, 4, 5))

        source2 = numpy.empty(2, dtype=object)
        source2[0] = numpy.arange(3 * 4 * 5, dtype='uint8').reshape((3, 4, 5))
        source2[1] = numpy.arange(3 * 4 * 6, dtype='uint8').reshape((3, 4, 6))

        source3 = [source2[0], source2[1]]

        self.source1 = source1
        self.source2 = source2
        self.source3 = source3

        axis_labels = {'source1': ('batch', 'channel', 'height', 'width'),
                       'source2': ('batch', 'channel', 'height', 'width'),
                       'source3': ('batch', 'channel', 'height', 'width')}
        self.dataset = \
            IndexableDataset(OrderedDict([('source1', source1),
                                          ('source2', source2),
                                          ('source3', source3)]),
                             axis_labels=axis_labels)
        self.common_setup()

    def test_format_exceptions(self):
        estream = Random2DRotation(self.example_stream,
                                   which_sources=('source2',))
        bstream = Random2DRotation(self.batch_stream,
                                   which_sources=('source2',))
        assert_raises(ValueError, estream.transform_source_example,
                      numpy.empty((5, 6)), 'source2')
        assert_raises(ValueError, bstream.transform_source_batch,
                      [numpy.empty((7, 6))], 'source2')
        assert_raises(ValueError, bstream.transform_source_batch,
                      [numpy.empty((8, 6))], 'source2')

    def test_maximum_rotation_invalid_exception(self):
        assert_raises(ValueError, Random2DRotation, self.example_stream,
                      maximum_rotation=0.0,
                      which_sources=('source2',))
        assert_raises(ValueError, Random2DRotation, self.example_stream,
                      maximum_rotation=3.1416,
                      which_sources=('source2',))

    def test_invalid_resample_exception(self):
        assert_raises(ValueError, Random2DRotation, self.example_stream,
                      resample='nonexisting')

    def test_random_2D_rotation_example_stream(self):
        maximum_rotation = 0.5
        rng = numpy.random.RandomState(123)
        estream = Random2DRotation(self.example_stream,
                                   maximum_rotation,
                                   rng=rng,
                                   which_sources=('source1',))
        # the C x X x Y image should have equal rotation for all c in C
        out = estream.transform_source_example(self.source1[0], 'source1')
        expected = numpy.array([[[0,  0,  0,  2,  3],
                                 [0,  0,  1,  7,  8],
                                 [0,  5,  6, 12, 13],
                                 [0, 10, 11, 17, 18]],
                                [[0,  0,  0, 22, 23],
                                 [0, 20, 21, 27, 28],
                                 [0, 25, 26, 32, 33],
                                 [0, 30, 31, 37, 38]],
                                [[0,  0,  0, 42, 43],
                                 [0, 40, 41, 47, 48],
                                 [0, 45, 46, 52, 53],
                                 [0, 50, 51, 57, 58]]], dtype='uint8')
        assert_equal(out, expected)

    def test_random_2D_rotation_batch_stream(self):
        rng = numpy.random.RandomState(123)
        bstream = Random2DRotation(self.batch_stream,
                                   maximum_rotation=0.5,
                                   rng=rng,
                                   which_sources=('source1',))
        # each C x X x Y image should have equal rotation for all c in C
        out = bstream.transform_source_batch(self.source1, 'source1')
        expected = numpy.array([[[[0,  0,  0,  2,  3],
                                  [0,  0,  1,  7,  8],
                                  [0,  5,  6, 12, 13],
                                  [0, 10, 11, 17, 18]],
                                 [[0,  0,  0, 22, 23],
                                  [0, 20, 21, 27, 28],
                                  [0, 25, 26, 32, 33],
                                  [0, 30, 31, 37, 38]],
                                 [[0,  0,  0, 42, 43],
                                  [0, 40, 41, 47, 48],
                                  [0, 45, 46, 52, 53],
                                  [0, 50, 51, 57, 58]]],
                                [[[0,  0,  1,  0,  0],
                                  [0,  5,  6,  2,  3],
                                  [0, 10, 11,  7,  8],
                                  [0, 15, 16, 12, 13]],
                                 [[0, 20, 21,  0,  0],
                                  [0, 25, 26, 22, 23],
                                  [0, 30, 31, 27, 28],
                                  [0, 35, 36, 32, 33]],
                                 [[0, 40, 41,  0,  0],
                                  [0, 45, 46, 42, 43],
                                  [0, 50, 51, 47, 48],
                                  [0, 55, 56, 52, 53]]]], dtype='uint8')
        assert_equal(out, expected)

        expected = \
            [numpy.array([[[0,  0,  0,  2,   3],
                           [0,  0,  1,  7,   8],
                           [0,  5,  6,  12, 13],
                           [0,  10, 11, 17, 18]],
                          [[0,  0,  0,  22, 23],
                           [0,  20, 21, 27, 28],
                           [0,  25, 26, 32, 33],
                           [0,  30, 31, 37, 38]],
                          [[0,  0,  0,  42, 43],
                           [0,  40, 41, 47, 48],
                           [0,  45, 46, 52, 53],
                           [0,  50, 51, 57, 58]]], dtype='uint8'),
             numpy.array([[[0,  0,  1,  2,  0,   0],
                           [0,  6,  7,  8,  3,   4],
                           [12, 13, 14, 15, 9,  10],
                           [18, 19, 20, 15, 16, 17]],
                          [[0,  24, 25, 26,  0,  0],
                           [0,  30, 31, 32, 27, 28],
                           [36, 37, 38, 39, 33, 34],
                           [42, 43, 44, 39, 40, 41]],
                          [[0,  48, 49, 50,  0,  0],
                           [0,  54, 55, 56, 51, 52],
                           [60, 61, 62, 63, 57, 58],
                           [66, 67, 68, 63, 64, 65]]], dtype='uint8')]

        rng = numpy.random.RandomState(123)
        bstream = Random2DRotation(self.batch_stream,
                                   maximum_rotation=0.5,
                                   rng=rng,
                                   which_sources=('source2',))
        out = bstream.transform_source_batch(self.source2, 'source2')
        assert_equal(out[0], expected[0])
        assert_equal(out[1], expected[1])

        rng = numpy.random.RandomState(123)
        bstream = Random2DRotation(self.batch_stream,
                                   maximum_rotation=0.5,
                                   rng=rng,
                                   which_sources=('source3',))
        out = bstream.transform_source_batch(self.source3, 'source3')
        assert_equal(out[0], expected[0])
        assert_equal(out[1], expected[1])
