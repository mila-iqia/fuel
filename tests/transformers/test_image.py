from collections import OrderedDict
from io import BytesIO
import numpy
from numpy.testing import assert_raises, assert_allclose, assert_equal
from PIL import Image
from picklable_itertools.extras import partition_all
from six.moves import zip
import pyximport
pyximport.install()
from fuel import config
from fuel.datasets.base import IndexableDataset, IterableDataset
from fuel.schemes import ShuffledScheme, SequentialExampleScheme
from fuel.streams import DataStream
from fuel.transformers.image import (ImagesFromBytes,
                                     MinimumImageDimensions,
                                     RandomFixedSizeCrop,
                                     RandomSpatialFlip,
                                     SamplewiseCropTransformer,
                                     FixedSizeCrop,
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


class TestRandomSpatialFlip(ImageTestingMixin):

    def setUp(self):

        # source is list of np.array with dim 3
        self.source_list = [
            numpy.array([[[1, 0],
                          [0, 2]],

                         [[3, 0],
                          [0, 4]]]),

            numpy.array([[[1, 2, 3],
                          [0, 0, 0]],

                         [[4, 5, 6],
                          [0, 0, 0]]])
        ]
        # source is np.object of np.array with dim 3
        self.source_ndobject = numpy.empty((2,), dtype=object)
        self.source_ndobject[0] = numpy.array([[[1, 0],
                                                [0, 2]],

                                               [[3, 0],
                                                [0, 4]]])

        self.source_ndobject[1] = numpy.array([[[1, 2, 3],
                                                [0, 0, 0]],

                                               [[4, 5, 6],
                                                [0, 0, 0]]])

        # source is np.array with dim 4
        self.source_ndarray = numpy.array([
            [[[1, 2, 3],
              [0, 0, 0]],

             [[4, 5, 6],
              [0, 0, 0]]],

            [[[1, 2, 3],
              [0, 0, 0]],

             [[4, 5, 6],
              [0, 0, 0]]]
        ])

        self.dataset = IndexableDataset(OrderedDict([
                ('source_list', self.source_list),
                ('source_ndobject', self.source_ndobject),
                ('source_ndarray', self.source_ndarray)
            ]))

        self.common_setup()

    def test_list_batch_source(self):

        source = self.source_list
        source_name = 'source_list'

        seed = 10
        # to_flip_h = [1, 0]
        # to_flip_h = [1, 1]

        # test no flip
        rng = numpy.random.RandomState(seed=seed)
        stream = RandomSpatialFlip(self.example_stream,
                                   which_sources=(source_name,),
                                   rng=rng)
        result = stream.transform_source_batch(source, source_name)
        expected = source
        all(assert_allclose(ex_result, ex_expected, err_msg="Mismatch no flip")
            for ex_result, ex_expected in zip(result, expected))

        # test flip horizontally
        rng = numpy.random.RandomState(seed=seed)
        stream = RandomSpatialFlip(self.example_stream,
                                   flip_h=True,
                                   which_sources=(source_name,),
                                   rng=rng)
        result = stream.transform_source_batch(source, source_name)
        expected = [
            numpy.array([[[0, 1],
                          [2, 0]],

                         [[0, 3],
                          [4, 0]]]),

            numpy.array([[[1, 2, 3],
                          [0, 0, 0]],

                         [[4, 5, 6],
                          [0, 0, 0]]])
        ]
        all(assert_allclose(ex_result, ex_expected,
                            err_msg="Mismatch flip horizontally")
            for ex_result, ex_expected in zip(result, expected))

        # test flip vertically
        rng = numpy.random.RandomState(seed=seed)
        rng.binomial(n=1, p=0.5, size=len(source))  # simulate first rng call
        stream = RandomSpatialFlip(self.example_stream,
                                   flip_v=True,
                                   which_sources=(source_name,),
                                   rng=rng)
        result = stream.transform_source_batch(source, source_name)
        expected = [
            numpy.array([[[0, 2],
                          [1, 0]],

                         [[0, 4],
                          [3, 0]]]),

            numpy.array([[[0, 0, 0],
                          [1, 2, 3]],

                         [[0, 0, 0],
                          [4, 5, 6]]])
        ]
        all(assert_allclose(ex_result, ex_expected,
                            err_msg="Mismatch flip vertically")
            for ex_result, ex_expected in zip(result, expected))

        # test flip both
        rng = numpy.random.RandomState(seed=seed)
        stream = RandomSpatialFlip(self.example_stream,
                                   flip_h=True, flip_v=True,
                                   which_sources=(source_name,),
                                   rng=rng)
        result = stream.transform_source_batch(source, source_name)
        expected = [
            numpy.array([[[2, 0],
                          [0, 1]],

                         [[4, 0],
                          [0, 3]]]),

            numpy.array([[[0, 0, 0],
                          [1, 2, 3]],

                         [[0, 0, 0],
                          [4, 5, 6]]])
        ]
        all(assert_allclose(ex_result, ex_expected,
                            err_msg="Mismatch flip both")
            for ex_result, ex_expected in zip(result, expected))

    def test_ndobject_batch_source(self):

        source = self.source_ndobject
        source_name = 'source_ndobject'
        expected = numpy.empty((2,), dtype=object)

        seed = 10
        # to_flip_h = [1, 0]
        # to_flip_v = [1, 1]

        # test no flip
        rng = numpy.random.RandomState(seed=seed)
        stream = RandomSpatialFlip(self.example_stream,
                                   which_sources=(source_name,),
                                   rng=rng)
        result = stream.transform_source_batch(source, source_name)
        expected = numpy.copy(source)
        all(assert_allclose(ex_result, ex_expected, err_msg="Mismatch no flip")
            for ex_result, ex_expected in zip(result, expected))

        # test flip horizontally
        rng = numpy.random.RandomState(seed=seed)
        stream = RandomSpatialFlip(self.example_stream,
                                   flip_h=True,
                                   which_sources=(source_name,),
                                   rng=rng)
        result = stream.transform_source_batch(source, source_name)
        expected[0] = numpy.array([[[0, 1],
                                    [2, 0]],

                                   [[0, 3],
                                    [4, 0]]])
        expected[1] = numpy.array([[[1, 2, 3],
                                    [0, 0, 0]],

                                   [[4, 5, 6],
                                    [0, 0, 0]]])

        all(assert_allclose(ex_result, ex_expected,
                            err_msg="Mismatch flip horizontally")
            for ex_result, ex_expected in zip(result, expected))

        # test flip vertically
        rng = numpy.random.RandomState(seed=seed)
        rng.binomial(n=1, p=0.5, size=source.shape[0])  # force first rng call
        stream = RandomSpatialFlip(self.example_stream,
                                   flip_v=True,
                                   which_sources=(source_name,),
                                   rng=rng)
        result = stream.transform_source_batch(source, source_name)
        expected[0] = numpy.array([[[0, 2],
                                    [1, 0]],

                                   [[0, 4],
                                    [3, 0]]])
        expected[1] = numpy.array([[[0, 0, 0],
                                    [1, 2, 3]],

                                   [[0, 0, 0],
                                    [4, 5, 6]]])
        all(assert_allclose(ex_result, ex_expected,
                            err_msg="Mismatch flip vertically")
            for ex_result, ex_expected in zip(result, expected))

        # test flip both
        rng = numpy.random.RandomState(seed=seed)
        stream = RandomSpatialFlip(self.example_stream,
                                   flip_h=True, flip_v=True,
                                   which_sources=(source_name,),
                                   rng=rng)
        result = stream.transform_source_batch(source, source_name)
        expected[0] = numpy.array([[[2, 0],
                                    [0, 1]],

                                   [[4, 0],
                                    [0, 3]]])
        expected[1] = numpy.array([[[0, 0, 0],
                                    [1, 2, 3]],

                                   [[0, 0, 0],
                                    [4, 5, 6]]])

        all(assert_allclose(ex_result, ex_expected,
                            err_msg="Mismatch flip both")
            for ex_result, ex_expected in zip(result, expected))

    def test_ndarray_batch_source(self):

        source = self.source_ndarray
        source_name = 'source_ndarray'

        seed = 10
        # to_flip_h = [1, 0]
        # to_flip_h = [1, 1]

        # test no flip
        rng = numpy.random.RandomState(seed=seed)
        stream = RandomSpatialFlip(self.example_stream,
                                   which_sources=(source_name,),
                                   rng=rng)
        result = stream.transform_source_batch(source, source_name)
        expected = source
        assert_allclose(result, expected, err_msg="Mismatch no flip")

        # test flip horizontally
        rng = numpy.random.RandomState(seed=seed)
        stream = RandomSpatialFlip(self.example_stream,
                                   flip_h=True,
                                   which_sources=(source_name,),
                                   rng=rng)
        result = stream.transform_source_batch(source, source_name)
        expected = numpy.array([
            [[[3, 2, 1],
              [0, 0, 0]],

             [[6, 5, 4],
              [0, 0, 0]]],

            [[[1, 2, 3],
              [0, 0, 0]],

             [[4, 5, 6],
              [0, 0, 0]]]
        ])
        assert_allclose(result, expected, err_msg="Mismatch flip horizontally")

        # test flip vertically
        rng = numpy.random.RandomState(seed=seed)
        rng.binomial(n=1, p=0.5, size=source.shape[0])  # force first rng call
        stream = RandomSpatialFlip(self.example_stream,
                                   flip_v=True,
                                   which_sources=(source_name,),
                                   rng=rng)
        result = stream.transform_source_batch(source, source_name)
        expected = numpy.array([
            [[[0, 0, 0],
              [1, 2, 3]],

             [[0, 0, 0],
              [4, 5, 6]]],

            [[[0, 0, 0],
              [1, 2, 3]],

             [[0, 0, 0],
              [4, 5, 6]]]
        ])
        assert_allclose(result, expected, err_msg="Mismatch flip vertically")

        # test flip both
        rng = numpy.random.RandomState(seed=seed)
        stream = RandomSpatialFlip(self.example_stream,
                                   flip_h=True, flip_v=True,
                                   which_sources=(source_name,),
                                   rng=rng)
        result = stream.transform_source_batch(source, source_name)
        expected = numpy.array([
            [[[0, 0, 0],
              [3, 2, 1]],

             [[0, 0, 0],
              [6, 5, 4]]],

            [[[0, 0, 0],
              [1, 2, 3]],

             [[0, 0, 0],
              [4, 5, 6]]]
        ])
        assert_allclose(result, expected, err_msg="Mismatch flip both")


class TestSamplewiseCropTransformer(object):
    def setUp(self):
        self.sources = ['volume1', 'volume2', 'weight']
        self.weight_source = 'weight'
        self.shape = (5, 5, 5)
        self.window_shape = (2, 2, 2)

        self.data_volume1 = [0 for x in range(10)]
        self.data_volume2 = [0 for x in range(10)]
        self.data_weight = [0 for x in range(10)]

        for k in range(10):
            self.data_volume1[k] = numpy.arange(numpy.prod(self.shape))\
                .reshape([1, 1] + list(self.shape)).astype(numpy.float32)
            self.data_volume2[k] = numpy.arange(numpy.prod(self.shape))\
                .reshape([1, 1] + list(self.shape)).astype(numpy.float32)
            self.data_weight[k] = numpy.random.uniform(size=self.shape)\
                .reshape([1, 1] + list(self.shape)).astype(numpy.float32)

        self.data = OrderedDict([('volume1', self.data_volume1),
                                 ('volume2', self.data_volume2),
                                 ('weight', self.data_weight)])

        layout = ('batch', 'channel', 'x', 'y', 'z')
        self.axis_labels = {self.sources[0]: layout,
                            self.sources[1]: layout,
                            self.sources[2]: layout}

        self.stream = DataStream(IterableDataset(self.data),
                                 axis_labels=self.axis_labels)
        self.stream.produces_examples = False

    def test_no_weight_crop(self):
        swcTransformer = SamplewiseCropTransformer(self.stream,
                                                   self.window_shape,
                                                   which_sources=None,
                                                   weight_source=None)
        epoch_iterat = swcTransformer.get_epoch_iterator()

        former_crop = []
        different = []
        for k in range(5):
            n = next(epoch_iterat)
            # Test if new array size is compliant to window_shape
            for k in range(3):
                assert n[k].shape[2:] == self.window_shape
            # Test if the crop is similar on all volumes of the sample
            assert numpy.prod(n[0] == n[1])
            # Test if random crop different from former random crop
            different.append(former_crop == n[0])
            former_crop = n[0]

    def test_no_weight_which_sources(self):
        swcTransformer = SamplewiseCropTransformer(self.stream,
                                                   self.window_shape,
                                                   which_sources=['volume1'],
                                                   weight_source=None)
        epoch_iterat = swcTransformer.get_epoch_iterator()
        for k in range(5):
            n = next(epoch_iterat)
            # Test if only sources directed by "which_sources" are affected
            assert n[0].shape == (1, 1) + self.window_shape
            assert n[1].shape == (1, 1) + self.shape
            assert n[2].shape == (1, 1) + self.shape

    def test_weight_crop(self):
        swcTransformer = SamplewiseCropTransformer(self.stream,
                                                   self.window_shape,
                                                   which_sources=None,
                                                   weight_source='weight')
        epoch_iterat = swcTransformer.get_epoch_iterator()
        for k in range(5):
            n = next(epoch_iterat)
            # Test if the output was actually upscaled from a factor p
            # The only way to do this is to look at every possible position
            # of the resulting cropped heatmap in the original heatmap and
            # assert they are never equal (because of the randomness of the
            # crop)
            self.search_volume(self.data_weight[k], n[2])

    def search_volume(self, big, small):
        for stx in range(big.shape[2]-small.shape[2]):
            for sty in range(big.shape[3]-small.shape[3]):
                for stz in range(big.shape[4]-small.shape[4]):
                    assert not numpy.allclose(small,
                                              big[:, :,
                                              stx:stx+small.shape[2],
                                              sty:sty+small.shape[3],
                                              stz:stz+small.shape[4]])

    def test_calculate_heatmap(self):
        swcTransformer = SamplewiseCropTransformer(self.stream,
                                                   self.window_shape,
                                                   which_sources=None,
                                                   weight_source=None)
        volume = numpy.arange(5 * 5 * 5).reshape((1, 1) + (5, 5, 5)).astype(
            numpy.float32)
        new_volume = swcTransformer.calculate_heatmap(volume)

        assert numpy.allclose(new_volume,
                              volume / volume[:, :, 2:-1, 2:-1, 2:-1].sum())


class TestFixedSizeCrop(ImageTestingMixin):
    def setUp(self):
        source1 = numpy.zeros((9, 3, 7, 5), dtype='uint8')
        source1[:] = numpy.arange(3 * 7 * 5, dtype='uint8').reshape(3, 7, 5)
        shapes = [(5, 8), (6, 8), (5, 6), (5, 5), (6, 4), (7, 4),
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
        source3 = numpy.empty((len(shapes),), dtype=object)
        for i in range(len(source2)):
            source3[i] = source2[i]
        axis_labels = {'source1': ('batch', 'channel', 'height', 'width'),
                       'source2': ('batch', 'channel', 'height', 'width'),
                       'source3': ('batch', 'channel', 'height', 'width')}
        self.dataset = IndexableDataset(OrderedDict([('source1', source1),
                                                     ('source2', source2),
                                                     ('source3', source3)]),
                                        axis_labels=axis_labels)
        self.common_setup()

    def test_ndarray_batch_source(self):
        # Make sure that with 4 corner crops we sample everything.
        seen_indices = numpy.array([], dtype='uint8')
        for loc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            stream = FixedSizeCrop(self.batch_stream, (5, 4),
                                   which_sources=('source1',), location=loc)
            batch = stream.get_epoch_iterator().next()
            assert batch[0].shape[1:] == (3, 5, 4)
            assert batch[0].shape[0] in (1, 2)
            # seen indices should only be of that length in after last location
            if 3 * 7 * 5 == len(seen_indices):
                assert False
            seen_indices = numpy.union1d(seen_indices, batch[0].flatten())
        assert 3 * 7 * 5 == len(seen_indices)

    def test_list_batch_source(self):
        # Make sure that with 4 corner crops we sample everything.
        seen_indices = numpy.array([], dtype='uint8')

        for loc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            stream = FixedSizeCrop(self.batch_stream, (5, 4),
                                   which_sources=('source2',), location=loc)
            # seen indices should only be of that length in after last location
            if self.source2_biggest == len(seen_indices):
                assert False
            for batch in stream.get_epoch_iterator():
                for example in batch[1]:
                    assert example.shape == (2, 5, 4)
                    seen_indices = numpy.union1d(seen_indices,
                                                 example.flatten())
        assert self.source2_biggest == len(seen_indices)

    def test_objectarray_batch_source(self):
        # Make sure that with 4 corner crops we sample everything.
        seen_indices = numpy.array([], dtype='uint8')

        for loc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            stream = FixedSizeCrop(self.batch_stream, (5, 4),
                                   which_sources=('source3',), location=loc)
            # seen indices should only be of that length in after last location
            if self.source2_biggest == len(seen_indices):
                assert False
            for batch in stream.get_epoch_iterator():
                for example in batch[2]:
                    print example.shape
                    assert example.shape == (2, 5, 4)
                    seen_indices = numpy.union1d(seen_indices,
                                                 example.flatten())
        assert self.source2_biggest == len(seen_indices)

    def test_wrong_location_exceptions(self):
        assert_raises(ValueError, FixedSizeCrop, self.example_stream, (5, 4),
                      which_sources=('source2',), location=1)
        assert_raises(ValueError, FixedSizeCrop, self.example_stream, (5, 4),
                      which_sources=('source2',), location=[0, 1, 0])
        assert_raises(ValueError, FixedSizeCrop, self.example_stream, (5, 4),
                      which_sources=('source2',), location=[2, 0])

    def test_format_exceptions(self):
        estream = FixedSizeCrop(self.example_stream, (5, 4),
                                which_sources=('source2',), location=[0, 0])
        bstream = FixedSizeCrop(self.batch_stream, (5, 4),
                                which_sources=('source2',), location=[0, 0])
        assert_raises(ValueError, estream.transform_source_example,
                      numpy.empty((5, 6)), 'source2')
        assert_raises(ValueError, bstream.transform_source_batch,
                      [numpy.empty((7, 6))], 'source2')
        assert_raises(ValueError, bstream.transform_source_batch,
                      [numpy.empty((8, 6))], 'source2')

    def test_window_too_big_exceptions(self):
        stream = FixedSizeCrop(self.example_stream, (5, 4),
                               which_sources=('source2',), location=[0, 0])

        assert_raises(ValueError, stream.transform_source_example,
                      numpy.empty((3, 4, 2)), 'source2')

        bstream = FixedSizeCrop(self.batch_stream, (5, 4),
                                which_sources=('source1',), location=[0, 0])

        assert_raises(ValueError, bstream.transform_source_batch,
                      numpy.empty((5, 3, 4, 2)), 'source1')

class TestRandom2DRotation(ImageTestingMixin):
    def setUp(self):
        source1 = numpy.zeros((2, 3, 4, 5), dtype='uint8')
        source1[:] = numpy.arange(3 * 4 * 5, dtype='uint8').reshape(3, 4, 5)

        source2 = numpy.empty(2, dtype=object)
        source2[0] = numpy.arange(3 * 4 * 5, dtype='uint8').reshape(3, 4, 5)
        source2[1] = numpy.arange(3 * 4 * 6, dtype='uint8').reshape(3, 4, 6)

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

    def test_random_2D_rotation_example_stream(self):
        maximum_rotation = 0.5
        rng = numpy.random.RandomState(123)
        estream = Random2DRotation(self.example_stream,
                                   maximum_rotation,
                                   rng=rng,
                                   which_sources=('source1',))
        # the C x X x Y image should have equal rotation for all c in C
        out = estream.transform_source_example(self.source1[0], 'source1')
        expected = numpy.array([[[0,  0,  2,  4,   0],
                                 [3,  5,  7,  9,   0],
                                 [0, 10, 12, 14,  16],
                                 [0, 15, 17,  0,   0]],
                                [[0,   0, 22, 24,  0],
                                 [23, 25, 27, 29,  0],
                                 [0,  30, 32, 34, 36],
                                 [0,  35, 37,  0,  0]],
                                [[0,   0, 42, 44,  0],
                                 [43, 45, 47, 49,  0],
                                 [0,  50, 52, 54, 56],
                                 [0,  55, 57,  0,  0]]], dtype=out.dtype)
        assert_equal(out, expected)

    def test_random_2D_rotation_batch_stream(self):
        rng = numpy.random.RandomState(123)
        bstream = Random2DRotation(self.batch_stream,
                                   maximum_rotation=0.5,
                                   rng=rng,
                                   which_sources=('source1',))
        # the C x X x Y image should have equal rotation for all c in C
        out = bstream.transform_source_batch(self.source1, 'source1')
        expected = numpy.array([[[[0,  0,  2,  4,   0],
                                  [3,  5,  7,  9,   0],
                                  [0, 10, 12, 14,  16],
                                  [0, 15, 17,  0,   0]],
                                 [[0,   0, 22, 24,  0],
                                  [23, 25, 27, 29,  0],
                                  [0,  30, 32, 34, 36],
                                  [0,  35, 37,  0,  0]],
                                 [[0,   0, 42, 44,  0],
                                  [43, 45, 47, 49,  0],
                                  [0,  50, 52, 54, 56],
                                  [0,  55, 57,  0,  0]]],
                                [[[0,   1,  2,  0,  0],
                                  [0,   7,  7,  7,  6],
                                  [13, 12, 12, 12,  0],
                                  [0,   0, 17, 18,  0]],
                                 [[0,  21, 22,  0,  0],
                                  [0,  27, 27, 27, 26],
                                  [33, 32, 32, 32,  0],
                                  [0,   0, 37, 38,  0]],
                                 [[0,  41, 42,  0,  0],
                                  [0,  47, 47, 47, 46],
                                  [53, 52, 52, 52,  0],
                                  [0,   0, 57, 58,  0]]]], dtype='uint8')
        assert_equal(out, expected)

        rng = numpy.random.RandomState(123)
        bstream = Random2DRotation(self.batch_stream,
                                   maximum_rotation=0.5,
                                   rng=rng,
                                   which_sources=('source2',))
        out = bstream.transform_source_batch(self.source2, 'source2')
        expected = \
            [numpy.array([[[0,   0,  2,  4,  0],
                           [3,   5,  7,  9,  0],
                           [0,  10, 12, 14, 16],
                           [0,  15, 17,  0,  0]],
                          [[0,   0, 22, 24,  0],
                           [23, 25, 27, 29,  0],
                           [0,  30, 32, 34, 36],
                           [0,  35, 37,  0,  0]],
                          [[0,   0, 42, 44,  0],
                           [43, 45, 47, 49,  0],
                           [0,  50, 52, 54, 56],
                           [0,  55, 57,  0,  0]]], dtype='uint8'),
             numpy.array([[[0,   2,  2,  0,  0,  0],
                           [0,   9,  9,  8,  7,  7],
                           [16, 16, 15, 14, 14,  0],
                           [0,   0,  0, 21, 21,  0]],
                          [[0,  26, 26,  0,  0,  0],
                           [0,  33, 33, 32, 31, 31],
                           [40, 40, 39, 38, 38,  0],
                           [0,   0,  0, 45, 45,  0]],
                          [[0,  50, 50,  0,  0,  0],
                           [0,  57, 57, 56, 55, 55],
                           [64, 64, 63, 62, 62,  0],
                           [0,  0,  0, 69, 69,  0]]], dtype='uint8')]
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
