from __future__ import division
from io import BytesIO
import math

import numpy
import scipy.ndimage
from PIL import Image
from six import PY3

try:
    from ._image import window_batch_bchw, window_batch_bchw3d
    window_batch_bchw_available = True
except ImportError:
    window_batch_bchw_available = False

from . import ExpectsAxisLabels, SourcewiseTransformer, Transformer

from .. import config


class ImagesFromBytes(SourcewiseTransformer):
    """Load from a stream of bytes objects representing encoded images.

    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The wrapped data stream. The individual examples returned by this
        should be the bytes (in a `bytes` container, or a `str` on legacy
        Python) comprising an image in a format readable by PIL, such as
        PNG, JPEG, etc.
    color_mode : str, optional
        Mode to pass to PIL for color space conversion. Default is RGB.
        If `None`, no coercion is performed.

    Notes
    -----
    Images are returned as NumPy arrays converted from PIL objects.
    If there is more than one color channel, then the array is transposed
    from the `(height, width, channel)` dimension layout native to PIL to
    the `(channel, height, width)` layout that is pervasive in the world
    of convolutional networks. If there is only one color channel, as for
    monochrome or binary images, a leading axis with length 1 is added for
    the sake of uniformity/predictability.

    This SourcewiseTransformer supports streams returning single examples
    as `bytes` objects (`str` on legacy Python) as well as streams that
    return iterables containing such objects. In the case of an iterable, a
    list of loaded images is returned.

    """
    def __init__(self, data_stream, color_mode='RGB', **kwargs):
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        # Acrobatics currently required to correctly set axis labels.
        which_sources = kwargs.get('which_sources', data_stream.sources)
        axis_labels = self._make_axis_labels(data_stream, which_sources,
                                             kwargs['produces_examples'])
        kwargs.setdefault('axis_labels', axis_labels)
        super(ImagesFromBytes, self).__init__(data_stream, **kwargs)
        self.color_mode = color_mode

    def transform_source_example(self, example, source_name):
        if PY3:
            bytes_type = bytes
        else:
            bytes_type = str
        if not isinstance(example, bytes_type):
            raise TypeError("expected {} object".format(bytes_type.__name__))
        pil_image = Image.open(BytesIO(example))
        if self.color_mode is not None:
            pil_image = pil_image.convert(self.color_mode)
        image = numpy.array(pil_image)
        if image.ndim == 3:
            # Transpose to `(channels, height, width)` layout.
            return image.transpose(2, 0, 1)
        elif image.ndim == 2:
            # Add a channels axis of length 1.
            image = image[numpy.newaxis]
        else:
            raise ValueError('unexpected number of axes')
        return image

    def transform_source_batch(self, batch, source_name):
        return [self.transform_source_example(im, source_name) for im in batch]

    def _make_axis_labels(self, data_stream, which_sources, produces_examples):
        # This is ugly and probably deserves a refactoring of how we handle
        # axis labels. It would be simpler to use memoized read-only
        # properties, but the AbstractDataStream constructor tries to set
        # self.axis_labels currently. We can't use self.which_sources or
        # self.produces_examples here, because this *computes* things that
        # need to be passed into the superclass constructor, necessarily
        # meaning that the superclass constructor hasn't been called.
        # Cooperative inheritance is hard, etc.
        labels = {}
        for source in data_stream.sources:
            if source in which_sources:
                if produces_examples:
                    labels[source] = ('channel', 'height', 'width')
                else:
                    labels[source] = ('batch', 'channel', 'height', 'width')
            else:
                labels[source] = (data_stream.axis_labels[source]
                                  if source in data_stream.axis_labels
                                  else None)
        return labels


class MinimumImageDimensions(SourcewiseTransformer, ExpectsAxisLabels):
    """Resize (lists of) images to minimum dimensions.

    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    minimum_shape : 2-tuple
        The minimum `(height, width)` dimensions every image must have.
        Images whose height and width are larger than these dimensions
        are passed through as-is.
    resample : str, optional
        Resampling filter for PIL to use to upsample any images requiring
        it. Options include 'nearest' (default), 'bilinear', and 'bicubic'.
        See the PIL documentation for more detailed information.

    Notes
    -----
    This transformer expects stream sources returning individual images,
    represented as 2- or 3-dimensional arrays, or lists of the same.
    The format of the stream is unaltered.

    """
    def __init__(self, data_stream, minimum_shape, resample='nearest',
                 **kwargs):
        self.minimum_shape = minimum_shape
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(MinimumImageDimensions, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return self._example_transform(example, source_name)

    def _example_transform(self, example, _):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        min_height, min_width = self.minimum_shape
        original_height, original_width = example.shape[-2:]
        if original_height < min_height or original_width < min_width:
            dt = example.dtype
            # If we're dealing with a colour image, swap around the axes
            # to be in the format that PIL needs.
            if example.ndim == 3:
                im = example.transpose(1, 2, 0)
            else:
                im = example
            im = Image.fromarray(im)
            width, height = im.size
            multiplier = max(1, min_width / width, min_height / height)
            width = int(math.ceil(width * multiplier))
            height = int(math.ceil(height * multiplier))
            im = numpy.array(im.resize((width, height))).astype(dt)
            # If necessary, undo the axis swap from earlier.
            if im.ndim == 3:
                example = im.transpose(2, 0, 1)
            else:
                example = im
        return example


class SamplewiseCropTransformer(Transformer):
    """Applies same transformation to all data from get_epoch ("batchwise").

    Subclasses must define `transform_source_example` (to transform
    examples), `transform_source_batch` (to transform batches) or
    both.

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    window_shape: tuple or list of int
        Shape of the crop, for example (30, 30, 30)
    which_sources : tuple of str, optional
        Which sources to apply the mapping to. Defaults to `None`, in
        which case the mapping is applied to all sources.
    weight_source: str
        Name of the source that is to be used to scale and influence the
        random cropping.
    """
    def __init__(self, data_stream, window_shape,
                 which_sources=None, weight_source=None, **kwargs):
        self.window_shape = window_shape
        self.rng = kwargs.pop('rng', None)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        self.weight_source = weight_source
        if which_sources is None:
            which_sources = data_stream.sources
        self.which_sources = which_sources
        super(SamplewiseCropTransformer, self).__init__(
            data_stream, **kwargs)
        if self.weight_source is not None:
            for i, source_name in enumerate(self.data_stream.sources):
                if source_name == self.weight_source:
                    self.weight_i = i
        self.data = None

    def _apply_samplewise_transformation(self, data, method):
        data = list(data)
        # If weight_source is given, we want to crop according to this weight
        #  matrix (or heatmap here). Depending on the dataset, the sources (
        # which are the different data[i] are either a numpy object or a
        # numpy array, which require different processing due to their size
        # and shape but the concept for both is the same. We compute the
        # global heatmap before cropping, which is a scaled up version of
        # the weight matrix where we compute the "probabilities" p of the
        # cubes near the center of the weight matrix (which are the most
        # probable in our specific case of bone segmentation as bone is
        # usually at the center of the volume), take the max(p) as reference
        #  to scale up the weight matrix.
        if self.weight_source is not None:
            if data[self.weight_i].dtype == numpy.object:
                heatmap = self.calculate_heatmap(
                    data[self.weight_i][0].reshape(
                        [1] + list(data[self.weight_i][0].shape)))
            else:
                heatmap = self.calculate_heatmap(data[self.weight_i])
            while True:
                # We crop the heatmap indefinetely until the random float r
                # is below the sum of its weights (so regions with few bone
                # will get low p, so low chances of being picked)
                seed = numpy.random.randint(9999)
                heatmap_crop = method(heatmap, self.weight_source,
                                      seed)
                p = numpy.minimum(numpy.sum(heatmap_crop), 1)
                r = numpy.random.uniform()
                if r < p:
                    break
            for i, source_name in enumerate(self.data_stream.sources):
                if i == self.weight_i:
                    # We scale up the heatmap accordingly
                    data[i] = (method(data[i], source_name, seed) / p)\
                        .astype(numpy.float32)
                elif source_name in self.which_sources:
                    data[i] = method(data[i], source_name, seed)
        else:
            seed = numpy.random.randint(9999)
            for i, source_name in enumerate(self.data_stream.sources):
                if source_name in self.which_sources:
                    data[i] = method(data[i], source_name, seed)

        return tuple(data)

    def calculate_heatmap(self, volume):
        """
        Returns the same volume, scaled according to the weight of its center.
        Assumes the volume is big enough to allow for window_shape to fit at
        its center, with offsets of 1 and -1 in every directions

        Parameters:
        -----------
        volume: numpy.array
            Volume (weights) to scale ('batch', 'channel', dimensions)

        Returns:
        --------
        volume: numpy.array
            Scaled volume according to the weight of its center.
        """
        if isinstance(volume, list):
            raise(ValueError, "Volume type should not be a list")
        else:
            wshp = self.window_shape
            off = {}
            for i in range(len(volume.shape[2:])):
                off[i] = (volume.shape[2 + i] - wshp[i]) / 2
            p = []

            if len(volume.shape[2:]) == 3:
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        for k in [-1, 0, 1]:
                            p.append(
                                volume[:, :,
                                       off[0] + i:off[0] + wshp[0] + i,
                                       off[1] + j:off[1] + wshp[1] + j,
                                       off[2] + k:off[2] + wshp[2] + k].sum())
            elif len(volume.shape[2:]) == 2:
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        p.append(volume[:, :,
                                        off[0] + i:off[0] + wshp[0] + i,
                                        off[1] + j:off[1] + wshp[1] + j]
                                 .sum())
            return volume / max(p)

    def transform_source_batch(self, source, source_name, seed=None):
        """
        Crops randomly the source at self.window_shape size, according to
        the random seed.

        Parameters:
        -----------
        source: numpy.object or numpy.ndarray
            Volume to crop, if source is of type numpy.object,
            transform_source_example is called on the content of
            numpy.object, which should be numpy.ndarray
        source_name: str
            Name of the volume to crop
        seed: int
            Random seed to be used for the random cropping

        Returns:
        --------
        out: numpy.object or numpy.array
            Cropped source according to window_shape size and seed
        """
        if seed is None:
            seed = config.default_seed
            rng = numpy.random.RandomState(seed)
        else:
            rng = numpy.random.RandomState(seed)

        if source.dtype == numpy.object:
            return numpy.array([self.transform_source_example(im, source_name,
                                                              seed)
                                for im in source])
        elif isinstance(source, numpy.ndarray) and \
                (source.ndim == 4 or source.ndim == 5):
            out = numpy.empty(source.shape[:2] + self.window_shape,
                              dtype=source.dtype)
            batch_size = source.shape[0]
            if len(self.window_shape) != len(source.shape[2:]):
                raise ValueError("Window shape dimensions ({}) not "
                                 "consistent with source dimensions ({})"
                                 .format(len(self.window_shape),
                                         len(source[2:])))
            max_indices = {}
            offsets = {}
            for i in range(len(self.window_shape)):
                max_indices[i] = source.shape[2:][i] - self.window_shape[i]
                if max_indices[i] < 0:
                    raise ValueError("Got ndarray batch with image dimensions "
                                     "{} but requested window shape of {}".
                                     format(source.shape[2:],
                                            self.window_shape))
                offsets[i] = rng.random_integers(0, max_indices[i],
                                                 size=batch_size)
            if len(self.window_shape) == 2:
                window_batch_bchw(source, offsets[0], offsets[1], out)
            elif len(self.window_shape) == 3:
                window_batch_bchw3d(source, offsets[0], offsets[1],
                                    offsets[2], out)
            else:
                raise(NotImplementedError, "Cropping of N-D images with N>3 "
                                           "is not implemented")
            return out.astype(source.dtype)
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 3, or an array with "
                             "ndim = 4")

    def transform_source_example(self, example, source_name, seed=None):
        """
        Crops randomly the source at self.window_shape size, according to
        the random seed.

        Parameters:
        -----------
        source: numpy.ndarray
            Volume to crop, if source is of type numpy.object,
            transform_source_example is called.
        source_name: str
            Name of the volume to crop
        seed: int
            Random seed to be used for the random cropping

        Returns:
        --------
        out: numpy.object
            Cropped source according to window_shape size and seed
        """
        if seed is None:
            rng = numpy.random.RandomState(config.default_seed)
        else:
            rng = numpy.random.RandomState(seed)

        if not isinstance(example, numpy.ndarray) or \
                example.ndim not in [3, 4]:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3 or ndim = 4")
        if example.ndim != len(self.window_shape) + 1:
            raise ValueError("Window shape dimensions ({}) is not "
                             "consistent with source dimensions ({})"
                             .format(len(self.window_shape), len(example[1:])))
        max_indices = {}
        offsets = {}
        for i in range(len(self.window_shape)):
            max_indices[i] = example.shape[1:][i] - self.window_shape[i]
            if max_indices[i] < 0:
                raise ValueError("Got ndarray batch with image dimensions "
                                 "{} but requested window shape of {}".
                                 format(example.shape[1:],
                                        self.window_shape))
            offsets[i] = rng.random_integers(0, max_indices[i])

        if len(self.window_shape) == 2:
            out = example[:,
                          offsets[0]:offsets[0] + self.window_shape[0],
                          offsets[1]:offsets[1] + self.window_shape[1]]
        else:
            out = example[:,
                          offsets[0]:offsets[0] + self.window_shape[0],
                          offsets[1]:offsets[1] + self.window_shape[1],
                          offsets[2]:offsets[2] + self.window_shape[2]]
        return out.astype(example.dtype)

    def transform_example(self, example):
        return self._apply_samplewise_transformation(
            data=example, method=self.transform_source_example)

    def transform_batch(self, batch):
        return self._apply_samplewise_transformation(
            data=batch, method=self.transform_source_batch)


class RandomFixedSizeCrop(SourcewiseTransformer, ExpectsAxisLabels):
    """Randomly crop images to a fixed window size.
    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    window_shape : tuple
        The `(height, width)` tuple representing the size of the output
        window.
    Notes
    -----
    This transformer expects to act on stream sources which provide one of
     * Single images represented as 3-dimensional ndarrays, with layout
       `(channel, height, width)`.
     * Batches of images represented as lists of 3-dimensional ndarrays,
       possibly of different shapes (i.e. images of differing
       heights/widths).
     * Batches of images represented as 4-dimensional ndarrays, with
       layout `(batch, channel, height, width)`.
    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.
    """
    def __init__(self, data_stream, window_shape, **kwargs):
        if not window_batch_bchw_available:
            raise ImportError('window_batch_bchw not compiled')
        self.window_shape = window_shape
        self.rng = kwargs.pop('rng', None)
        self.warned_axis_labels = False
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomFixedSizeCrop, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        windowed_height, windowed_width = self.window_shape
        if isinstance(source, list) and all(isinstance(b, numpy.ndarray) and
                                            b.ndim == 3 for b in source):
            return [self.transform_source_example(im, source_name)
                    for im in source]
        elif isinstance(source, numpy.ndarray) and \
                source.dtype == numpy.object:
            return numpy.array([self.transform_source_example(im,
                                                              source_name)
                                for im in source])
        elif isinstance(source, numpy.ndarray) and source.ndim == 4:
            # Hardcoded assumption of (batch, channels, height, width).
            # This is what the fast Cython code supports.
            out = numpy.empty(source.shape[:2] + self.window_shape,
                              dtype=source.dtype)
            batch_size = source.shape[0]
            image_height, image_width = source.shape[2:]
            max_h_off = image_height - windowed_height
            max_w_off = image_width - windowed_width
            if max_h_off < 0 or max_w_off < 0:
                raise ValueError("Got ndarray batch with image dimensions {} "
                                 "but requested window shape of {}".format(
                                     source.shape[2:], self.window_shape))
            offsets_w = self.rng.random_integers(0, max_w_off, size=batch_size)
            offsets_h = self.rng.random_integers(0, max_h_off, size=batch_size)
            window_batch_bchw(source, offsets_h, offsets_w, out)
            return out
        elif all(isinstance(b, numpy.ndarray) and b.ndim == 3 for b in source):
            return [self.transform_source_example(im, source_name)
                    for im in source]
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 3, or an array with "
                             "ndim = 4")

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        windowed_height, windowed_width = self.window_shape
        if not isinstance(example, numpy.ndarray) or example.ndim != 3:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3")
        image_height, image_width = example.shape[1:]
        if image_height < windowed_height or image_width < windowed_width:
            raise ValueError("can't obtain ({}, {}) window from image "
                             "dimensions ({}, {})".format(
                                 windowed_height, windowed_width,
                                 image_height, image_width))
        if image_height - windowed_height > 0:
            off_h = self.rng.random_integers(0, image_height - windowed_height)
        else:
            off_h = 0
        if image_width - windowed_width > 0:
            off_w = self.rng.random_integers(0, image_width - windowed_width)
        else:
            off_w = 0
        return example[:, off_h:off_h + windowed_height,
                       off_w:off_w + windowed_width]


class RandomFixedSizeCrop3D(RandomFixedSizeCrop):
    """An extension of the RandomFixedSizeCrop that works for 3D arrays.
    It assumes that the first dimension (2nd in batch mode) are the channels.
    All other dimension will be preserved.
    Randomly crop images to a fixed window size.
    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    window_shape : tuple
        The `(height, width, depth)` tuple representing the size of the output
        window.
    Notes
    -----
    This transformer expects to act on stream sources which provide one of
     * Single images represented as 4-dimensional ndarrays, with layout
       `(channel, height, width, depth)`.
     * Batches of images represented as lists of 4-dimensional ndarrays,
       possibly of different shapes (i.e. images of differing
       heights/widths/depths).
     * Batches of images represented as 5-dimensional ndarrays, with
       layout `(batch, channel, height, width, depth)`.
    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.
    """
    def transform_source_batch(self, source, source_name):
        self.verify_axis_labels(('batch', 'channel', 'x', 'y', 'z'),
                                self.data_stream.axis_labels,
                                source_name)
        window_x, window_y, window_z = self.window_shape
        if isinstance(source, list) and \
                all(isinstance(b, numpy.ndarray) and
                    b.ndim == 4 for b in source):
            return [self.transform_source_example(im, source_name)
                    for im in source]
        elif isinstance(source, numpy.ndarray) and source.ndim == 5:
            # Hardcoded assumption of (batch, channels, height, width, depth).
            # This is what the fast Cython code supports.
            out = numpy.empty(source.shape[:2] + self.window_shape,
                              dtype=source.dtype)
            batch_size = source.shape[0]
            image_x, image_y, image_z = source.shape[2:]
            max_x_off = image_x - window_x
            max_y_off = image_y - window_y
            max_z_off = image_z - window_z
            if max_x_off < 0 or max_y_off < 0 or max_z_off < 0:
                raise ValueError("Got ndarray batch with image dimensions {} "
                                 "but requested window shape of {}".format(
                                     source.shape[2:], self.window_shape))
            offsets_x = self.rng.random_integers(0, max_x_off, size=batch_size)
            offsets_y = self.rng.random_integers(0, max_y_off, size=batch_size)
            offsets_z = self.rng.random_integers(0, max_z_off, size=batch_size)
            window_batch_bchw3d(source, offsets_x, offsets_y, offsets_z, out)
            return out
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 4, or an array with "
                             "ndim = 5")

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'x', 'y', 'z'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        window_x, window_y, window_z = self.window_shape
        if not isinstance(example, numpy.ndarray) or example.ndim != 4:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 4")
        image_x, image_y, image_z = example.shape[1:]
        if image_x < window_x or image_y < window_y or image_z < window_z:
            raise ValueError("can't obtain ({}, {}, {}) window from image "
                             "dimensions ({}, {}, {})".format(
                                 window_x, window_y, window_z,
                                 image_x, image_y, image_z))
        if image_x - window_x > 0:
            off_x = self.rng.random_integers(0, image_x - window_x)
        else:
            off_x = 0
        if image_y - window_y > 0:
            off_y = self.rng.random_integers(0, image_y - window_y)
        else:
            off_y = 0
        if image_z - window_z > 0:
            off_z = self.rng.random_integers(0, image_z - window_z)
        else:
            off_z = 0
        return example[:,
                       off_x:off_x + window_x,
                       off_y:off_y + window_y,
                       off_z:off_z + window_z]


class FixedSizeCrop(SourcewiseTransformer, ExpectsAxisLabels):
    """Crop images to a fixed window size.
    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    window_shape : tuple
        The `(height, width)` tuple representing the size of the output
        window.
    location : tuple
        Location of the crop (height, width) given relatively to the volume
        (each between 0 and 1, where (0, 0) is the top left corner and (1,
        1) the lower right corner and (.5, .5) the center).
    Notes
    -----
    This transformer expects to act on stream sources which provide one of
     * Single images represented as 3-dimensional ndarrays, with layout
       `(channel, height, width)`.
     * Batches of images represented as lists of 3-dimensional ndarrays,
       possibly of different shapes (i.e. images of differing
       heights/widths).
     * Batches of images represented as 4-dimensional ndarrays, with
       layout `(batch, channel, height, width)`.
    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.
    """
    def __init__(self, data_stream, window_shape, location, **kwargs):
        self.window_shape = window_shape
        self.warned_axis_labels = False
        if not isinstance(location, (list, tuple)) or len(location) != 2:
            raise ValueError('Location must be a tuple or list of length 2 '
                             '(given {}).'.format(location))
        if location[0] < 0 or location[0] > 1 or location[1] < 0 or \
                location[1] > 1:
            raise ValueError('Location height and width must be between 0 '
                             'and 1 (given {}).'.format(location))
        self.location = location
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(FixedSizeCrop, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        if isinstance(source, list) and all(isinstance(b, numpy.ndarray) and
                                            b.ndim == 3 for b in source):
            return [self.transform_source_example(im, source_name)
                    for im in source]
        elif isinstance(source, numpy.ndarray) and \
                source.dtype == numpy.object:
            return numpy.array([self.transform_source_example(im,
                                                              source_name)
                                for im in source])
        elif isinstance(source, numpy.ndarray) and source.ndim == 4:
            # Hardcoded assumption of (batch, channels, height, width).
            # This is what the fast Cython code supports.
            windowed_height, windowed_width = self.window_shape
            image_height, image_width = source.shape[2:]
            loc_height, loc_width = self.location
            if image_height < windowed_height or image_width < windowed_width:
                raise ValueError("can't obtain ({}, {}) window from image "
                                 "dimensions ({}, {})".format(
                                     windowed_height, windowed_width,
                                     image_height, image_width))
            off_h = int(round((image_height - windowed_height) * loc_height))
            off_w = int(round((image_width - windowed_width) * loc_width))
            return source[:, :, off_h:off_h + windowed_height,
                          off_w:off_w + windowed_width]
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 3, or an array with "
                             "ndim = 4")

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        windowed_height, windowed_width = self.window_shape
        loc_height, loc_width = self.location
        if not isinstance(example, numpy.ndarray) or example.ndim != 3:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3")
        image_height, image_width = example.shape[1:]
        if image_height < windowed_height or image_width < windowed_width:
            raise ValueError("can't obtain ({}, {}) window from image "
                             "dimensions ({}, {})".format(
                                 windowed_height, windowed_width,
                                 image_height, image_width))
        off_h = int(round((image_height - windowed_height) * loc_height))
        off_w = int(round((image_width - windowed_width) * loc_width))
        return example[:, off_h:off_h + windowed_height,
                       off_w:off_w + windowed_width]


class FixedSizeCropND(SourcewiseTransformer):
    """Crop N-D volumes to a fixed window size.
    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    window_shape : tuple
        The tuple representing the size of the output window.
    location : tuple
        Location of the crop given relatively to the volume
        (each between 0 and 1, where (0, 0) is the top left corner and (1,
        1) the lower right corner and (.5, .5) the center).
    Notes
    -----
    This transformer expects to act on stream sources which provide one of
     * Single volumes represented as n-dimensional ndarrays, with first
       dimension being the channel dimension
     * Batches of volumes represented as lists of n-dimensional ndarrays,
       possibly of different shapes
     * Batches of images represented as n-dimensional ndarrays, with
       layout `(batch, channel) + volume_shape`

    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.
    """
    def __init__(self, data_stream, window_shape, location, **kwargs):
        self.window_shape = window_shape
        if not isinstance(location, (list, tuple)):
            raise ValueError('Location must be a tuple or list'
                             '(given {}).'.format(type(location)))
        if len(location) != len(window_shape):
            raise ValueError('Location (ndims={}) and window shape (ndims={} '
                             'must have the same number of dimensions'
                             .format(len(location), len(window_shape)))
        if not all(0 <= i <= 1 for i in location):
            raise ValueError('Location values must be between 0 '
                             'and 1 (given {}).'.format(location))
        self.location = location
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(FixedSizeCropND, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        if isinstance(source, list) and all(isinstance(b, numpy.ndarray)
                                            for b in source):
            return [self.transform_source_example(im, source_name)
                    for im in source]
        elif isinstance(source, numpy.ndarray) and \
                source.dtype == numpy.object:
            return numpy.array([self.transform_source_example(im, source_name)
                                for im in source])
        elif isinstance(source, numpy.ndarray):
            if any(vol_sh < win_sh for vol_sh, win_sh
                   in zip(source.shape[2:], self.window_shape)):
                raise ValueError("can't obtain {} window from image "
                                 "dimensions {}"
                                 .format(self.window_shape, source.shape[2:]))
            for i in range(len(self.window_shape)):
                off = int(round((source.shape[2 + i] - self.window_shape[i]) *
                          self.location[i]))
                source = numpy.take(source,
                                    range(off, off + self.window_shape[i]),
                                    axis=2 + i)
            return source
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays, or an array")

    def transform_source_example(self, example, source_name):
        if not isinstance(example, numpy.ndarray):
            raise ValueError("uninterpretable example format; expected "
                             "ndarray")
        if len(example.shape[1:]) != len(self.window_shape) or \
            any(vol_sh < win_sh for vol_sh, win_sh in zip(example.shape[1:],
                                                          self.window_shape)):
            raise ValueError("can't obtain {} window from image dimensions {}"
                             .format(self.window_shape, example.shape[1:]))
        for i in range(len(self.window_shape)):
            off = int(round((example.shape[1 + i] - self.window_shape[i]) *
                            self.location[i]))

            example = numpy.take(example,
                                 range(off, off + self.window_shape[i]),
                                 axis=1 + i)
        return example


class RandomSpatialFlip(SourcewiseTransformer):
    """Randomly flip images horizontally and/or vertically.
    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    flip_h : bool
        Whether to flip images horizontally
    flip_v : bool
        Whether to flip images vertically
    Notes
    -----
    This transformer expects to act on stream sources which provide one of

     * Single images represented as 3-dimensional ndarrays, with layout
       `(channel, height, width)`.
     * Batches of images represented as lists of 3-dimensional ndarrays,
       possibly of different shapes (i.e. images of differing
       heights/widths).
     * Batches of images represented as 4-dimensional ndarrays, with
       layout `(batch, channel, height, width)`.

    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.
    """
    def __init__(self, data_stream, flip_h=False, flip_v=False, **kwargs):
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.rng = kwargs.pop('rng', None)
        self.warned_axis_labels = False
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        super(RandomSpatialFlip, self).__init__(data_stream, **kwargs)

    def transform_source_example(self, source_example, source_name,
                                 example_flip_h=False, example_flip_v=False):
        output = source_example

        if example_flip_h:
            output = output[..., ::-1]

        if example_flip_v:
            output = output[..., ::-1, :]

        return output.astype(source_example.dtype)

    def transform_source_batch(self, source, source_name):

        # source is list of np.array with dim 3
        if isinstance(source, list) \
                and all(isinstance(example, numpy.ndarray) and
                        example.ndim == 3 for example in source):
            to_flip_h, to_flip_v = self.get_flip_vectors(
                batch_size=len(source))
            to_flip_h = to_flip_h == 1  # convert to bool list
            to_flip_v = to_flip_v == 1

            output = [self.transform_source_example(example, source_name,
                                                    ex_flip_h, ex_flip_v)
                      for example, ex_flip_h, ex_flip_v
                      in zip(source, to_flip_h, to_flip_v)]

            return output

        # source is np.object of np.array with dim 3
        elif source.dtype == numpy.object:
            to_flip_h, to_flip_v = self.get_flip_vectors(
                batch_size=source.shape[0])
            to_flip_h = to_flip_h == 1  # convert to bool list
            to_flip_v = to_flip_v == 1

            output = numpy.empty(source.shape[0], dtype=object)

            for i in range(source.shape[0]):
                output[i] = self.transform_source_example(source[i],
                                                          source_name,
                                                          to_flip_h[i],
                                                          to_flip_v[i])

            return output

        # source is np.array with dim 4 (batch, channels, height, width)
        elif isinstance(source, numpy.ndarray) and source.ndim == 4:
            to_flip_h, to_flip_v = self.get_flip_vectors(
                batch_size=source.shape[0])
            to_flip_h = to_flip_h.reshape([source.shape[0]] + [1] * 3)
            to_flip_v = to_flip_v.reshape([source.shape[0]] + [1] * 3)

            output = self.flip_batch(source, to_flip_h, to_flip_v)

            return output.astype(source.dtype)

        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 3, or an array with "
                             "ndim = 4")

    def get_flip_vectors(self, batch_size):

        if self.flip_h:
            to_flip_h = self.rng.binomial(n=1, p=0.5, size=batch_size)
        else:
            to_flip_h = numpy.zeros(batch_size)

        if self.flip_v:
            to_flip_v = self.rng.binomial(n=1, p=0.5, size=batch_size)
        else:
            to_flip_v = numpy.zeros(batch_size)

        return to_flip_h, to_flip_v

    @staticmethod
    def flip_batch(batch, to_flip_h, to_flip_v):
        batch = batch * (1 - to_flip_h) + batch[..., ::-1] * to_flip_h
        batch = batch * (1 - to_flip_v) + batch[..., ::-1, :] * to_flip_v
        return batch


class Image2DSlicer(SourcewiseTransformer):
    """Applies a transformation to a source.

    The data can either be an example or a batch of examples.

    Parameters
    ----------
    source_data : :class:`numpy.ndarray`
        Data from a source.
        Assuming images of dimensionality (num_samples, channel, x, y, z)
    source_name : str
        The name of the source being operated upon.
    dimension_to_slice: str or int
        Dimension "x", "y", "z" or 0, 1, 2.
    slice_location: str
        Randomly or centerwise.exit
    batch_or_channel: int
        If slicing along each dimension: 0 for batchwise or 1 for
        channelwise concatenation of the output.
    """
    def __init__(self, data_stream,
                 slice_location='center',
                 dimension_to_slice=None,
                 batch_or_channel=None, **kwargs):
        super(Image2DSlicer, self).__init__(
            data_stream=data_stream,
            produces_examples=data_stream.produces_examples,
            **kwargs)
        self.dim_to_slice = dimension_to_slice
        self.slice_loc = slice_location
        self.batch_or_channel = batch_or_channel

    def transform_source_batch(self, source, name):
        # Assuming all dimensions are of the same size
        src_shape = source.shape
        if self.slice_loc == 'random':
            pick = numpy.random.binomial(src_shape[2], p=0.5, size=3)
        elif self.slice_loc == 'center':
            pick = numpy.asarray(src_shape[2:]) / 2
        else:
            raise ValueError('Slice location must be either "random" or '
                             '"center".')

        # Slice along a specified dimension
        if self.dim_to_slice is not None:
            check = str(self.dim_to_slice).lower()
            if (check not in 'xyz012') or (len(check) > 1):
                raise ValueError('Unknown dimension {}. Use either one of '
                                 '"x", "y", "z" or 0 ,1 ,2.'
                                 .format(self.dim_to_slice))
            else:
                # return a 2D slice along the specified dimension
                if check in 'x0':
                    return source[:, :, pick[0]]        # dimension x
                elif check in 'y1':
                    return source[:, :, :, pick[1]]     # dimension y
                elif check in 'z2':
                    return source[:, :, :, :, pick[2]]  # dimension z

        # Slice along each dimension
        else:
            if self.batch_or_channel is None:
                raise ValueError('If slicing along each dimension, need to  '
                                 'specify axis along which to concatenate '
                                 'the output.')
            elif str(self.batch_or_channel).lower() not in '01':
                raise ValueError('Invalid concatenation axis, use either 0 '
                                 'for  channelwise or 1 for batchwise '
                                 'concatenation of the slices.')
            else:
                x = source[:, :, pick[0]]        # x-th slice
                y = source[:, :, :, pick[1]]     # y-th slice
                z = source[:, :, :, :, pick[2]]  # z-th slice

                return numpy.concatenate((x, y, z), axis=self.batch_or_channel)


class GammaCorrectionND(SourcewiseTransformer):
    """
    Applies gamma correction to a source, only to pixel which values are
    between 0 and 1.

    Parameters
    ----------
    gamma: float
        Gamma correction to apply
    """
    def __init__(self, data_stream, gamma, **kwargs):
        self.gamma = gamma
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(GammaCorrectionND, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        if isinstance(source, list) and all(isinstance(b, numpy.ndarray)
                                            for b in source):
            return [self.transform_source_example(im, source_name)
                    for im in source]
        elif isinstance(source, numpy.ndarray) and \
                source.dtype == numpy.object:
            return numpy.array([self.transform_source_example(im, source_name)
                                for im in source])
        elif isinstance(source, numpy.ndarray):
            return self.gamma_correction(source, self.gamma)
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays, or an array")

    def transform_source_example(self, example, source_name):
        if not isinstance(example, numpy.ndarray):
            raise ValueError("uninterpretable example format; expected "
                             "ndarray")
        return self.gamma_correction(example, self.gamma)

    @staticmethod
    def gamma_correction(image, gamma):
        gamma_corrected = image.copy()
        mask = numpy.logical_and(gamma_corrected >= 0, gamma_corrected <= 1)
        gamma_corrected[mask] = numpy.power(gamma_corrected[mask], gamma)
        return gamma_corrected


class Random2DRotation(SourcewiseTransformer, ExpectsAxisLabels):
    """Randomly rotate 2D images in the spatial plane.

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    maximum_rotation : float, default `math.pi`
        Maximum amount of rotation in radians. The image will be rotated by
        an angle in the range [-maximum_rotation, maximum_rotation].
    resample : str, optional
        Resampling filter for PIL to use to upsample any images requiring
        it. Options include 'nearest' (default), 'bilinear', and 'bicubic'.
        See the PIL documentation for more detailed information.

    Notes
    -----
    This transformer expects to act on stream sources which provide one of

     * Single images represented as 3-dimensional ndarrays, with layout
       `(channel, height, width)`.
     * Batches of images represented as lists of 3-dimensional ndarrays,
       possibly of different shapes (i.e. images of differing
       heights/widths).
     * Batches of images represented as 4-dimensional ndarrays, with
       layout `(batch, channel, height, width)`.

    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.

    """
    def __init__(self, data_stream, maximum_rotation=math.pi,
                 resample='nearest', **kwargs):
        if maximum_rotation <= 0 or maximum_rotation > math.pi:
            raise ValueError('maximum_rotation ({:.5f}) must be in the range '
                             '(0, math.pi]'.format(maximum_rotation))
        self.maximum_rotation = numpy.rad2deg(maximum_rotation)
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))

        self.rng = kwargs.pop('rng', None)
        self.warned_axis_labels = False
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(Random2DRotation, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        rotation_angles = self.rng.uniform(-self.maximum_rotation,
                                           self.maximum_rotation,
                                           len(source))
        if isinstance(source, list) and all(isinstance(b, numpy.ndarray) and
                                            b.ndim == 3 for b in source):
            return [self._example_transform(im, angle)
                    for im, angle in zip(source, rotation_angles)]
        elif isinstance(source, numpy.ndarray) and source.dtype == object and \
                all(isinstance(b, numpy.ndarray) and b.ndim == 3 for b in
                    source):
            out = numpy.empty(len(source), dtype=object)
            for im_idx, (im, angle) in enumerate(zip(source, rotation_angles)):
                out[im_idx] = self._example_transform(im, angle)
            return out
        elif isinstance(source, numpy.ndarray) and source.ndim == 4:
            return numpy.array([self._example_transform(im, angle)
                                for im, angle in zip(source, rotation_angles)],
                               dtype=source.dtype)
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 3, or an array with "
                             "ndim = 4")

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        if not isinstance(example, numpy.ndarray) or example.ndim != 3:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3")
        rotation_angle = self.rng.uniform(-self.maximum_rotation,
                                          self.maximum_rotation)
        return self._example_transform(example, rotation_angle)

    def _example_transform(self, example, rotation_angle):
        dt = example.dtype
        im = Image.fromarray(example.transpose(1, 2, 0))
        example = numpy.array(im.rotate(rotation_angle,
                                        resample=self.resample)).astype(dt)
        return example.transpose(2, 0, 1)


class Drop(SourcewiseTransformer):
    """
    Implement border drop (of size `border) and dropout`(with probability of
    dropping of `dropout`) on the volume directed by the variable
    `which_weight`.

    Parameters:
    -----------
    stream: instance of :class:`DataStream`
        The wrapped data stream.
    which_sources: str or list of str
        Name of the sources that will be affected by the transformer.
    border: int
        Size of the border of the volume.
    dropout: float
        Probability of dropping out an element of the volume.
    produces_examples: bool
        True for example streams, False for batch streams
    """
    def __init__(self, stream, which_sources,
                 border=None, dropout=None,
                 produces_examples=False,
                 **kwargs):
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        super(Drop, self).__init__(stream, produces_examples, which_sources,
                                   **kwargs)
        if border is None or isinstance(border, int):
            self.border = border
        else:
            raise TypeError("Parameter border should be an int "
                            "(type passed {}).".format(type(border)))
        if dropout is not None:
            if not isinstance(dropout, (float, int)):
                raise TypeError("Parameter dropout should be float or int, "
                                "received type {}".format(type(dropout)))
            if 0 <= dropout <= 1:
                self.dropout = dropout
            else:
                raise ValueError("Parameter dropout should be between "
                                 "0 and 1 (value passed: {}).".format(dropout))
        else:
            self.dropout = dropout

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        if self.produces_examples:
            return self.transform_example(data)
        else:
            return self.transform_batch(data)

    def transform_source_batch(self, source, source_name):
        if isinstance(source, numpy.ndarray) and source.dtype == numpy.object:
            return numpy.array([self.transform_source_example(im, source_name)
                                for im in source])
        elif isinstance(source, numpy.ndarray) and \
                (source.ndim == 4 or source.ndim == 5):
            if self.border is not None:
                source = self._border_func(source, self.border, 'source')
            if self.dropout is not None:
                source = self._dropout_func(source, self.dropout, self.rng)
            return source
        else:
            raise ValueError("uninterpretable source format; expected ndarray "
                             "with ndim = 4 or ndim = 5, got {} instead."
                             .format(type(source)))

    def transform_source_example(self, example, source_name):
        if isinstance(example, numpy.ndarray) and \
                        example.ndim in [3, 4]:
            if self.border is not None:
                example = self._border_func(example, self.border, 'example')
            if self.dropout is not None:
                example = self._dropout_func(example, self.dropout, self.rng)
            return example
        else:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3 or ndim = 4, "
                             "got {} instead".format(type(example)))

    def _border_func(self, volume, border, flag=None):
        if flag == 'source':
            for i in range(len(volume.shape[2:])):
                if volume.shape[2+i] <= 2 * border:
                    raise ValueError("border does not fit in image (dimension"
                                     "{} size {}, borders {}"
                                     .format(i, volume.shape[2+i],
                                             2 * border))
            if volume.ndim == 5:
                volume[:, :, :border, :, :] = 0
                volume[:, :, :, :border, :] = 0
                volume[:, :, :, :, :border] = 0
                volume[:, :, -border:, :, :] = 0
                volume[:, :, :, -border:, :] = 0
                volume[:, :, :, :, -border:] = 0
            elif volume.ndim == 4:
                volume[:, :, :border, :] = 0
                volume[:, :, :, :border] = 0
                volume[:, :, -border:, :] = 0
                volume[:, :, :, -border:] = 0
            else:
                raise ValueError("uninterpretable number of dimensions for "
                                 "source volume, expected 4 or 5, got {} "
                                 "instead".format(volume.ndim))

        elif flag == 'example':
            for i in range(len(volume.shape[1:])):
                if volume.shape[1+i] <= 2 * border:
                    raise ValueError("border does not fit in image (dimension "
                                     "{} size {}, borders {}"
                                     .format(i, volume.shape[1+i], 2 * border))
            if volume.ndim == 4:
                volume[:, :border, :, :] = 0
                volume[:, :, :border, :] = 0
                volume[:, :, :, :border] = 0
                volume[:, -border:, :, :] = 0
                volume[:, :, -border:, :] = 0
                volume[:, :, :, -border:] = 0
            elif volume.ndim == 3:
                volume[:, :border, :] = 0
                volume[:, :, :border] = 0
                volume[:, -border:, :] = 0
                volume[:, :, -border:] = 0
            else:
                raise ValueError("uninterpretable number of dimensions for "
                                 "source volume, expected 3 or 4, got {} "
                                 "instead".format(volume.ndim))
        else:
            raise ValueError("Expected flag as 'source' or 'example' "
                             "got {} instead".format(flag))
        return volume.astype(volume.dtype)

    def _dropout_func(self, volume, dropout, rng):
        dropout_cast = rng.binomial(1,
                                    1 - dropout,
                                    size=volume.shape)
        return (volume * dropout_cast).astype(volume.dtype)
