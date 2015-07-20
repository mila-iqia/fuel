from __future__ import division
from io import BytesIO
import math

import numpy
from PIL import Image
from six import PY3

from ._image import window_batch_bchw
from . import ExpectsAxisLabels, SourcewiseTransformer
from .. import config


class ImagesFromBytes(SourcewiseTransformer):
    """Load from a stream of image-formatted data represented as bytes.

    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The wrapped data stream.
    which_sources : tuple of str
        Which sources to apply the mapping to.
    color_mode : str, optional
        Mode to pass to PIL for color space conversion. Default is RGB.
        If `None`, no coercion is performed.

    Notes
    -----
    Images are returned as NumPy arrays converted from PIL objects.
    If `color_mode` is RGB, then the array is transposed from the
    `(height, width, channel)` dimension layout native to PIL to the
    `(channel, height, width)` layout that is pervasive in the world of
    convolutional networks.

    This SourcewiseTransformer supports streams returning single examples
    as `bytes` objects (`str` on Python 2.x) as well as streams that
    return iterables containing such objects. In the case of an
    iterable, a list of loaded images is returned.

    """
    def __init__(self, data_stream, which_sources, color_mode='RGB', **kwargs):
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        super(ImagesFromBytes, self).__init__(data_stream, which_sources,
                                              **kwargs)
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
        if self.color_mode == 'RGB':
            # Transpose to `(channels, height, width)` layout.
            return image.transpose(2, 0, 1)
        else:
            return image

    def transform_source_batch(self, batch, source_name):
        return [self.transform_source_example(im, source_name) for im in batch]


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
                im = example.tranpose(2, 0, 1)
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
       `(n_channels, height, width)`.
     * Batches of images represented as lists of 3-dimensional ndarrays,
       possibly of different shapes (i.e. images of differing
       heights/widths).
     * Batches of images represented as 4-dimensional ndarrays, with
       layout `(batch, n_channels, height, width)`.

    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.

    """
    def __init__(self, data_stream, window_shape, **kwargs):
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
        if isinstance(source, list):
            return [self.transform_source_example(im, source_name)
                    for im in source]
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
