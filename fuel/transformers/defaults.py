"""Commonly-used default transformers."""
from fuel.transformers import ScaleAndShift, Cast, SourcewiseTransformer
from fuel.transformers.image import ImagesFromBytes


def uint8_pixels_to_floatX(which_sources):
    return (
        (ScaleAndShift, [1 / 255.0, 0], {'which_sources': which_sources}),
        (Cast, ['floatX'], {'which_sources': which_sources}))


class ToBytes(SourcewiseTransformer):
    """Transform a stream of ndarray examples to bytes.

    Notes
    -----
    Used for retrieving variable-length byte data stored as, e.g. a uint8
    ragged array.

    """
    def __init__(self, stream, **kwargs):
        kwargs.setdefault('produces_examples', stream.produces_examples)
        axis_labels = (stream.axis_labels
                       if stream.axis_labels is not None
                       else {})
        for source in kwargs.get('which_sources', stream.sources):
            axis_labels[source] = (('batch', 'bytes')
                                   if 'batch' in axis_labels.get(source, ())
                                   else ('bytes',))
        kwargs.setdefault('axis_labels', axis_labels)
        super(ToBytes, self).__init__(stream, **kwargs)

    def transform_source_example(self, example, _):
        return example.tostring()

    def transform_source_batch(self, batch, _):
        return [example.tostring() for example in batch]


def rgb_images_from_encoded_bytes(which_sources):
    return ((ToBytes, [], {'which_sources': which_sources}),
            (ImagesFromBytes, [], {'which_sources': which_sources}))
