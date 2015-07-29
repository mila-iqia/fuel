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
    def transform_source_example(self, example, _):
        return example.tostring()

    def transform_source_batch(self, batch, _):
        return [example.tostring() for example in batch]


def rgb_images_from_encoded_bytes(which_sources):
    return ((ToBytes, [], {'which_sources': which_sources}),
            (ImagesFromBytes, [], {}))
