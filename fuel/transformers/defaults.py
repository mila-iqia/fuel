"""Commonly-used default transformers."""
from fuel.transformers import ScaleAndShift, Cast


def uint8_pixels_to_floatX(which_sources):
    return (
        (ScaleAndShift, [1 / 255.0, 0], {'which_sources': which_sources}),
        (Cast, ['floatX'], {'which_sources': which_sources}))
