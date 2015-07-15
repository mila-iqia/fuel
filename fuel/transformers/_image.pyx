cimport cython
from cython.parallel cimport prange


ctypedef long Py_intptr_t

ctypedef fused image_dtype:
    float
    double
    unsigned char


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef window_batch_bchw(image_dtype[:, :, :, :] batch,
                        long[:] height_offsets, long[:] width_offsets,
                        image_dtype[:, :, :, :] out):
    """window_batch_bchw(batch, window_height, window_width,
                         height_offsets, width_offsets, out)

    Perform windowing on a (batch, channels, height, width) image tensor.

    Parameters
    ----------
    batch : memoryview, 4-dimensional
        A 4-d tensor containing a batch of images in the expected
        format above.
    height_offsets : memoryview, integer, 1-dimensional
        An array of offsets for the height dimension of each image.
        Assumed that batch.shape[0] <= height_offsets.shape[0].
    width_offsets : memoryview, integer, 1-dimensional
        An array of offsets for the width dimension of each image.
        Assumed that batch.shape[0] <= width_offsets.shape[0].
    out : memoryview
        The array to which to write output. It is assumed that
        `out.shape[2] + height_offsets[i] <= batch.shape[2]` and
        `out.shape[3] + width_offsets[i] <= batch.shape[3]`, for
        all values of `i`.

    Notes
    -----
    Operates on a batch in parallel via OpenMP. Set `OMP_NUM_THREADS`
    to benefit from this parallelism.

    This is a low-level utility that, for the sake of speed, does
    not check its input for validity. Some amount of protection is
    provided by Cython memoryview objects.

    """
    cdef Py_intptr_t index
    cdef Py_intptr_t window_width = out.shape[3]
    cdef Py_intptr_t window_height = out.shape[2]
    cdef Py_intptr_t h_off, w_off, h_extent, w_extent
    with nogil:
        for index in prange(batch.shape[0]):
            h_off = height_offsets[index]
            w_off = width_offsets[index]
            h_extent = h_off + window_height
            w_extent = w_off + window_width
            out[index] = batch[index, :, h_off:h_extent, w_off:w_extent]
