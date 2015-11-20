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


cpdef window_batch_bchw3d(image_dtype[:, :, :, :, :] batch,
                        long[:] x_offsets,
                        long[:] y_offsets,
                        long[:] z_offsets,
                        image_dtype[:, :, :, :, :] out):
    """window_batch_bchw(batch, window_height, window_width,
                         height_offsets, width_offsets, out)

    Perform windowing on a (batch, channels, height, width, depth) image
    tensor.

    Parameters
    ----------
    batch : memoryview, 4-dimensional
        A 4-d tensor containing a batch of images in the expected
        format above.
    x_offsets : memoryview, integer, 1-dimensional
        An array of offsets for the x dimension of each image.
        Assumed that batch.shape[0] <= x_offsets.shape[0].
    y_offsets : memoryview, integer, 1-dimensional
        An array of offsets for the y dimension of each image.
        Assumed that batch.shape[0] <= y_offsets.shape[0].
    z_offsets: memoryview, integer, 1-dimensional
        An array of offsets for the z dimension of each image.
        Assumed that batch.shape[0] <= z_offsets.shape[0].
    out : memoryview
        The array to which to write output. It is assumed that
        `out.shape[2] + x_offsets[i] <= batch.shape[2]`,
        `out.shape[3] + y_offsets[i] <= batch.shape[3]`, and
        `out.shape[4] + z_offsets[i] <= batch.shape[4]`, for
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
    cdef Py_intptr_t window_x = out.shape[2]
    cdef Py_intptr_t window_y = out.shape[3]
    cdef Py_intptr_t window_z = out.shape[4]
    cdef Py_intptr_t x_off, y_off, z_off, x_extent, y_extent, z_extent
    with nogil:
        for index in prange(batch.shape[0]):
            x_off = x_offsets[index]
            y_off = y_offsets[index]
            z_off = z_offsets[index]
            x_extent = x_off + window_x
            y_extent = y_off + window_y
            z_extent = z_off + window_z
            out[index] = batch[index, :, x_off:x_extent,
                                         y_off:y_extent,
                                         z_off:z_extent]
