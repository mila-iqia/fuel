import os
import tarfile
import h5py
import shutil

import numpy
from PIL import Image

from fuel.converters.base import fill_hdf5_file, check_exists

COMPLETE_DATASET = 'weizmann_horse_db.tar.gz'


@check_exists(required_files=[COMPLETE_DATASET])
def convert_weizmann_horses(directory, output_directory, output_filename=None,
                            resize=False, resize_size=-1, zero_pad=True,
                            crop=False, split=[.44, .22]):
    """Converts the Weizmann Horse Dataset to HDF5.

    Converts the Weizmann Horse Dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.weizmann_horses`. The converted dataset is
    saved as 'weizmann_horse.hdf5'.

    This method assumes the existence of the following file:
    `weizmann_horse_db.tar.gz`

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to `None`, in which case a name
        based on `dtype` will be used.
    resize : boolean, optional
        If True, the images will be resized according to resize_size. Either
        zero pad or crop should be True.
    resize_size : list, optional
        The size to resize the images to. If not set and resize is True, the
        mean image size will be used instead.
    zero_pad : boolean, optional
        If resize is True and zero_pad is True, the image will be resized so
        that the longest axis is equal to the size specified in resize_size and
        the other axis will be zero-padded. Cannot be True if crop is True.
    crop : boolean, optional
        If resize is True and crop is True, the image will be resized so
        that the shortest axis is equal to the size specified in resize_size
        and the other axis will be cropped. Cannot be True if zero_pad is True.
    split : list, optional
        The size of the training and validation set, expressed as a percentage
        wrt the size of the dataset.

    Return
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    train_feat_path = os.path.join(directory, COMPLETE_DATASET)
    [train, valid, test, mean, bw_mean, std, bw_std,
     filenames] = read_weizmann_horses(train_feat_path, resize, zero_pad,
                                       crop, split)

    if not output_filename:
        output_filename = 'weizmann_horse'
        if resize:
            output_filename += '_' + str(resize_size)
            if zero_pad:
                output_filename += '_zero_padded'
            else:
                output_filename += '_cropped'
        output_filename += '_' + str(split[0]) + '_' + str(split[1]) + '.hdf5'
    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')

    if resize:
        data = (('train', 'color_images', train[0]),
                ('valid', 'color_images', valid[0]),
                ('test', 'color_images', test[0]),
                ('train', 'gray_images', train[1]),
                ('valid', 'gray_images', valid[1]),
                ('test', 'gray_images', test[1]),
                ('train', 'targets', train[2]),
                ('valid', 'targets', valid[2]),
                ('test', 'targets', test[2]),
                ('train', 'filenames', numpy.array(filenames[0])),
                ('valid', 'filenames', numpy.array(filenames[1])),
                ('test', 'filenames', numpy.array(filenames[2])),
                ('color', 'mean', numpy.array([mean])),
                ('gray', 'mean', numpy.array([bw_mean])),
                ('color', 'std_dev', numpy.array([std])),
                ('gray', 'std_dev', numpy.array([bw_std])))
        fill_hdf5_file(h5file, data)
        h5file['color_images'].dims[0].label = 'batch'
        h5file['color_images'].dims[1].label = 'height'
        h5file['color_images'].dims[2].label = 'width'
        h5file['color_images'].dims[3].label = 'channel'
        h5file['gray_images'].dims[0].label = 'batch'
        h5file['gray_images'].dims[1].label = 'height'
        h5file['gray_images'].dims[2].label = 'width'
        h5file['targets'].dims[0].label = 'batch'
        h5file['targets'].dims[1].label = 'height'
        h5file['targets'].dims[2].label = 'width'
        h5file['filenames'].dims[0].label = 'filename'
        h5file['mean'].dims[0].label = 'height'
        h5file['mean'].dims[1].label = 'width'
        h5file['mean'].dims[2].label = 'channel'
        h5file['std_dev'].dims[0].label = 'height'
        h5file['std_dev'].dims[1].label = 'width'
        h5file['std_dev'].dims[2].label = 'channel'

    else:
        data = (('train', 'filenames', numpy.array(filenames[0])),
                ('valid', 'filenames', numpy.array(filenames[1])),
                ('test', 'filenames', numpy.array(filenames[2])),
                ('color', 'mean', numpy.array([mean])),
                ('color', 'std_dev', numpy.array([std])),
                ('gray', 'mean', numpy.array([bw_mean])),
                ('gray', 'std_dev', numpy.array([bw_std])))
        fill_hdf5_file(h5file, data)
        h5file['filenames'].dims[0].label = 'filename'
        h5file['mean'].dims[0].label = 'dataset_mean'
        h5file['std_dev'].dims[0].label = 'dataset_std_dev'

        # adding variable length data
        dtype = h5py.special_dtype(vlen=numpy.dtype('float32'))
        # color
        concat_data = numpy.hstack((train[0], valid[0], test[0]))
        d = h5file.create_dataset('color_images', (len(concat_data),),
                                  dtype=dtype)
        d[...] = [image.flatten() for image in concat_data]
        d.dims[0].label = 'batch'
        s = h5file.create_dataset('color_images_shapes', (len(concat_data), 3),
                                  dtype='int32')
        s[...] = numpy.array([image.shape for image in concat_data])
        d.dims.create_scale(s, 'shapes')
        d.dims[0].attach_scale(s)
        l = h5file.create_dataset('color_images_shapes_labels', (3,),
                                  dtype='S7')
        l[...] = ['height'.encode('utf8'), 'width'.encode('utf8'),
                  'channel'.encode('utf8')]
        d.dims.create_scale(l, 'shape_labels')
        d.dims[0].attach_scale(l)

        # gray
        concat_data = numpy.hstack((train[1], valid[1], test[1]))
        d = h5file.create_dataset('gray_images', (len(concat_data),),
                                  dtype=dtype)
        d[...] = [image.flatten() for image in concat_data]
        d.dims[0].label = 'batch'
        s = h5file.create_dataset('gray_images_shapes', (len(concat_data), 2),
                                  dtype='int32')
        s[...] = numpy.array([image.shape for image in concat_data])
        d.dims.create_scale(s, 'shapes')
        d.dims[0].attach_scale(s)
        l = h5file.create_dataset('gray_images_shapes_labels', (3,),
                                  dtype='S7')
        l[...] = ['height'.encode('utf8'), 'width'.encode('utf8'),
                  'channel'.encode('utf8')]
        d.dims.create_scale(l, 'shape_labels')
        d.dims[0].attach_scale(l)

        # target
        concat_data = numpy.hstack((train[2], valid[2], test[2]))
        d = h5file.create_dataset('targets', (len(concat_data),),
                                  dtype=dtype)
        d[...] = [image.flatten() for image in concat_data]
        d.dims[0].label = 'batch'
        s = h5file.create_dataset('targets_shapes', (len(concat_data), 2),
                                  dtype='int32')
        s[...] = numpy.array([image.shape for image in concat_data])
        d.dims.create_scale(s, 'shapes')
        d.dims[0].attach_scale(s)
        l = h5file.create_dataset('target_images_shapes_labels', (3,),
                                  dtype='S7')
        l[...] = ['height'.encode('utf8'), 'width'.encode('utf8'),
                  'channel'.encode('utf8')]
        d.dims.create_scale(l, 'shape_labels')
        d.dims[0].attach_scale(l)

        # (re)define split array
        split_array = numpy.empty(
            24,
            dtype=numpy.dtype([
                ('split', 'a', 12),
                ('source', 'a', 17),
                ('start', numpy.int64, 1),
                ('stop', numpy.int64, 1),
                ('indices', h5py.special_dtype(ref=h5py.Reference)),
                ('available', numpy.bool, 1),
                ('comment', 'a', 1)]))

        for i, el in enumerate(h5file.attrs['split']):
            split_array[i] = tuple(el)

        split_array[15] = ('train'.encode('utf8'),
                           'color_images'.encode('utf8'), 0,
                           len(train[0]), h5py.Reference(), True,
                           '.'.encode('utf8'))
        split_array[16] = ('valid'.encode('utf8'),
                           'color_images'.encode('utf8'), len(train[0]),
                           len(valid[0]), h5py.Reference(), True,
                           '.'.encode('utf8'))
        split_array[17] = ('test'.encode('utf8'),
                           'color_images'.encode('utf8'), len(valid[0]),
                           len(test[0]), h5py.Reference(), True,
                           '.'.encode('utf8'))
        split_array[18] = ('train'.encode('utf8'),
                           'gray_images'.encode('utf8'), 0,
                           len(train[1]), h5py.Reference(), True,
                           '.'.encode('utf8'))
        split_array[19] = ('valid'.encode('utf8'),
                           'gray_images'.encode('utf8'), len(train[1]),
                           len(valid[1]), h5py.Reference(), True,
                           '.'.encode('utf8'))
        split_array[20] = ('test'.encode('utf8'),
                           'gray_images'.encode('utf8'), len(valid[1]),
                           len(test[1]), h5py.Reference(), True,
                           '.'.encode('utf8'))
        split_array[21] = ('train'.encode('utf8'),
                           'targets'.encode('utf8'), 0,
                           len(train[2]), h5py.Reference(), True,
                           '.'.encode('utf8'))
        split_array[22] = ('valid'.encode('utf8'),
                           'targets'.encode('utf8'), len(train[2]),
                           len(valid[2]), h5py.Reference(), True,
                           '.'.encode('utf8'))
        split_array[23] = ('test'.encode('utf8'),
                           'targets'.encode('utf8'), len(valid[2]),
                           len(test[2]), h5py.Reference(), True,
                           '.'.encode('utf8'))
        h5file.attrs['split'] = split_array

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the Weizmann Horse Dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `mnist` command.

    """
    subparser.add_argument(
        "--output_filename", help="Name of the saved dataset. Defaults to " +
        "None, in which case a name based on the parameters will be used.",
        type=str, default=None)
    subparser.add_argument(
        "--resize", help="If True, the images will be resized according to " +
        "resize_size. Either zero pad or crop should be True.",
        type=bool, default=False)
    subparser.add_argument(
        "--resize_size", help="The size to resize the images to. If not set " +
        "and resize is True, the mean image size will be used instead.",
        type=int, default=-1)
    subparser.add_argument(
        "--zero_pad", help="If resize is True and zero_pad is True, the " +
        "image will be resized so that the longest axis is equal to the " +
        "size specified in resize_size and the other axis will be " +
        "zero-padded. Cannot be True if crop is True.",
        type=bool, default=True)
    subparser.add_argument(
        "--crop", help="If resize is True and crop is True, the image will " +
        "be resized so that the shortest axis is equal to the size " +
        "specified in resize_size and the other axis will be cropped. " +
        "Cannot be True if zero_pad is True.",
        type=bool, default=False)
    subparser.add_argument(
        "--split", help="The size of the training and validation set, " +
        "expressed as a percentage wrt the size of the dataset.",
        type=int, default=[.44, .22])

    return convert_weizmann_horses


def read_weizmann_horses(filename, resize=False, resize_size=-1, zero_pad=True,
                         crop=False, split=[.44, .22], rng=None):
    """ Reads the Weizmann Horse Dataset compressed file and returns
    the split data and some statistic over the dataset.

    Parameters
    ----------
    filename : string
        The path to the tar file containing the dataset
    resize : boolean, optional
        If True, the images will be resized according to resize_size. Either
        zero pad or crop should be True.
    resize_size : list, optional
        The size to resize the images to. If not set and resize is True, the
        mean image size will be used instead.
    zero_pad : boolean, optional
        If resize is True and zero_pad is True, the image will be resized so
        that the longest axis is equal to the size specified in resize_size and
        the other axis will be zero-padded. Cannot be True if crop is True.
    crop : boolean, optional
        If resize is True and crop is True, the image will be resized so
        that the shortest axis is equal to the size specified in resize_size
        and the other axis will be cropped. Cannot be True if zero_pad is True.
    split : list, optional
        The size of the training and validation set, expressed as a percentage
        wrt the size of the dataset.
    rng : numpy.random.RandomState, optional
        An initialized numpy pseudo-random number generator instance.

    Note: RGB images will be resized to the same size of the corresponding
    mask.
    """
    #############
    # LOAD DATA #
    #############
    if resize:
        assert zero_pad or crop
        assert not zero_pad or not crop

    if rng is None:
        numpy.random.RandomState(0xbeef)

    print "Extracting the data ..."
    base_path = os.path.dirname(os.path.abspath(filename))
    with tarfile.open(filename, 'r') as f:
        f.extractall(path=base_path)
    base_path = os.path.join(base_path, 'weizmann_horse_db')
    im_path = os.path.join(base_path, 'rgb')
    bw_im_path = os.path.join(base_path, 'gray')

    print "Processing the data ..."
    filenames = []
    for directory, _, images in os.walk(im_path):
        filenames.extend([im for im in images])
    filenames = sorted(filenames)

    if resize and resize_size is -1:
        print "Computing the mean image ..."
        # compute the mean height and mean width
        from operator import add
        resize_size = [0, 0]
        for f in filenames:
            mask = Image.open(os.path.join(base_path, 'figure_ground', f))
            resize_size = map(add, resize_size, mask.size)
        resize_size = [resize_size[0]/len(filenames),
                       resize_size[1]/len(filenames)]
        print('Image will be resized to: w={}, h={}').format(
            resize_size[0], resize_size[1])

    images = []
    bw_images = []
    masks = []
    cropped_px = 0
    for f in filenames:
        im = Image.open(os.path.join(im_path, f))
        bw_im = Image.open(os.path.join(bw_im_path, f))
        mask = Image.open(os.path.join(base_path, 'figure_ground', f))
        size = mask.size

        # RGB have different size than the mask..who knows why
        im = im.resize(size, Image.ANTIALIAS)
        assert im.size == bw_im.size

        if resize:
            ry, rx = resize_size
            if crop:
                # resize (keeping proportions)
                [y, x] = im.size
                dy = float(ry)/y
                dx = float(rx)/x
                ratio = max(dx, dy)
                y = int(y * ratio)
                x = int(x * ratio)

                im = im.resize((y, x), Image.ANTIALIAS)
                bw_im = bw_im.resize((y, x), Image.ANTIALIAS)
                mask = mask.resize((y, x), Image.NEAREST)

                # crop
                if y != ry:
                    excess = (y - ry) / 2
                    im = im.crop((excess, 0, ry+excess, rx))
                    bw_im = bw_im.crop((excess, 0, ry+excess, rx))
                    mask = mask.crop((excess, 0, ry+excess, rx))
                elif x != rx:
                    excess = (x - rx) / 2
                    im = im.crop((0, excess, ry, rx+excess))
                    bw_im = bw_im.crop((0, excess, ry, rx+excess))
                    mask = mask.crop((0, excess, ry, rx+excess))
                cropped_px += excess*2
            elif zero_pad:
                # resize (keeping proportions)
                [y, x] = im.size
                dy = float(ry)/y
                dx = float(rx)/x
                ratio = min(dx, dy)
                y = int(y * ratio)
                x = int(x * ratio)

                im = im.resize((y, x), Image.ANTIALIAS)
                bw_im = bw_im.resize((y, x), Image.ANTIALIAS)
                mask = mask.resize((y, x), Image.NEAREST)

                tmp = im
                im = Image.new("RGB", (ry, rx))
                im.paste(tmp, (0, 0))
                tmp = bw_im
                bw_im = Image.new("L", (ry, rx))
                bw_im.paste(tmp, (0, 0))
                tmp = mask
                mask = Image.new("L", (ry, rx))
                mask.paste(tmp, (0, 0))
            else:
                # resize (not keeping proportions)
                im = im.resize((ry, rx), Image.ANTIALIAS)
                bw_im = bw_im.resize((ry, rx), Image.ANTIALIAS)
                mask = mask.resize((ry, rx), Image.NEAREST)

            assert tuple(im.size) == tuple(resize_size)

        # damn dataset: mask is between 0 and 255...convert to binary!
        mask = mask.convert('1')
        # PIL/Numpy BUG! Convert to grayscale to avoid problems.
        mask = mask.convert('L')

        im = numpy.array(im).astype('float32') / 255.
        bw_im = numpy.array(bw_im).astype('float32') / 255.
        mask = numpy.array(mask).astype(numpy.int32) / 255

        assert 0. <= numpy.min(im) < 1.
        assert 0. <= numpy.min(bw_im) < 1.
        assert 0. <= numpy.min(mask) < 1
        assert 0. < numpy.max(im) <= 1.
        assert 0. < numpy.max(bw_im) <= 1.
        assert 0 < numpy.max(mask) <= 1
        assert numpy.min(im) != numpy.max(im)
        assert numpy.min(bw_im) != numpy.max(bw_im)
        assert numpy.min(mask) != numpy.max(mask)

        images.append(im)
        bw_images.append(bw_im)
        masks.append(mask)

    if crop:
        print "Cropped pixels: {}".format(cropped_px)
    images = numpy.asarray(images)
    bw_images = numpy.asarray(bw_images)
    masks = numpy.asarray(masks)

    ntot = len(images)
    ntrain = int(ntot*split[0])
    nvalid = ntrain + int(ntot*split[1])

    # split datasets
    train = (numpy.array(images[:ntrain]),
             numpy.array(bw_images[:ntrain]),
             numpy.array(masks[:ntrain]))
    valid = (numpy.array(images[ntrain:nvalid]),
             numpy.array(bw_images[ntrain:nvalid]),
             numpy.array(masks[ntrain:nvalid]))
    test = (numpy.array(images[nvalid:]),
            numpy.array(bw_images[nvalid:]),
            numpy.array(masks[nvalid:]))

    print "Computing the dataset statistics ..."
    if resize:
        # we can compute per pixel statistics
        mean = [train[0].mean(axis=0),
                valid[0].mean(axis=0),
                test[0].mean(axis=0)]
        bw_mean = [train[1].mean(axis=0),
                   valid[1].mean(axis=0),
                   test[1].mean(axis=0)]
        std = [numpy.maximum(train[0].std(axis=0), 1e-08),
               numpy.maximum(valid[0].std(axis=0), 1e-08),
               numpy.maximum(test[0].std(axis=0), 1e-08)]
        bw_std = [numpy.maximum(train[1].std(axis=0), 1e-08),
                  numpy.maximum(valid[1].std(axis=0), 1e-08),
                  numpy.maximum(test[1].std(axis=0), 1e-08)]
    else:

        def compute_stats(data, color):
            """Computes the mean and the standard deviation over images with
            different shapes.

            Parameters
            ----------
            data : array
                An "b01" or "b01c" array of images
            color : bool
                True if data are RGB images, False else

            Return
            ------
            mean : int
                The mean of data
            std : int
                The standard deviation of data
            """
            # create a masked array
            max_size = [0, 0]
            for el in data:
                max_size = map(max, max_size, el.shape)
            arr = numpy.ma.empty([data.shape[0]] + max_size)
            arr.mask = True
            # fill masked array
            for i, el in enumerate(data):
                arr[i, :el.shape[0], :el.shape[1], ...] = el
            return arr.mean(), arr.std()

        mean = []
        std = []
        bw_mean = []
        bw_std = []

        def extend_stat(mean, std, val):
            return mean.append(val[0]), std.append(val[1])

        extend_stat(mean, std, compute_stats(train[0], True))
        extend_stat(mean, std, compute_stats(valid[0], True))
        extend_stat(mean, std, compute_stats(test[0], True))

        # bw
        extend_stat(bw_mean, bw_std, compute_stats(train[1], True))
        extend_stat(bw_mean, bw_std, compute_stats(valid[1], True))
        extend_stat(bw_mean, bw_std, compute_stats(test[1], True))
    print "load_data Done!"
    print('Tot images:{} Train:{}->{} Valid:{}->{} Test:{}->{}').format(
        ntot, 0, ntrain-1, ntrain, nvalid-1, nvalid,
        nvalid+len(test[0])-1)

    shutil.rmtree(base_path)

    return (train, valid, test, mean, bw_mean, std, bw_std,
            [filenames[:ntrain], filenames[ntrain:nvalid], filenames[nvalid:]])
