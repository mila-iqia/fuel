import os
import h5py

import numpy
import scipy.misc

from six.moves import range

from fuel.converters.base import fill_hdf5_file, MissingInputFiles

CATEGORIES = (
    'Leopards',
    'emu',
    'hedgehog',
    'binocular',
    'cougar_body',
    'buddha',
    'Faces_easy',
    'beaver',
    'windsor_chair',
    'yin_yang',
    'anchor',
    'pagoda',
    'mayfly',
    'flamingo_head',
    'headphone',
    'joshua_tree',
    'wrench',
    'platypus',
    'dollar_bill',
    'dalmatian',
    'mandolin',
    'llama',
    'electric_guitar',
    'panda',
    'lamp',
    'pyramid',
    'kangaroo',
    'strawberry',
    'stop_sign',
    'flamingo',
    'gerenuk',
    'crayfish',
    'ketch',
    'crocodile_head',
    'chandelier',
    'cellphone',
    'brain',
    'car_side',
    'ferry',
    'nautilus',
    'BACKGROUND_Google',
    'metronome',
    'water_lilly',
    'dolphin',
    'euphonium',
    'crocodile',
    'Faces',
    'sunflower',
    'garfield',
    'soccer_ball',
    'stapler',
    'scorpion',
    'wheelchair',
    'saxophone',
    'starfish',
    'lotus',
    'okapi',
    'octopus',
    'hawksbill',
    'chair',
    'crab',
    'menorah',
    'helicopter',
    'accordion',
    'rhino',
    'ant',
    'bass',
    'bonsai',
    'butterfly',
    'ewer',
    'pizza',
    'umbrella',
    'revolver',
    'airplanes',
    'grand_piano',
    'trilobite',
    'cannon',
    'wild_cat',
    'inline_skate',
    'watch',
    'pigeon',
    'rooster',
    'cup',
    'dragonfly',
    'barrel',
    'ceiling_fan',
    'lobster',
    'minaret',
    'schooner',
    'cougar_face',
    'Motorbikes',
    'laptop',
    'elephant',
    'sea_horse',
    'snoopy',
    'brontosaurus',
    'gramophone',
    'camera',
    'stegosaurus',
    'tick',
    'scissors',
    'ibis')

NUM_TRAIN = 20
NUM_TEST = 10


def read_image(imfile):
    im = scipy.misc.imresize(scipy.misc.imread(imfile), (256, 256))
    if im.ndim == 2:
        return im.reshape(1, 256, 256)
    else:
        return numpy.rollaxis(im, 2, 0)


def convert_silhouettes(directory, output_directory,
                        output_file=None):
    """ Convert the CalTech 101 Datasets.

    Parameters
    ----------
    directory : str
        Directory in which the required input files reside.
    output_file : str
        Where to save the converted dataset.

    """
    if output_file is None:
        output_file = 'caltech101.hdf5'
    output_file = os.path.join(output_directory, output_file)

    input_dir = '101_ObjectCategories'
    input_dir = os.path.join(directory, input_dir)

    if not os.path.isdir(input_dir):
        raise MissingInputFiles('Required files missing', [input_dir])

    with h5py.File(output_file, mode="w") as h5file:
        train_features = numpy.empty(
            (len(CATEGORIES) * NUM_TRAIN, 3, 256, 256), dtype='uint8')
        test_features = numpy.empty((len(CATEGORIES) * NUM_TEST, 3, 256, 256),
                                    dtype='uint8')

        for i, c in enumerate(CATEGORIES):
            for j in range(NUM_TRAIN):
                imfile = os.path.join(input_dir, c,
                                      'image_{:04d}.jpg'.format(j + 1))
                train_features[i * NUM_TRAIN + j] = read_image(imfile)
            for j in range(NUM_TEST):
                imfile = os.path.join(
                    input_dir, c, 'image_{:04d}.jpg'.format(j + NUM_TRAIN + 1))
                test_features[i * NUM_TEST + j] = read_image(imfile)

        train_targets = numpy.repeat(numpy.arange(len(CATEGORIES)), NUM_TRAIN)
        train_targets = train_targets.reshape(-1, 1)
        test_targets = numpy.repeat(numpy.arange(len(CATEGORIES)), NUM_TEST)
        test_targets = test_targets.reshape(-1, 1)

        data = (
            ('train', 'features', train_features),
            ('train', 'targets', train_targets),
            ('test', 'features', test_features),
            ('test', 'targets', test_targets),
        )
        fill_hdf5_file(h5file, data)

        for i, label in enumerate(('batch', 'channel', 'height', 'width')):
            h5file['features'].dims[i].label = label

        for i, label in enumerate(('batch', 'index')):
            h5file['targets'].dims[i].label = label

    return (output_file,)


def fill_subparser(subparser):
    """Sets up a subparser to convert CalTech101 Silhouettes Database files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `caltech101_silhouettes` command.

    """
    return convert_silhouettes
