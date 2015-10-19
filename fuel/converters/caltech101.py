import os
import tarfile
import h5py

from scipy.misc import imread

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

    input_file = '101_ObjectCategories.tar.gz'.format(size)
    input_file = os.path.join(directory, input_file)

    if not os.path.isfile(input_file):
        raise MissingInputFiles('Required files missing', [input_file])

    tar_file = tarfile.open(input_file, 'r:gz')

    with h5py.File(output_file, mode="w") as h5file:
        train_features = numpy.empty((len(CATEGORIES) * 30, 3, 256, 256), dtype='uint8')
        test_features = numpy.empty((len(CATEGORIES) * 10, 3, 256, 256), dtype='uint8')

        for i, c in enumerate(CATEGORIES):
            for j in xrange(30):
                imfile = 'image_{:04d}.jpg'.format(j)
                im = scipy.imresize(scipy.misc.imread(imfile), (256, 256))
                train_features[i * 30 + j] = numpy.rollaxis(im, 2, 0)
            for j in xrange(10):
                imfile = 'image_{:04d}.jpg'.format(j + 30)
                im = scipy.imresize(scipy.misc.imread(imfile), (256, 256))
                test_features[i * 10 + j] = numpy.rollaxis(im, 2, 0)

        train_targets = numpy.repeat(numpy.arange(len(CATEGORIES)), 30).reshape(-1, 1)
        test_targets = numpy.repeat(numpy.arange(len(CATEGORIES)), 10).reshape(-1, 1)

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
