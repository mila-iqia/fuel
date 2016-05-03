import io
import os
import tarfile
import linecache
import shutil

import h5py
import numpy
from PIL import Image

from fuel.converters.base import progress_bar, check_exists
from fuel.datasets.hdf5 import H5PYDataset

DISTRIBUTION_FILE = 'mjsynth.tar.gz'
TAR_PATH = 'mnt/ramdisk/max/90kDICT32px'
TRAIN_SIZE = 7224586
TEST_SIZE = 891924
VAL_SIZE = 802731
BAD = {'train': ['./2194/2/334_EFFLORESCENT_24742.jpg',
                 './2128/2/369_REDACTED_63458.jpg',
                 './2069/4/192_whittier_86389.jpg',
                 './2025/2/364_SNORTERS_72304.jpg',
                 './2013/2/370_refract_63890.jpg',
                 './1881/4/225_Marbling_46673.jpg',
                 './1863/4/223_Diligently_21672.jpg',
                 './1817/2/363_actuating_904.jpg',
                 './1730/2/361_HEREON_35880.jpg',
                 './1696/4/211_Queened_61779.jpg',
                 './1650/2/355_stony_74902.jpg',
                 './1332/4/224_TETHERED_78397.jpg',
                 './936/2/375_LOCALITIES_44992.jpg',
                 './913/4/231_randoms_62372.jpg',
                 './905/4/234_Postscripts_59142.jpg',
                 './869/4/234_TRIASSIC_80582.jpg',
                 './627/6/83_PATRIARCHATE_55931.jpg',
                 './596/2/372_Ump_81662.jpg',
                 './554/2/366_Teleconferences_77948.jpg',
                 './495/6/81_MIDYEAR_48332.jpg',
                 './429/4/208_Mainmasts_46140.jpg',
                 './384/4/220_bolts_8596.jpg',
                 './368/4/232_friar_30876.jpg',
                 './275/6/96_hackle_34465.jpg',
                 './264/2/362_FORETASTE_30276.jpg',
                 './173/2/358_BURROWING_10395.jpg'],
       'test': ['./2911/6/77_heretical_35885.jpg',
                './2852/6/60_TOILSOME_79481.jpg',
                './2749/6/101_Chided_13155.jpg', ],
       'val': ['./2557/2/351_DOWN_23492.jpg',
               './2540/4/246_SQUAMOUS_73902.jpg',
               './2489/4/221_snored_72290.jpg']}


@check_exists(required_files=[DISTRIBUTION_FILE])
def convert_mjsynth(directory, output_directory,
                    output_filename='mjsynth.hdf5'):
    """Converts the MJSynth dataset to HDF5.

    Converts the MJSynth dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.MJSynth`. The converted dataset is saved as
    'mjsynth.hdf5'.

    It assumes the existence of the following file:

    * `mjsynth.tar.gz`

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'mjsynth.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    num_examples = TRAIN_SIZE + TEST_SIZE + VAL_SIZE
    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')
    dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    hdf_features = h5file.create_dataset('features', (num_examples,),
                                         dtype=dtype)
    hdf_features_shapes = h5file.create_dataset('features_shapes',
                                                (num_examples, 3),
                                                dtype='int32')
    dtype = h5py.special_dtype(vlen=numpy.dtype('S1'))
    hdf_labels = h5file.create_dataset('targets', (num_examples,), dtype=dtype)
    hdf_labels_shapes = h5file.create_dataset('targets_shapes',
                                              (num_examples, 1),
                                              dtype='int32')

    # Attach shape annotations and scales
    hdf_features.dims.create_scale(hdf_features_shapes, 'shapes')
    hdf_features.dims[0].attach_scale(hdf_features_shapes)

    hdf_labels.dims.create_scale(hdf_labels_shapes, 'shapes')
    hdf_labels.dims[0].attach_scale(hdf_labels_shapes)

    hdf_shapes_labels = h5file.create_dataset('features_shapes_labels',
                                              (3,), dtype='S7')
    hdf_shapes_labels[...] = ['channel'.encode('utf8'),
                              'height'.encode('utf8'),
                              'width'.encode('utf8')]
    hdf_features.dims.create_scale(hdf_shapes_labels, 'shape_labels')
    hdf_features.dims[0].attach_scale(hdf_shapes_labels)

    hdf_shapes_labels = h5file.create_dataset('targets_shapes_labels',
                                              (1,), dtype='S5')

    hdf_shapes_labels[...] = ['index'.encode('utf8')]
    hdf_labels.dims.create_scale(hdf_shapes_labels, 'shape_labels')
    hdf_labels.dims[0].attach_scale(hdf_shapes_labels)

    # Add axis annotations
    hdf_features.dims[0].label = 'batch'
    hdf_labels.dims[0].label = 'batch'

    # Extract the temp files from TAR
    if not os.path.exists(os.path.join(output_directory, TAR_PATH)):
        input_file = os.path.join(directory, DISTRIBUTION_FILE)
        with progress_bar('tar', os.path.getsize(input_file),
                          prefix='Extracting') as bar:
            class ProgressFileObject(io.FileIO):
                def read(self, size=-1):
                    bar.update(self.tell())
                    return io.FileIO.read(self, size)

            tar_file = tarfile.open(fileobj=ProgressFileObject(input_file))
            tar_file.extractall(path=output_directory)
            tar_file.close()

    # Convert
    i = 0
    for split, split_size in zip(['train', 'test', 'val'],
                                 [TRAIN_SIZE, TEST_SIZE, VAL_SIZE]):
        annotation_file = os.path.join(output_directory, TAR_PATH,
                                       'annotation_%s.txt' % split)
        # Convert from JPEG to NumPy arrays
        with progress_bar(split, split_size) as bar:
            for line in open(annotation_file):
                # Save image
                filename, word_index = line.split()
                if filename not in BAD[split]:
                    image = numpy.array(
                        Image.open(
                            os.path.join(output_directory, TAR_PATH,
                                         filename[2:])))

                    image = image.transpose(2, 0, 1)
                    hdf_features[i] = image.flatten()
                    hdf_features_shapes[i] = image.shape

                    # Read word from lexicon without '\n'
                    word = list(linecache.getline(
                        os.path.join(output_directory, TAR_PATH,
                                     'lexicon.txt'),
                        int(word_index))[:-1])
                    hdf_labels[i] = numpy.array(word)
                    hdf_labels_shapes[i] = len(word)

                    if i % 1000 == 0:
                        h5file.flush()
                    # Update progress
                    i += 1
                    value = i if split == 'train' else \
                        i - TRAIN_SIZE if split == 'test' else \
                            i - TRAIN_SIZE - TEST_SIZE
                    bar.update(value)

    # Add the labels
    split_dict = {}
    sources = ['features', 'targets']
    split_dict['train'] = dict(zip(sources, [(0, TRAIN_SIZE)] * 2))
    split_dict['test'] = dict(
        zip(sources, [(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE)] * 2))
    split_dict['val'] = dict(zip(sources, [
        (TRAIN_SIZE + TEST_SIZE, TRAIN_SIZE + TEST_SIZE + VAL_SIZE)] * 2))
    h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    h5file.flush()
    h5file.close()

    # Remove the extracted temp files
    shutil.rmtree(os.path.join(directory, TAR_PATH.split('/')[0]))

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the MJSynth dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `mjsynth` command.

    """
    return convert_mjsynth
