import os

import h5py
import scipy.io.wavfile

from fuel.converters.base import fill_hdf5_file


def convert_youtube_audio(directory, output_directory, youtube_id, channels,
                          sample, output_filename=None):
    input_file = os.path.join(directory, '{}.m4a'.format(youtube_id))
    wav_filename = '{}.wav'.format(youtube_id)
    wav_file = os.path.join(directory, wav_filename)
    command = "ffmpeg -y -i {} -ac {} -ar {} {}".format(
            input_file, channels, sample, wav_file)
    os.system(command)

    # Load WAV into array
    _, data = scipy.io.wavfile.read(wav_file)
    if data.ndim == 1:
        data = data[:, None]

    # Store in HDF5
    if output_filename is None:
        output_filename = '{}.hdf5'.format(youtube_id)
    output_file = os.path.join(output_directory, output_filename)

    with h5py.File(output_file, 'w') as h5file:
        fill_hdf5_file(h5file, (('train', 'features', data),))
        h5file['features'].dims[0].label = 'time'
        h5file['features'].dims[1].label = 'feature'

    return (output_file,)


def fill_subparser(subparser):
    subparser.add_argument(
        '--youtube-id', type=str, required=True,
        help=("The YouTube ID of the video from which to extract audio, "
              "usually an 11-character string.")
    )
    subparser.add_argument(
        '--channels', type=int, default=1,
        help="The number of audio channels to convert to"
    )
    subparser.add_argument(
        '--sample', type=int, default=16000,
        help="The sampling rate in Hz"
    )
    return convert_youtube_audio
