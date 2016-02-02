import os
import subprocess
import sys

import h5py
import scipy.io.wavfile

from fuel.converters.base import fill_hdf5_file


def convert_youtube_audio(directory, output_directory, youtube_id, channels,
                          sample, output_filename=None):
    """Converts downloaded YouTube audio to HDF5 format.

    Requires `ffmpeg` to be installed and available on the command line
    (i.e. available on your `PATH`).

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    youtube_id : str
        11-character video ID (taken from YouTube URL)
    channels : int
        The number of audio channels to use in the PCM Wave file.
    sample : int
        The sampling rate to use in Hz, e.g. 44100 or 16000.
    output_filename : str, optional
        Name of the saved dataset. If `None` (the default),
        `youtube_id.hdf5` is used.

    """
    input_file = os.path.join(directory, '{}.m4a'.format(youtube_id))
    wav_filename = '{}.wav'.format(youtube_id)
    wav_file = os.path.join(directory, wav_filename)
    ffmpeg_not_available = subprocess.call(['ffmpeg', '-version'])
    if ffmpeg_not_available:
        raise RuntimeError('conversion requires ffmpeg')
    subprocess.check_call(['ffmpeg', '-y', '-i', input_file, '-ac',
                           str(channels), '-ar', str(sample), wav_file],
                          stdout=sys.stdout)

    # Load WAV into array
    _, data = scipy.io.wavfile.read(wav_file)
    if data.ndim == 1:
        data = data[:, None]
    data = data[None, :]

    # Store in HDF5
    if output_filename is None:
        output_filename = '{}.hdf5'.format(youtube_id)
    output_file = os.path.join(output_directory, output_filename)

    with h5py.File(output_file, 'w') as h5file:
        fill_hdf5_file(h5file, (('train', 'features', data),))
        h5file['features'].dims[0].label = 'batch'
        h5file['features'].dims[1].label = 'time'
        h5file['features'].dims[2].label = 'feature'

    return (output_file,)


def fill_subparser(subparser):
    """Sets up a subparser to convert YouTube audio files.

    Adds the compulsory `--youtube-id` flag as well as the optional
    `sample` and `channels` flags.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `youtube_audio` command.

    """
    subparser.add_argument(
        '--youtube-id', type=str, required=True,
        help=("The YouTube ID of the video from which to extract audio, "
              "usually an 11-character string.")
    )
    subparser.add_argument(
        '--channels', type=int, default=1,
        help=("The number of audio channels to convert to. The default of 1"
              "means audio is converted to mono.")
    )
    subparser.add_argument(
        '--sample', type=int, default=16000,
        help=("The sampling rate in Hz. The default of 16000 is "
              "significantly downsampled compared to normal WAVE files; "
              "pass 44100 for the usual sampling rate.")
    )
    return convert_youtube_audio
