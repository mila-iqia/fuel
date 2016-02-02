from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path


class YouTubeAudio(H5PYDataset):
    r"""Dataset of audio from YouTube video.

    Assumes the existence of a dataset file with the name
    `youtube_id.hdf5`. These datasets don't have any split; the entire
    audio sequence is considered training.

    Note that the data structured in the form `(batch, time, features)`
    where `features` are the audio channels (dimension 1 or 2) and batch is
    equal to 1 in this case (since there is only one audiotrack).

    Parameters
    ----------
    youtube_id : str
        11-character video ID (taken from YouTube URL)
    \*\*kwargs
        Passed to the `H5PYDataset` class.

    """
    def __init__(self, youtube_id, **kwargs):
        super(YouTubeAudio, self).__init__(
            file_or_path=find_in_data_path('{}.hdf5'.format(youtube_id)),
            which_sets=('train',), **kwargs
        )
