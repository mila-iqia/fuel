from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path


class YouTubeAudio(H5PYDataset):
    def __init__(self, youtube_id, **kwargs):
        super(YouTubeAudio, self).__init__(
            file_or_path=find_in_data_path('{}.hdf5'.format(youtube_id)),
            which_sets=('train',), **kwargs
        )
