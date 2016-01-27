try:
    import pafy
    PAFY_AVAILABLE = True
except ImportError:
    PAFY_AVAILABLE = False


def download(directory, youtube_id, clear=False):
    if not PAFY_AVAILABLE:
        raise ImportError("pafy is required to download YouTube videos")
    url = 'https://www.youtube.com/watch?v={}'.format(youtube_id)
    video = pafy.new(url)
    audio = video.getbestaudio()
    filepath = '{}.m4a'.format(youtube_id)
    audio.download(quiet=False, filepath=filepath)


def fill_subparser(subparser):
    subparser.add_argument(
        '--youtube-id', type=str, required=True,
        help=("The YouTube ID of the video from which to extract audio, "
              "usually an 11-character string.")
    )
    return download
