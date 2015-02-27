import os.path
from pkg_resources import get_distribution, DistributionNotFound

from fuel.config_parser import config  # noqa

try:
    DIST = get_distribution('fuel')
    DIST_LOC = os.path.normcase(DIST.location)
    HERE = os.path.normcase(__file__)
    if not HERE.startswith(os.path.join(DIST_LOC, 'fuel')):
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'not installed'
else:
    __version__ = DIST.version
