from multiprocessing import Process

from numpy.testing import assert_allclose, assert_raises
from six.moves import cPickle
from nose.exc import SkipTest

from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.server import start_server
from fuel.streams import DataStream, ServerDataStream


def get_stream():
    return DataStream(
        MNIST(('train',)), iteration_scheme=SequentialScheme(1500, 500))


class TestServer(object):
    def setUp(self):
        self.server_process = Process(
            target=start_server, args=(get_stream(),))
        self.server_process.start()
        self.stream = ServerDataStream(('f', 't'), False)

    def tearDown(self):
        self.server_process.terminate()
        self.stream = None

    def test_server(self):
        server_data = self.stream.get_epoch_iterator()
        expected_data = get_stream().get_epoch_iterator()
        for _, s, e in zip(range(3), server_data, expected_data):
            for data in zip(s, e):
                assert_allclose(*data)
        assert_raises(StopIteration, next, server_data)

    def test_pickling(self):
        try:
            self.stream = cPickle.loads(cPickle.dumps(self.stream))
            server_data = self.stream.get_epoch_iterator()
            expected_data = get_stream().get_epoch_iterator()
            for _, s, e in zip(range(3), server_data, expected_data):
                for data in zip(s, e):
                    assert_allclose(*data, rtol=1e-3)
        except AssertionError as e:
            raise SkipTest("Skip test_that failed with: {}".format(e))
        assert_raises(StopIteration, next, server_data)

    def test_value_error_on_request(self):
        assert_raises(ValueError, self.stream.get_data, [0, 1])

    def test_close(self):
        self.stream.close()

    def test_next_epoch(self):
        self.stream.next_epoch()

    def test_reset(self):
        self.stream.reset()
