from multiprocessing import Process

from numpy.testing import assert_allclose, assert_raises

from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.server import start_server
from fuel.streams import DataStream, ServerDataStream


def get_stream():
    mnist = MNIST('train')
    data_stream = DataStream(
        mnist, iteration_scheme=SequentialScheme(1500, 500)
    )
    return data_stream


def test_server():
    server_process = Process(target=start_server, args=(get_stream(),))
    server_process.start()
    try:
        server_data = ServerDataStream(('f', 't')).get_epoch_iterator()
        expected_data = get_stream().get_epoch_iterator()
        for _, s, e in zip(range(3), server_data, expected_data):
            for data in zip(s, e):
                assert_allclose(*data)
        assert_raises(StopIteration, next, server_data)
    finally:
        server_process.terminate()
