from multiprocessing import Process, Queue

from numpy.testing import assert_allclose

from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.server import broker, server
from fuel.streams import DataStream, ServerDataStream


def get_stream():
    mnist = MNIST('train')
    data_stream = DataStream(
        mnist, iteration_scheme=SequentialScheme(1500, 500)
    )
    return data_stream


def client(q):
    data_stream = ServerDataStream(('features', 'targets'))
    epoch = data_stream.get_epoch_iterator()
    for i in range(3):
        q.put(next(epoch))
    try:
        next(epoch)
        q.put(False)
    except StopIteration:
        q.put(StopIteration)


def start_server():
    server(get_stream(), 5560)


def test_server():
    q = Queue()
    client_process = Process(target=client, args=(q,))
    broker_process = Process(target=broker, args=(5560, 5559))
    server_process = Process(target=start_server)
    server_process.start()
    broker_process.start()
    client_process.start()
    try:
        for data in zip(q.get(), next(get_stream().get_epoch_iterator())):
            assert_allclose(*data)
        q.get()
        q.get()
        assert q.get() == StopIteration
    finally:
        server_process.terminate()
        broker_process.terminate()
        client_process.join()
