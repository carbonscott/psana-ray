import ray
import time
from contextlib import contextmanager

class DataReader:
    def __init__(self, address='auto', queue_name="shared_queue", namespace='my'):
        self.address = address
        self.queue_name = queue_name
        self.namespace = namespace
        self._queue = None

    def __enter__(self):
        try:
            ray.init(address=self.address)
            self._queue = ray.get_actor(self.queue_name, namespace=self.namespace)
        except Exception as e:
            print(f"Error initializing Ray or getting queue actor: {e}")
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ray.shutdown()

    def read(self):
        if self._queue is None:
            raise RuntimeError("DataReader is not initialized. Use with 'with' statement.")
        try:
            return ray.get(self._queue.get.remote())
        except ray.exceptions.RayActorError as e:
            raise DataReaderError("Queue actor is dead.") from e

class DataReaderError(Exception):
    """Custom exception for DataReader errors."""
    pass
