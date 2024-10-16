import ray
import time

class DataReader:
    def __init__(self, address='auto', queue_name="shared_queue", namespace='my'):
        self.address = address
        self.queue_name = queue_name
        self.namespace = namespace
        self._queue = None

    def connect(self):
        if not ray.is_initialized():
            try:
                ray.init(address=self.address)
            except Exception as e:
                print(f"Error initializing Ray: {e}")
                raise

        try:
            self._queue = ray.get_actor(self.queue_name, namespace=self.namespace)
        except Exception as e:
            print(f"Error getting queue actor: {e}")
            self.close()
            raise

    def close(self):
        if ray.is_initialized():
            ray.shutdown()
        self._queue = None

    def read(self):
        if self._queue is None:
            raise RuntimeError("DataReader is not connected. Call connect() first.")
        try:
            return ray.get(self._queue.get.remote())
        except ray.exceptions.RayActorError as e:
            raise DataReaderError("Queue actor is dead.") from e

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class DataReaderError(Exception):
    """Custom exception for DataReader errors."""
    pass
