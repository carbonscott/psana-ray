import ray
from collections import deque

@ray.remote
class Queue:
    def __init__(self, maxsize=100):
        self.items = deque(maxlen=maxsize)

    def put(self, item):
        try:
            if len(self.items) < self.items.maxlen:
                self.items.append(item)
                return True
            return False
        except Exception as e:
            print(f"Error in put: {e}")
            return False

    def get(self):
        try:
            return self.items.popleft() if self.items else None
        except Exception as e:
            print(f"Error in get: {e}")
            return None

    def size(self):
        try:
            return len(self.items)
        except Exception as e:
            print(f"Error in size: {e}")
            return 0

def create_queue(name="shared_queue", namespace="default", maxsize=100):
    try:
        return Queue.options(name=name, namespace=namespace).remote(maxsize=maxsize)
    except Exception as e:
        print(f"Error creating queue '{name}' in namespace '{namespace}': {e}")
        return None
