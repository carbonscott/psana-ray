"""
Launch ray server on producer nodes:
    ray start --head --node-ip-address=127.0.0.1 --port=6379 --num-cpus=10 --block

Launch producer:
    mpirun -n 4 psana-ray-producer --exp mfxl1038923 --run 58 --detector_name epix10k2M --queue_size 400

Launch ray server on consumer nodes:
    ray start --address='<PRODUCER IP>:<PORT>'
    The ray server usually tells you which IP and port to connect to.  For example, ray start --address='172.24.48.144:6379'

Launch consumer:
    python psana_consumer.py 1

Stop ray server on consumer
    ray stop
"""

import time
import sys
import signal
from psana_ray.data_reader import DataReader, DataReaderError

def signal_handler(sig, frame):
    print("Ctrl+C pressed. Shutting down...")
    sys.exit(0)

def consume_data(consumer_id):
    with DataReader() as reader:
        while True:
            try:
                # Try to get data from queue
                result = reader.read()
                if result is not None:
                    rank, idx, data = result
                    # Process data
                    print(f"Consumer {consumer_id} processed: rank={rank} | idx={idx} | shape={data.shape}")
                else:
                    print(f"Consumer {consumer_id} waiting for data...")
                    time.sleep(1)
            except DataReaderError as e:
                print(f"DataReader error: {e}")
                print("Queue actor is dead. Exiting...")
                break
            except Exception as e:
                print(f"Error in consume_data: {e}")
                time.sleep(1)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    consumer_id = sys.argv[1] if len(sys.argv) > 1 else 1
    try:
        consume_data(consumer_id)
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
