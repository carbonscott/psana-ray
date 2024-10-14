import argparse
import time
import signal
import ray

from .shared_queue import create_queue
from psana_wrapper import PsanaWrapperSmd, ImageRetrievalMode

from mpi4py import MPI

def parse_arguments():
    parser = argparse.ArgumentParser(description="PsanaWrapper Data Producer")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--run", type=int, required=True, help="Run number")
    parser.add_argument("--detector_name", type=str, required=True, help="Detector name")
    parser.add_argument("--queue_size", type=int, default=100, help="Maximum queue size")
    return parser.parse_args()

def initialize_ray(queue_size, rank, max_retries=10, retry_delay=1):
    comm = MPI.COMM_WORLD

    try:
        ray.init(address='auto', namespace='my')
        print(f"Rank {rank}: Ray initialized successfully.")

        if rank == 0:
            queue = create_queue(maxsize=queue_size)
            print(f"Rank {rank}: Shared queue created successfully.")
            # Signal that the queue has been created
            comm.Barrier()
        else:
            print(f"Rank {rank}: Waiting for shared queue to be created...")
            # Wait for rank 0 to create the queue
            comm.Barrier()

        # All ranks try to get the queue
        retries = 0
        while retries < max_retries:
            try:
                queue = ray.get_actor("shared_queue")
                print(f"Rank {rank}: Successfully connected to shared queue.")
                return queue
            except ValueError:  # Actor not found
                retries += 1
                print(f"Rank {rank}: Attempt {retries}/{max_retries} to connect to shared queue failed. Retrying...")
                time.sleep(retry_delay)

        raise TimeoutError(f"Rank {rank}: Timeout waiting for shared_queue actor")

    except Exception as e:
        print(f"Rank {rank}: Error in initialize_ray: {e}")
        return None

def signal_handler(sig, frame):
    print("Ctrl+C pressed. Shutting down...")
    ray.shutdown()
    exit(0)

def produce_data(psana_wrapper, queue, rank, size):
    comm = MPI.COMM_WORLD
    for idx, data in enumerate(psana_wrapper.iter_events(mode=ImageRetrievalMode.image)):
        while True:
            try:
                success = ray.get(queue.put.remote([rank, idx, data]))
                if success:
                    print(f"Rank {rank} produced: idx={idx} | shape={data.shape}")
                    break  # Break the while loop and move to the next event
                else:
                    print(f"Rank {rank}: Queue is full, waiting...")
                    time.sleep(1)  # Consider implementing a more sophisticated backoff strategy
            except ray.exceptions.RayActorError:
                print(f"Rank {rank}: Queue actor is dead. Exiting...")
                return  # Exit the function if the queue actor is dead
            except Exception as e:
                print(f"Rank {rank}: Error in produce_data: {e}")
                time.sleep(1)  # Wait before retrying

    # Signal completion
    comm.Barrier()
    if rank == 0:
        # Put a sentinel value to signal end of data
        try:
            ray.get(queue.put.remote(None))
            print("Rank 0: Sentinel value sent successfully")
        except ray.exceptions.RayActorError:
            print("Rank 0: Queue actor is dead. Unable to send sentinel.")
        except Exception as e:
            print(f"Rank 0: Error putting sentinel value: {e}")

def main():
    args = parse_arguments()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        signal.signal(signal.SIGINT, signal_handler)

    queue = initialize_ray(args.queue_size, rank)
    if queue is None:
        MPI.Finalize()
        return

    psana_wrapper = PsanaWrapperSmd(
        exp=args.exp,
        run=args.run,
        detector_name=args.detector_name,
    )

    try:
        produce_data(psana_wrapper, queue, rank, size)
    except Exception as e:
        print(f"Rank {rank}: Unhandled exception in main: {e}")
    finally:
        if rank == 0:
            ray.shutdown()
        MPI.Finalize()

if __name__ == "__main__":
    main()
