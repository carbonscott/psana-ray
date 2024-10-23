import argparse
import random
import time
import signal
import ray
import sys
import numpy as np

from .shared_queue import create_queue
from psana_wrapper import PsanaWrapperSmd, ImageRetrievalMode

from mpi4py import MPI

def parse_arguments():
    parser = argparse.ArgumentParser(description="PsanaWrapper Data Producer")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--run", type=int, required=True, help="Run number")
    parser.add_argument("--detector_name", type=str, required=True, help="Detector name")
    parser.add_argument("--calib", type=bool, default=True, help="Use calib mode")
    parser.add_argument("--uses_bad_pixel_mask", type=bool, default=True, help="Use bad pixel mask")
    parser.add_argument("--manual_mask_path", type=str, default=None, help="Path to a manual mask in npy")
    parser.add_argument("--ray_address", type=str, default="auto", help="Address of the Ray cluster")
    parser.add_argument("--ray_namespace", type=str, default="default", help="Ray namespace to use for both queues")
    parser.add_argument("--queue_name", type=str, default='my', help="Queue name")
    parser.add_argument("--queue_size", type=int, default=100, help="Maximum queue size")
    return parser.parse_args()

def initialize_ray(ray_address, ray_namespace, queue_name, queue_size, rank, max_retries=10, retry_delay=1):
    comm = MPI.COMM_WORLD

    try:
        ray.init(address=ray_address, namespace=ray_namespace)
        print(f"Rank {rank}: Ray initialized successfully.")

        if rank == 0:
            queue = create_queue(queue_name=queue_name, ray_namespace=ray_namespace, maxsize=queue_size)
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
                queue = ray.get_actor(queue_name)
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

def produce_data(psana_wrapper, psana_mode, queue, rank, size, uses_bad_pixel_mask=False, manual_mask_path=None):
    comm = MPI.COMM_WORLD

    bad_pixel_mask = psana_wrapper.create_bad_pixel_mask() if uses_bad_pixel_mask else None
    manual_pixel_mask = np.load(manual_mask_path) if manual_mask_path is not None else None

    # Delay for exponential backoff
    base_delay_in_sec = 0.1
    max_delay_in_sec  = 2.0

    for idx, (data, photon_energy) in enumerate(psana_wrapper.iter_events(mode=psana_mode)):
        if bad_pixel_mask is not None:
            data = np.where(bad_pixel_mask, data, 0)
        if manual_pixel_mask is not None:
            data = np.where(manual_pixel_mask, data, 0)
        retries = 0
        while True:
            try:
                success = ray.get(queue.put.remote([rank, idx, data, photon_energy]))
                if success:
                    print(f"Rank {rank} produced: idx={idx} | shape={data.shape} | photon_energy={photon_energy}")
                    break  # Break the while loop and move to the next event
                else:
                    print(f"Rank {rank}: Queue is full, waiting...")
                    # Use exponential backoff with jitter
                    delay_in_sec = min(max_delay_in_sec, base_delay_in_sec * (2 ** retries))
                    jitter_in_sec = random.uniform(0, 0.5)
                    time.sleep(delay_in_sec + jitter_in_sec)
                    if delay_in_sec < max_delay_in_sec: retries += 1
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

    queue = initialize_ray(args.ray_address, args.ray_namespace, args.queue_name, args.queue_size, rank)
    if queue is None:
        MPI.Finalize()
        return

    psana_wrapper = PsanaWrapperSmd(
        exp=args.exp,
        run=args.run,
        detector_name=args.detector_name,
    )

    psana_mode = dict(
        calib = ImageRetrievalMode.calib,
        image = ImageRetrievalMode.image,
    )['calib' if args.calib else 'image']

    try:
        produce_data(psana_wrapper, psana_mode, queue, rank, size, args.uses_bad_pixel_mask, args.manual_mask_path)
    except Exception as e:
        print(f"Rank {rank}: Unhandled exception in main: {e}")
    finally:
        if rank == 0:
            ray.shutdown()
        MPI.Finalize()

if __name__ == "__main__":
    main()
