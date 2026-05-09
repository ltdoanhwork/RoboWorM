import os
import re
import subprocess
import signal
import sys
import argparse
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


def train_func():
    # Launch bash script, but tie subprocess to same process group.
    # This ensures `ray job stop` will kill the subprocess as well.
    proc = subprocess.Popen(
        ["bash", "launch.sh"],
        shell=False,
        preexec_fn=os.setsid  # <<=== KEY: put subprocess in its own process group
    )

    def handle_sigterm(signum, frame):
        print("[train_func] Received SIGTERM, terminating subprocess...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # kill process group
        except ProcessLookupError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    proc.wait()  # Wait for subprocess to finish
    print("[train_func] Training finished or terminated.")


@ray.remote
class _TrainingWorker:
    def __init__(self, training_fn) -> None:
        self.training_fn = training_fn

    def get_address(self):
        from ray.train._internal.utils import get_address_and_port
        return get_address_and_port()

    def setup_env_vars(self, addr, port, node_rank, nnodes, world_size):
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)
        os.environ["NODE_RANK"] = str(node_rank)
        os.environ["NNODES"] = str(nnodes)
        os.environ["WORLD_SIZE"] = str(world_size)

    def train(self):
        # Handle termination inside the actor too
        def handle_sigterm(signum, frame):
            print("[_TrainingWorker] Caught SIGTERM, cleaning up.")
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_sigterm)
        signal.signal(signal.SIGINT, handle_sigterm)

        try:
            self.training_fn()
        except Exception as e:
            if "ECC" in str(e) or "CUDA driver initialization failed" in str(e):
                print(e)
                sys.exit(1)
            elif "trainer_state.json" in str(e):
                path_match = re.search(r"'([^']*trainer_state\.json)'", str(e))
                if path_match:
                    ckpt_dir = os.path.dirname(path_match.group(1))
                    subprocess.run(["rm", "-rf", ckpt_dir], check=False)
                raise
            else:
                raise


class TorchTrainer:
    def __init__(self, placement_group) -> None:
        self.placement_group = placement_group
        self.workers = []
        self.training_fn = None

    def setup_workers(self, training_fn):
        self.training_fn = training_fn
        self.workers = []

        for i, bundle in enumerate(self.placement_group.bundle_specs):
            worker = _TrainingWorker.options(
                num_cpus=bundle["CPU"],
                num_gpus=bundle["GPU"],
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.placement_group,
                    placement_group_bundle_index=i,
                ),
            ).remote(training_fn)
            self.workers.append(worker)

        if self.workers:
            head_node_addr, port = ray.get(self.workers[0].get_address.remote())
            ray.get([
                w.setup_env_vars.remote(head_node_addr, port, i, len(self.workers), len(self.workers) * 8)
                for i, w in enumerate(self.workers)
            ])

    def _cleanup_workers(self):
        for w in self.workers:
            try:
                ray.kill(w)
            except Exception:
                pass
        self.workers = []

    def train(self, training_fn):
        self.setup_workers(training_fn)
        ray.get([w.train.remote() for w in self.workers])

    def shutdown(self):
        ray.util.remove_placement_group(self.placement_group)


def main(n_nodes=8):
    ray.init()

    placement_group = ray.util.placement_group(
        [{"CPU": 80, "GPU": 8} for _ in range(n_nodes)]
    )
    trainer = TorchTrainer(placement_group)

    trainer.train(train_func)
    trainer.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_nodes", type=int, default=4)
    args = parser.parse_args()

    main(n_nodes=args.n_nodes)
