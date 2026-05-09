import os
import argparse


pool_resources = {
    "groot-h100-01": "-p groot-h100-01 -n {num_nodes}",
    "groot-h100-02": "-p groot-h100-02 -n {num_nodes}",
}

TRAIN_TEMPLATE = r"""
gear fast \
    {resources} \
    -i nvcr.io/nvidian/gear-trinity-train:latest \
    --name {job_name} \
    --env-var SEED=42 \
    "bash {workdir}/{launch_script} {config_index}"
"""


def launch_jobs(
    job_name: str,
    pool: str,
    num_nodes=8,
    script: str = "launch.sh",
    config_index: int = 0,
):
    workdir = os.path.dirname(os.path.abspath(__file__))
    resources = pool_resources[pool].format(num_nodes=num_nodes)
    launch_cmd = TRAIN_TEMPLATE.format(
        resources=resources,
        workdir=workdir,
        job_name=job_name,
        launch_script=script,
        config_index=config_index,
    )

    print(launch_cmd)
    input("Press Enter to launch...")
    os.system(launch_cmd)


if __name__ == "__main__":
    job_name = "iwm_training"

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pool", type=str, default="groot-h100-01", choices=list(pool_resources.keys()))
    parser.add_argument("-n", "--num_nodes", type=int, default=4, help="Number of nodes to use")
    parser.add_argument("-c", "--config_index", type=int, default=0, help="Config index to launch")
    parser.add_argument("-r", "--resume_ckpt", type=str, default=None, help="Resume checkpoint")
    args = parser.parse_args()

    launch_jobs(
        job_name,
        pool=args.pool,
        num_nodes=args.num_nodes,
        script="launch.sh",
        config_index=args.config_index,
    )
