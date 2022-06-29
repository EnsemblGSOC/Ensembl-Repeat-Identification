"""
Submit a cluster LSF job to train a neural network.

e.g.:
    python submit_cluster_job.py --pipeline train.py --configuration configuration.yaml

    python submit_cluster_job.py --gpu V100 --pipeline train.py --configuration configuration.yaml
"""


# standard library imports
import argparse
import datetime as dt
import pathlib
import shutil
import subprocess
import sys

# third party imports
import yaml

from pytorch_lightning.utilities import AttributeDict


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--pipeline", type=str, help="pipeline script path")
    argument_parser.add_argument(
        "--configuration", type=str, help="experiment configuration file path"
    )
    argument_parser.add_argument(
        "--gpu",
        default="A100",
        choices=["A100", "V100"],
        type=str,
        help="GPU nodes queue to submit job",
    )
    argument_parser.add_argument(
        "--num_gpus", default=1, type=int, help="number of GPUs to use"
    )
    argument_parser.add_argument(
        "--mem_limit", default=65536, type=int, help="GPU node RAM memory limit"
    )

    args = argument_parser.parse_args()

    # submit new training job
    if args.pipeline and args.configuration:
        datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")

        pipeline_path = pathlib.Path(args.pipeline)

        with open(args.configuration) as file:
            configuration = yaml.safe_load(file)
        configuration = AttributeDict(configuration)

        experiment_name = f"{configuration.experiment_prefix}_{datetime}"

        experiments_directory = configuration.save_directory

        experiment_directory = pathlib.Path(
            f"{experiments_directory}/{experiment_name}"
        )
        experiment_directory.mkdir(parents=True, exist_ok=True)

        # save configuration file to experiment directory
        configuration_copy = shutil.copy(args.configuration, experiment_directory)

        pipeline_command_elements = [
            f"python {pipeline_path}",
            # f"--configuration {configuration_copy}",
        ]

    # no task specified
    else:
        print(__doc__)
        argument_parser.print_help()
        sys.exit()

    pipeline_command = " ".join(pipeline_command_elements)

    # common job arguments
    bsub_command_elements = [
        "bsub",
        f"-M {args.mem_limit}",
        f"-o {experiment_directory}/stdout.log",
        f"-e {experiment_directory}/stderr.log",
    ]

    # run training on an NVIDIA A100 GPU node
    if args.gpu == "A100":
        gpu_memory = 81000  # ~80 GiBs, NVIDIA A100 memory with safety margin
        # gpu_memory = 81920  # 80 GiBs, total NVIDIA A100 memory

        mem_limit = args.mem_limit
        bsub_command_elements.append("-q gpu-a100")

    # run training on an NVIDIA V100 GPU node
    elif args.gpu == "V100":
        gpu_memory = 32256  # 31.5 GiBs, NVIDIA V100 memory with safety margin
        # gpu_memory = 32510  # ~32 GiBs, total NVIDIA V100 memory

        mem_limit = 32768
        bsub_command_elements.append("-q gpu")

    bsub_command_elements.extend(
        [
            f'-gpu "num={args.num_gpus}:gmem={gpu_memory}:j_exclusive=yes"',
            f"-M {mem_limit}",
            f'-R"select[mem>{mem_limit}] rusage[mem={mem_limit}] span[hosts=1]"',
            pipeline_command,
        ]
    )

    bsub_command = " ".join(bsub_command_elements)
    print(f"running command:\n{bsub_command}")

    subprocess.run(bsub_command, shell=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
