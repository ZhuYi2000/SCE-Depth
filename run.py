#!/usr/bin/env python3

import argparse
import datetime
import getpass
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from compute_environment import compute_environment
from sce_depth.utils import get_paths


def assert_mlflow_db_exists():
    db_path = get_paths.get_mlflow_db_path()
    os.makedirs(Path(db_path).parent, exist_ok=True)
    if not os.path.isfile(db_path):
        open(db_path, "w").close()


def env_prefix(env):
    if env == "local":
        return []
    if "singularity" in env:
        container_name = compute_environment.CONTAINER.singularity_container_name
        container_path = os.path.join(get_paths.get_container_path(), container_name)
        command = ["singularity", "exec", "--nv", "--nvccli"]
        if env == "singularity":
            command += ["--no-home"]
        command += ["--env", "PYTHONPATH=$PYTHONPATH:" + get_paths.get_base_path()]
        command += ["--env", "GUNICORN_CMD_ARGS='--timeout 180'"]
        if "MASTER_PORT" in os.environ:
            command += ["--env", "MASTER_PORT=" + os.environ["MASTER_PORT"]]
        if "MASTER_ADDR" in os.environ:
            command += ["--env", "MASTER_ADDR=" + os.environ["MASTER_ADDR"]]
        command += ["--env", "MPLCONFIGDIR=" + get_paths.get_mpl_cache_path()]
        for bind_path in get_paths.get_bind_paths():
            command += ["--bind", bind_path]
        command += [container_path]
        return command
    if env == "docker":
        command = [
            "docker",
            "run",
            "-it",
            "-u",
            f"{os.getuid()}:{os.getgid()}",
            "--network",
            "host",
        ]
        command += ["--env", "PYTHONPATH=$PYTHONPATH:" + get_paths.get_base_path()]
        if "MASTER_PORT" in os.environ:
            command += ["--env", "MASTER_PORT=" + os.environ["MASTER_PORT"]]
        if "MASTER_ADDR" in os.environ:
            command += ["--env", "MASTER_ADDR=" + os.environ["MASTER_ADDR"]]
        for bind_path in get_paths.get_bind_paths():
            command += ["--mount", "type=bind,src=" + bind_path + ",dst=" + bind_path]
        command += ["-w", get_paths.get_base_path()]
        command += ["sce_depth"]
        return command
    raise ValueError(f"Unknown environment: {env}")


def run_and_print(command, cwd=None):
    print(f"running: {' '.join(command)}")
    subprocess.run(command, cwd=cwd)


def mlf_server(run_args, sub_args):
    if sub_args:
        print(f"unknown arguments: {' '.join(sub_args)}")
        sys.exit(1)

    if run_args.backend == "filesystem":
        command = env_prefix(run_args.env) + ["mlflow", "server"]
        command += ["--backend-store-uri", "file://" + get_paths.get_mlruns_path()]
        command += ["--workers", str(run_args.workers)]
        command += ["--port", str(run_args.port)]
        run_and_print(command)
        return

    server_file = get_paths.get_tracking_server_file_path()
    if os.path.isfile(server_file):
        with open(server_file, "r", encoding="utf-8") as f:
            server_data = json.load(f)
        print(
            f"The tracking server is already running on {server_data['host']}:{server_data['port']} "
            + f"(started {server_data['start_time']} by {server_data['user']}). Aborting."
        )
        sys.exit(1)

    server_data = {
        "user": getpass.getuser(),
        "start_time": datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y"),
        "host": socket.gethostname(),
        "port": run_args.port,
        "workers": run_args.workers,
        "timeout": run_args.timeout,
    }
    with open(server_file, "w", encoding="utf-8") as f:
        json.dump(server_data, f)

    command = env_prefix(run_args.env) + ["mlflow", "server"]
    command += ["--backend-store-uri"]
    command += ["sqlite:///" + get_paths.get_mlflow_db_path() + "?timeout=" + str(run_args.timeout)]
    command += ["--default-artifact-root", "file://" + get_paths.get_mlruns_path()]
    command += ["--workers", str(run_args.workers)]
    command += ["--host", "0.0.0.0"]
    command += ["--port", str(run_args.port)]

    try:
        run_and_print(command)
    except KeyboardInterrupt:
        pass

    if os.path.isfile(server_file):
        os.remove(server_file)
        print(f"removed server file {server_file}")


def build_singularity(run_args, sub_args):
    if sub_args:
        print(f"unknown arguments: {' '.join(sub_args)}")
        sys.exit(1)
    containers_path = os.path.join(get_paths.get_base_path(), "containers")
    output_path = get_paths.get_container_path()
    container_name = compute_environment.CONTAINER.singularity_container_name
    output_file = os.path.join(output_path, container_name)
    command = ["singularity", "build"]
    if run_args.tmpdir is not None:
        command += ["--tmpdir", run_args.tmpdir]
    command += [output_file, "singularity_recipe"]
    run_and_print(command, cwd=containers_path)


def build_docker(run_args, sub_args):
    if sub_args:
        print(f"unknown arguments: {' '.join(sub_args)}")
        sys.exit(1)
    container_path = get_paths.get_container_path()
    command = ["docker", "build", "-t", "sce_depth", "."]
    run_and_print(command, cwd=container_path)


def bash(run_args, sub_args):
    command = env_prefix(run_args.env) + ["/bin/bash"] + sub_args
    run_and_print(command)


def train_isaac(run_args, sub_args):
    path = Path(__file__).parent.joinpath("sce_depth/train_isaac.py").absolute()
    command = env_prefix(run_args.env) + ["python3", "-u", str(path)] + sub_args
    run_and_print(command)


def inference_isaac(run_args, sub_args):
    path = Path(__file__).parent.joinpath("sce_depth/inference_isaac.py").absolute()
    command = env_prefix(run_args.env) + ["python3", "-u", str(path)] + sub_args
    run_and_print(command)


def main():
    compute_environment.inform()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", choices=["local", "singularity", "singularity_vscode", "docker"], default="local"
    )
    subparsers = parser.add_subparsers(dest="subparser_name")

    parser_bash = subparsers.add_parser("bash")
    parser_bash.set_defaults(func=bash)

    parser_mlf = subparsers.add_parser("start-mlflow-server")
    parser_mlf.add_argument("--port", type=int, default=5000)
    parser_mlf.add_argument("--workers", type=int, default=1)
    parser_mlf.add_argument("--timeout", type=int, default=30)
    parser_mlf.add_argument("--backend", type=str, choices=["sqlite", "filesystem"], default="sqlite")
    parser_mlf.set_defaults(func=mlf_server)

    parser_build_sin = subparsers.add_parser("build-singularity")
    parser_build_sin.add_argument("--tmpdir", type=str, default=None)
    parser_build_sin.set_defaults(func=build_singularity)

    parser_build_docker = subparsers.add_parser("build-docker")
    parser_build_docker.set_defaults(func=build_docker)

    parser_train = subparsers.add_parser("train_isaac")
    parser_train.set_defaults(func=train_isaac)

    parser_infer = subparsers.add_parser("inference_isaac")
    parser_infer.set_defaults(func=inference_isaac)

    run_args, sub_args = parser.parse_known_args()

    if not hasattr(run_args, "func"):
        parser.print_help()
        sys.exit(1)

    assert_mlflow_db_exists()
    run_args.func(run_args, sub_args)


if __name__ == "__main__":
    main()

