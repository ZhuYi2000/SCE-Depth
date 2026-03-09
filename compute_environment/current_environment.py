#!/usr/bin/env python

import os
from pathlib import Path
from compute_environment.compute_environment_config import ProjectPaths, Container, Logging

ROOT_DIR = Path(__file__).resolve().parents[1]
_data_root = Path(os.environ.get("SCE_DEPTH_DATA_DIR", str(ROOT_DIR / "datasets")))
DATASETS_DIR = _data_root if _data_root.name == "datasets" else _data_root / "datasets"

PATHS = ProjectPaths(
    datasets=DATASETS_DIR,
    mlruns=ROOT_DIR / Path("mlruns"),
    containers=ROOT_DIR / Path("containers"),
    slurm=ROOT_DIR / Path("slurm"),
    matplotlib_cache=ROOT_DIR / Path("mpl_cache"),
)

CONTAINER = Container(singularity_container_name="sce_depth_container.sif")

LOGGING = Logging(mlflow_backend="filesystem")

