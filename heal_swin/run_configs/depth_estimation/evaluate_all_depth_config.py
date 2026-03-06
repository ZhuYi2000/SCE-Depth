#!/usr/bin/env -S python3 -u
# fmt: off
#SBATCH -t 1-00:00:00  # noqa: E265
#SBATCH -o ../../../slurm/slurm-%A_%a.out  # noqa: E265
dummy="dummy"  # noqa: E225
# fmt: on

import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

#######################################################################
# Run this file as an array job.

# # Run this file as an array job.
# # For flat:
# # sbatch -a 0-4 evaluate_all_depth_config.py
# # For HP:
# # sbatch -a 0-5 evaluate_all_depth_config.py

# zhu-20241023:
# python3 run.py --env singularity inference_isaac --config_path=heal_swin/run_configs/depth_estimation/evaluate_all_depth_config.py
#######################################################################

# RUN_ID = "0955e4ee6aea4e2bafe4593e37ef4f3b"  # compoundeye hp, LR=5e-3, gcv=0.1
# RUN_ID = "a9c7757133484f9eb6ba4760be12adc7"  # compoundeye hp, LR=1e-4, gcv=0.1
# RUN_ID = "b0e0b89bf57a4cf482d8472a86007b6b"  # compoundeye hp, LR=1e-4, gcv=1.0, fov=3d, 600-test0
# RUN_ID = "447c48f671d94b529efedbee94a92c9a" # compoundeye hp, LR=1e-4, gcv=1.0, fov=6d
# RUN_ID = "9416843a0b9846c3bc14e6519ff144ee" # compoundeye hp, LR=1e-4, gcv=1.0, fov=3d, 1080
# RUN_ID = "6500e7f1737a4764808e1aa8a85c8e22"  # compoundeye hp, LR=1e-4, gcv=1.0, fov=3d, 600-test1
# RUN_ID = "279470c56b6b41cbb1f3f22bc685960b"  # compoundeye hp, LR=1e-4, gcv=1.0, fov=6d, 600-test1
# RUN_ID = "ea164f34d98e46139c1d812cd98fdc87"  # # compoundeye hp, LR=1e-4, gcv=1.0, fov=4d, 600
# RUN_ID = "2e9f19d844314ed38fc25d7ae735d691"  # # compoundeye hp, LR=1e-4, gcv=1.0, fov=2d, 600

# RUN_ID = "ffdfb6b4de2847f4961318a9f3401cce" # fisheye flat
# RUN_ID = "2fbde4a02b05455d80b0242107d10bb9" # fisheye hp, LR=5e-3, gcv=0.1 
# RUN_ID = "6b90b57a9e3540b5a39e71786d8d87ae" # fisheye hp, LR=1e-4, gcv=0.1
# RUN_ID = "1a921bab567c4ba6b7479d02768bbd50" # fisheye hp, LR=1e-4, gcv=1.0

# RUN_ID = "51a28b44116e46fe8e8d41e095775ae2"  # Compound-128-max10-fov5-600-meanD, LR=1e-4, gcv=1.0
# RUN_ID = "504a91e8df34484bb40641c21c53129a"  # Compound-128-max10-fov5-600-meanD, LR=1e-4, gcv=0.0
# RUN_ID = "5e77bd0a0d7e422b8ccac2c91e16df8a"  # Compound-128-max10-fov3-600-meanD, LR=1e-4, gcv=1.0
# RUN_ID = "a24c332f74c449aaa24780c1eb02323c"  # Compound-128-max10-fov3-600-meanD, LR=1e-4, gcv=1.0, blur_alpha=0.1
# RUN_ID = "8e77c5241c9d4ee1ab710339789962ff"  # Compound-128-max10-fov3-600-meanD, LR=1e-4, gcv=1.0, blur_alpha=0.1  -- 871
# RUN_ID = "b3cf2dd5d8194be9ae0e707f170090fd"  # Compound-128-max10-fov3-600-meanD, LR=1e-4, gcv=1.0, blur_alpha=1.0  -- 899
# RUN_ID = "c95bef88c055428ea1348d3ebba6d7a0"  # Compound-128-max10-fov3-600-meanD, LR=1e-4, gcv=1.0, blur_alpha=1.0  -- 899
# RUN_ID = "17acfc9aa706430a9f4019e8cca0ab08"

# RUN_ID = "721f18526108470b94b1fed30f9222b7"  # Compound-128-max10-fov3-1402-meanD, LR=4e-5, gcv=1.0, blur_alpha=0.1
# RUN_ID = "01c59d7081aa46d4aeba0987284cc02b"  # Compound-128-max10-fov3-1402-meanD, LR=4e-5, gcv=1.0, blur_alpha=1.0
# RUN_ID = "f785928d076a49749c1582bbb555c7b4"  # Compound-128-max10-fov3-1402-meanD, LR=4e-5, gcv=1.0

# RUN_ID = "4c9f4b7f43ab4a94b063e2f638b0e169"  # Compound-128-max10-fov3-2730-meanD, LR=4e-5, gcv=1.0, blur_alpha=1.0
# RUN_ID = "2028e384fd2b443bab4d60ad0d54aa87"  # Compound-128-max10-fov3-2730-meanD, LR=4e-5, gcv=1.0, blur_alpha=0.1
# RUN_ID = "42f7cc0a6a674d928458f81ffd2e4cc1"  # Compound-128-max10-fov3-2730-meanD, LR=4e-5, gcv=1.0, blur_alpha=0.1
# RUN_ID = "371218ea94d74c5abf9a0a1b5bec6baf"  # Compound-128-max10-fov3-2730-meanD, LR=5e-5, gcv=1.0, blur_alpha=1.0

# RUN_ID = "1b8bd42493cf45ddb576076b9cc4aa38"  # Compound-128-max10-fov3-2730-meanD, LR=2e-5, gcv=1.0, blur_alpha=1.0
# RUN_ID = "3e181370c83a4ffc8ea9a4085b27a96d"  # Compound-128-max10-fov5-2730-meanD, LR=2e-5, gcv=1.0, blur_alpha=1.0
# RUN_ID = "532ef31537cd446b89b5a6c35f2caa8e"  # Compound-128-max10-fov7-2730-meanD, LR=2e-5, gcv=1.0, blur_alpha=1.0  # number=1143


# RUN_ID = "2bb24817c9e24fe99311aa88ff08208e"  # FisheyeFlat-128-max10_2730, LR=5e-5, gcv=0.0
# RUN_ID = "5f9429dca0934effa5a0562905b3b404"  # FisheyeFlat-128-max10_2730, LR=5e-5, gcv=0.0, blur=0.1
# RUN_ID = "ddf9e057552f41b0b18c3f02720969c4"  # FisheyeFlat-128-max10_2730, LR=5e-5, gcv=0.0, blur=1.0
# RUN_ID = "b61699b5b2914a17bfd4a706c4336791"  # FisheyeFlat-128-max10_1402, LR=5e-5, gcv=0.0
# RUN_ID = "6a032203175e4a8e9d119caadbf56ac5"  # FisheyeFlat-128-max10, LR=5e-5, gcv=0.0
# RUN_ID = "09b2b8e13e084a388a5e38066906494a"  # Fov70Flat-128-max10, LR=5e-5, gcv=0.0
# RUN_ID = "5755581544dd4f3d81e05018629a9096"  # CompoundFlat-128-max10-fov3-600-meanD, LR=5e-5, gcv=0.0



# RUN_ID = "0a164b1a78574dfbaff5766da4ecc15d"  # Compound-128-max10-fov3-1364-meanD  best
# RUN_ID = "b0622ffb40664fb8b19f44806488a79f"  # Compound-128-max10-fov3-1364-meanD  best  baseline(no blur loss)

# RUN_ID = "305925228fdd44b49c39ac195464a6c4"  # CompoundFlat-128-max10-fov3-1364-meanD  best
# RUN_ID = "a157387d6bce4cd8b4a69030fe84cf73"  # CompoundFlat-128-max10-fov3-1364-meanD  best  baseline(no blur loss)

# RUN_ID = "81f2fbdfe27242aebff22e5550edf599"  # FisheyeSpherical-1364  best
# RUN_ID = "fae74d20f3f645bc8066424a4058bf35"  # FisheyeSpherical-1364  best  baseline(no blur loss)

# RUN_ID = "375a1691d5024019afd68bfe55b90721"  # FisheyeFlat-128-max10-1364  best
# RUN_ID = "39893fb220be4498b20c7c550e5c8fc7"  # FisheyeFlat-128-max10-1364  best  baseline(no blur loss)



# RUN_ID = "7911b8b6c264427ba6501094b340e7f2"  # Compound-128-max10-fov5-1364-meanD best
# RUN_ID = "a022836d14684df595985bc7dabb70b0"  # Compound-128-max10-fov5-1364-meanD best  baseline(no blur loss)

# RUN_ID = "c851eaa8821546c999a001f17da2e25b"  # Compound-128-max10-fov7-1364-meanD number=978
# RUN_ID = "5f849a3b94a0448e87c7e468391f8323"  # Compound-128-max10-fov7-1364-meanD best  baseline(no blur loss)

# RUN_ID = "23ec77d2295a47429437d6a8e45cd72f"  # Compound-128-max10-fov1-1364-meanD best
# RUN_ID = "714b074a91c24058b23aa10e1c03a449"  # Compound-128-max10-fov1-1364-meanD best  baseline(no blur loss)


### fisheye 1364
# RUN_ID = ""

RUN_ID = os.getenv("RUN_ID", default="")
EPOCH = "best"
# EPOCH = "number"
# EPOCH = "last"
EPOCH_NUMBER = "492"


def get_eval_run_config():
    from heal_swin.utils import utils
    from heal_swin.evaluation.evaluate_config import EvaluateConfig
    from heal_swin.data.data_config import WoodscapeDepthFlatConfig, WoodscapeHPDepthConfig, WoodscapeISAACDepthConfig, WoodscapeISAACFlatDepthConfig, WoodscapeISAACDepthSobelConfig

    train_run_config = utils.load_config(RUN_ID, "run_config")
    data_config = train_run_config.data
    train_config = train_run_config.train
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")

    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")
    if isinstance(data_config, WoodscapeDepthFlatConfig):
        flat_hp = "flat"
        if task_count != "5":
            print(f"\n\nWARNING: found {task_count} tasks, expected 5\n\n")
    elif isinstance(data_config, WoodscapeHPDepthConfig):
        flat_hp = "hp"
        if task_count != "6":
            print(f"\n\nWARNING: found {task_count} tasks, expected 6\n\n")
    elif isinstance(data_config, WoodscapeISAACDepthConfig):
        flat_hp = "hp"
        if task_count != "6":
            print(f"\n\nWARNING: found {task_count} tasks, expected 6\n\n")
    elif isinstance(data_config, WoodscapeISAACDepthSobelConfig):
        flat_hp = "hp"
        if task_count != "6":
            print(f"\n\nWARNING: found {task_count} tasks, expected 6\n\n")
    elif isinstance(data_config, WoodscapeISAACFlatDepthConfig):
        flat_hp = "flat"
        if task_count != "6":
            print(f"\n\nWARNING: found {task_count} tasks, expected 6\n\n")

    flat_hp = "hp"
    if task_count != "6":
        print(f"\n\nWARNING: found {task_count} tasks, expected 6\n\n")

    if EPOCH in ["best", "last"]:
        metric_prefix = EPOCH
    elif EPOCH == "number":
        metric_prefix = f"epoch_{EPOCH_NUMBER}"

    print(40 * "-")
    print(f"Evaluating RUN_ID: {RUN_ID} on {metric_prefix} epoch.")
    print(40 * "-")

    # default values for some rarely changed parameters
    ranking_metric = "mse"
    sort_dir = "desc"  # asc: best have highest metric value
    proj_res = 966
    pred_part = "val"
    pred_samples = 1.0
    predict = True
    validate = False
    top_k = 5

    eval_config_name = ""
    eval_config_name_suffix = ""

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    job_id = f"{os.environ.get('SLURM_ARRAY_JOB_ID', 'no_job_id')}_{task_id}"

    if task_id == "0":
        eval_config_name += f"{metric_prefix}_validation_{job_id}"
        pred_writer = "base_writer"
        pred_samples = 10
        validate = True
    if task_id == "1":
        eval_config_name += f"{metric_prefix}_validation_{job_id}"
        pred_writer = "chamfer_distance"
        top_k = 2
        ranking_metric = "chamfer_distance"
    elif task_id == "2":
        eval_config_name += f"{metric_prefix}_val_best_worst_{job_id}"
        pred_writer = "best_worst_preds"
    elif task_id == "3":
        eval_config_name += f"{metric_prefix}_train_best_worst_{job_id}"
        pred_writer = "best_worst_preds"
        pred_part = "train"

    if flat_hp == "flat":
        if task_id == "4":
            eval_config_name += f"{metric_prefix}_projected_to_hp_{job_id}"
            pred_writer = "val_on_hp_projected"

    if flat_hp == "hp":
        if task_id == "4":
            eval_config_name += f"{metric_prefix}_back_projected_full_res_{job_id}"
            pred_writer = "val_on_back_projected"
        if task_id == "5":
            eval_config_name += f"{metric_prefix}_back_projected_flat_res_{job_id}"
            pred_writer = "val_on_back_projected"
            proj_res = (640, 768)

    eval_config_name = eval_config_name + eval_config_name_suffix

    data_config.common.pred_samples = pred_samples
    data_config.predict_part = pred_part
    return EvaluateConfig(
        path=RUN_ID,
        epoch=EPOCH,
        epoch_number=EPOCH_NUMBER,
        eval_config_name=eval_config_name,
        metric_prefix=metric_prefix,
        override_eval_config=True,
        ranking_metric=ranking_metric,
        sort_dir=sort_dir,
        pred_writer=pred_writer,
        predict=predict,
        validate=validate,
        log_masked_iou=False,
        top_k=top_k,
        proj_res=proj_res,
        data_config=data_config,
        train_config=train_config,
    )


def get_pl_config():
    from heal_swin.utils import utils
    from heal_swin.training.train_config import PLConfig

    try:
        train_pl_config = utils.load_config(RUN_ID, "pl_config")
    except AssertionError:  # thrown by file not found in load_config
        train_args_dict = utils.load_config(RUN_ID, "args")
        train_pl_config = PLConfig()
        for key, value in train_args_dict.items():
            if hasattr(train_pl_config, key):
                setattr(train_pl_config, key, value)
    train_pl_config.gpus = 1
    return train_pl_config


def main():
    this_path = str(Path(__file__).absolute())

    if "SLURM_SUBMIT_DIR" in os.environ:
        base_path = str(Path(os.environ["SLURM_SUBMIT_DIR"]).parents[2])
    else:
        base_path = str(Path(this_path).parents[3])

    run_py_path = os.path.join(base_path, "run.py")

    command = ["python3", "-u", run_py_path]
    command += ["--env", "singularity"]
    command += ["inference_isaac"]
    command += ["--config_path", this_path]
    print(" ".join(command))

    subprocess.run(command)


if __name__ == "__main__":
    main()
