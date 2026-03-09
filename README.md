# SCE-Depth

This project is built on top of [HEAL-SWIN](https://github.com/JanEGerken/HEAL-SWIN).  
We sincerely thank the HEAL-SWIN authors for releasing their codebase, which this repository heavily reuses and extends.

Official code repository for **SCE-Depth: A Spherical Compound Eye Framework for Wide FOV Depth Estimation**.

## Environment

The environment setup follows HEAL-SWIN.

- Use the same dependency/container workflow as HEAL-SWIN.
- Training and evaluation are run with the **Singularity** container backend.
- The default container path is:
  - `containers/sce_depth_container.sif`

## Dataset

The CompoundDepth dataset is publicly available at:

- https://huggingface.co/datasets/SJTU-Ramos/CompoundDepth

Download the dataset and place it under the repository `datasets/` directory, with the depth dataset root at:

- `datasets/CompoundDepth/`

For the default training config, the extracted dataset is expected under:

- `datasets/CompoundDepth/isaac_depth_images_nside=128_base_pix=8_eye_fov=3.0_max_depth=10.0_dataset_size=1364_compound_eye=ico20609+sobel+2026`

If you use a different local storage location, update `compute_environment/current_environment.py` or set `SCE_DEPTH_DATA_DIR` accordingly.

## Training

Run training with:

```bash
python3 run.py --env singularity train_isaac --config_path=sce_depth/run_configs/depth_estimation/depth_swin_isaac_train_run_config.py
```

## Citation

If you use this codebase, please cite our paper (citation entry will be added after publication metadata is finalized).

## Acknowledgment

- HEAL-SWIN repository and authors.
- All open-source contributors whose tools made this project possible.

