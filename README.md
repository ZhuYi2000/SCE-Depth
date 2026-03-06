# SCE-Depth

This project is built on top of [HEAL-SWIN](https://github.com/JanEGerken/HEAL-SWIN).  
We sincerely thank the HEAL-SWIN authors for releasing their codebase, which this repository heavily reuses and extends.

Official code repository for **SCE-Depth: A Spherical Compound Eye Framework for Wide FOV Depth Estimation**.

## Environment

The environment setup follows HEAL-SWIN.

- Use the same dependency/container workflow as HEAL-SWIN.
- Training and evaluation are run with the **Singularity** container backend.
- The default container path is:
  - `containers/heal_swin_container.sif`

## Dataset

Place datasets under the repository `datasets/` directory, with the depth dataset root at:

- `datasets/CompoundDepth/`

For the default training config, data is expected under:

- `datasets/CompoundDepth/isaac_depth_images_nside=128_base_pix=8_eye_fov=3.0_max_depth=10.0_dataset_size=1364_compound_eye=ico20609+sobel+2025`

Details and code for dataset generation are provided in the paper supplementary material.

## Training

Run training with:

```bash
python3 run.py --env singularity train_isaac --config_path=heal_swin/run_configs/depth_estimation/depth_swin_isaac_train_run_config.py
```

## Citation

If you use this codebase, please cite our paper (citation entry will be added after publication metadata is finalized).

## Acknowledgment

- HEAL-SWIN repository and authors.
- All open-source contributors whose tools made this project possible.
