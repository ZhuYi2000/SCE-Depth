from typing import Tuple

from heal_swin.data.depth_estimation import (
    flat_depth_datamodule,
    hpSobel_depth_datasets,
    hp_depth_datasets,
    isaac_depth_datasets,
    isaacFlat_depth_datamodule,
    isaacSobel_depth_datasets,
)
from heal_swin.data.depth_estimation.data_spec_depth import (
    DepthDataSpec,
    create_depth_dataspec_from_data_module,
)
from heal_swin.data.depth_estimation.flat_depth_datamodule import WoodscapeFlatDepthDataModule
from heal_swin.data.depth_estimation.hp_depth_datasets import WoodscapeHPDepthDataModule
from heal_swin.data.depth_estimation.hpSobel_depth_datasets import WoodscapeHPSobelHPDepthDataModule
from heal_swin.data.depth_estimation.isaac_depth_datasets import IsaacHPDepthDataModule
from heal_swin.data.depth_estimation.isaacFlat_depth_datamodule import IsaacFlatDepthDataModule
from heal_swin.data.depth_estimation.isaacSobel_depth_datasets import IsaacSobelHPDepthDataModule
from heal_swin.data.data_config import (
    WoodscapeDepthFlatConfig,
    WoodscapeHPDepthConfig,
    WoodscapeHPDepthSobelConfig,
    WoodscapeISAACDepthConfig,
    WoodscapeISAACDepthSobelConfig,
    WoodscapeISAACFlatDepthConfig,
)


def get_depth_flat_data_module(config) -> Tuple[WoodscapeFlatDepthDataModule, DepthDataSpec]:
    dm = flat_depth_datamodule.WoodscapeFlatDepthDataModule(
        pred_part=config.pred_part,
        padding=config.padding,
        bandwidth=config.input_bandwidth,
        size=config.input_bandwidth * 2,
        shuffle_train_val_split=config.shuffle_train_val_split,
        nside=config.nside,
        base_pix=config.base_pix,
        data_transform=config.common_depth.data_transform,
        mask_background=config.common_depth.mask_background,
        normalize_data=config.common_depth.normalize_data,
        **config.common.__dict__,
    )
    data_spec = create_depth_dataspec_from_data_module(dm)
    return dm, data_spec


def get_depth_hp_data_module(config) -> Tuple[WoodscapeHPDepthDataModule, DepthDataSpec]:
    dm = hp_depth_datasets.WoodscapeHPDepthDataModule(
        pred_part=config.pred_part,
        nside=config.input_nside,
        base_pix=config.input_base_pix,
        shuffle_train_val_split=config.shuffle_train_val_split,
        data_transform=config.common_depth.data_transform,
        mask_background=config.common_depth.mask_background,
        normalize_data=config.common_depth.normalize_data,
        **config.common.__dict__,
    )
    data_spec = create_depth_dataspec_from_data_module(dm, base_pix=config.input_base_pix)
    return dm, data_spec


def get_depth_isaac_data_module(config) -> Tuple[IsaacHPDepthDataModule, DepthDataSpec]:
    dm = isaac_depth_datasets.IsaacHPDepthDataModule(
        pred_part=config.pred_part,
        nside=config.input_nside,
        base_pix=config.input_base_pix,
        eye_fov=config.eye_fov,
        max_depth=config.max_depth,
        dataset_size=config.dataset_size,
        compound_eye=config.compound_eye,
        shuffle_train_val_split=config.shuffle_train_val_split,
        data_transform=config.common_depth.data_transform,
        mask_background=config.common_depth.mask_background,
        normalize_data=config.common_depth.normalize_data,
        **config.common.__dict__,
    )
    data_spec = create_depth_dataspec_from_data_module(dm, base_pix=config.input_base_pix)
    return dm, data_spec


def get_depth_isaacFlat_data_module(config) -> Tuple[IsaacFlatDepthDataModule, DepthDataSpec]:
    dm = isaacFlat_depth_datamodule.IsaacFlatDepthDataModule(
        pred_part=config.pred_part,
        padding=config.padding,
        bandwidth=config.input_bandwidth,
        size=config.input_bandwidth * 2,
        shuffle_train_val_split=config.shuffle_train_val_split,
        nside=config.nside,
        base_pix=config.base_pix,
        data_transform=config.common_depth.data_transform,
        mask_background=config.common_depth.mask_background,
        normalize_data=config.common_depth.normalize_data,
        **config.common.__dict__,
    )
    data_spec = create_depth_dataspec_from_data_module(dm)
    return dm, data_spec


def get_depth_isaacSobel_data_module(config) -> Tuple[IsaacSobelHPDepthDataModule, DepthDataSpec]:
    dm = isaacSobel_depth_datasets.IsaacSobelHPDepthDataModule(
        pred_part=config.pred_part,
        nside=config.input_nside,
        base_pix=config.input_base_pix,
        eye_fov=config.eye_fov,
        max_depth=config.max_depth,
        dataset_size=config.dataset_size,
        compound_eye=config.compound_eye,
        shuffle_train_val_split=config.shuffle_train_val_split,
        data_transform=config.common_depth.data_transform,
        mask_background=config.common_depth.mask_background,
        normalize_data=config.common_depth.normalize_data,
        **config.common.__dict__,
    )
    data_spec = create_depth_dataspec_from_data_module(dm, base_pix=config.input_base_pix)
    return dm, data_spec


def get_depth_hpSobel_data_module(config) -> Tuple[WoodscapeHPSobelHPDepthDataModule, DepthDataSpec]:
    dm = hpSobel_depth_datasets.WoodscapeHPSobelHPDepthDataModule(
        pred_part=config.pred_part,
        nside=config.input_nside,
        base_pix=config.input_base_pix,
        shuffle_train_val_split=config.shuffle_train_val_split,
        data_transform=config.common_depth.data_transform,
        mask_background=config.common_depth.mask_background,
        normalize_data=config.common_depth.normalize_data,
        **config.common.__dict__,
    )
    data_spec = create_depth_dataspec_from_data_module(dm, base_pix=config.input_base_pix)
    return dm, data_spec


def get_data_module(data_config):
    data_dispatch = {
        WoodscapeDepthFlatConfig.__name__: get_depth_flat_data_module,
        WoodscapeHPDepthConfig.__name__: get_depth_hp_data_module,
        WoodscapeISAACDepthConfig.__name__: get_depth_isaac_data_module,
        WoodscapeISAACFlatDepthConfig.__name__: get_depth_isaacFlat_data_module,
        WoodscapeISAACDepthSobelConfig.__name__: get_depth_isaacSobel_data_module,
        WoodscapeHPDepthSobelConfig.__name__: get_depth_hpSobel_data_module,
    }
    return data_dispatch[data_config.__class__.__name__](data_config)
