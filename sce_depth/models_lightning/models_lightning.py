from typing import Literal

from sce_depth.models_lightning.depth_estimation import (
    model_lightning_depth_swin,
    model_lightning_depth_swin_hp,
    model_lightning_depth_swin_hp_sobel,
)

MODEL_CLASSES = [
    model_lightning_depth_swin.WoodscapeDepthSwin,  # CONFIG_CLASS = WoodscapeDepthSwinConfig
    model_lightning_depth_swin_hp.WoodscapeDepthSwinHP,  # CONFIG_CLASS = WoodscapeDepthSwinHPConfig
    model_lightning_depth_swin_hp_sobel.WoodscapeDepthSwinHPSobel,  # CONFIG_CLASS = WoodscapeDepthSwinHPSobelConfig
]

MODEL_CONFIGS_LITERAL = Literal[
    model_lightning_depth_swin.WoodscapeDepthSwin.CONFIG_CLASS,
    model_lightning_depth_swin_hp.WoodscapeDepthSwinHP.CONFIG_CLASS,
    model_lightning_depth_swin_hp_sobel.WoodscapeDepthSwinHPSobel.CONFIG_CLASS,
]

MODELS = {model.NAME: model for model in MODEL_CLASSES}

MODEL_NAME_FROM_CONFIG_NAME = {
    model.CONFIG_CLASS.__name__: model.__name__ for model in MODEL_CLASSES
}

MODEL_FROM_CONFIG_NAME = {model.CONFIG_CLASS.__name__: model for model in MODEL_CLASSES}

MODEL_FROM_NAME = {model.__name__: model for model in MODEL_CLASSES}

