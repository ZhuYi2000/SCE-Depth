from dataclasses import dataclass, field
import torchvision.models as models
import torch

import pytorch_lightning as pl
from torchmetrics import MetricCollection
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from heal_swin.training import loss_depth_regression
from heal_swin.evaluation import custom_metrics
from heal_swin.utils import depth_utils
from heal_swin.models_lightning.depth_estimation.depth_common_config import CommonDepthConfig
from heal_swin.data.depth_estimation.data_spec_depth import DepthDataSpec
from heal_swin.models_torch.swin_hp_transformer import SwinHPTransformerConfig, SwinHPTransformerSys, SwinHPSobelTransformerSys, SwinHPSobelTransformerSys_2, SwinHPSobelTransformerSys_3, SwinHPSobelTransformerSys_4, SwinHPSobelTransformerSys_5
from heal_swin.training.optimizer import OptimizerConfig, get_lightning_optimizer_dict
from pathlib import Path


HEAL_SWIN_ROOT = Path(__file__).resolve().parents[2]

def _runtime_assets_path(path: str) -> str:
    return str(HEAL_SWIN_ROOT / "runtime_assets" / path)


@dataclass
class WoodscapeDepthSwinHPSobelConfig:
    swin_hp_transformer_config: SwinHPTransformerConfig = field(
        default_factory=SwinHPTransformerConfig
    )
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    common_depth_config: CommonDepthConfig = field(default_factory=CommonDepthConfig)
    use_sobel: bool = True


class WoodscapeDepthSwinHPSobel(pl.LightningModule):

    CONFIG_CLASS = WoodscapeDepthSwinHPSobelConfig
    NAME = "depth_swin_hp"

    def __init__(self, config: WoodscapeDepthSwinHPSobelConfig, data_spec: DepthDataSpec, data_config):
        super().__init__()
        print("")
        print("Creating a HEAL-SWIN model for depth estimation")
        print("")

        self.config = config
        self.mlflow_params = {}
        self.mlflow_tags = {}

        self.val_metrics_prefix = ""

        if isinstance(config.common_depth_config.train_uncertainty_after, int):
            assert config.common_depth_config.train_uncertainty_after > 0, (
                "Can't switch loss immediately (got switching after epoch "
                f"{config.train_uncertainty_after}), instead set 'train_uncertainty_after=False' "
                "in WoodscapeDepthCommonConfig."
            )

        self.train_uncertainty_after = config.common_depth_config.train_uncertainty_after

        self.use_logvar = config.common_depth_config.use_logvar
        self.data_transform = data_config.common_depth.data_transform
        self.mask_background = data_config.common_depth.mask_background
        self.normalize_data = data_config.common_depth.normalize_data  # normalize_data = "standardize"

        self.depth_data_statistics = data_spec.data_stats

        layer_norms = {"LayerNorm": nn.LayerNorm}
        if isinstance(config.swin_hp_transformer_config.norm_layer, str):
            config.swin_hp_transformer_config.norm_layer = layer_norms[
                config.swin_hp_transformer_config.norm_layer
            ]

        ###############################################################################################################################
        
        # self.model = SwinHPSobelTransformerSys(
        #     config.swin_hp_transformer_config,
        #     data_spec=data_spec.replace(f_out=2) if self.use_logvar else data_spec,
        # )

        # self.model = SwinHPSobelTransformerSys_2(
        #     config.swin_hp_transformer_config,
        #     data_spec=data_spec.replace(f_out=2) if self.use_logvar else data_spec,
        # )
            
        # self.model = SwinHPSobelTransformerSys_3(
        #     config.swin_hp_transformer_config,
        #     data_spec=data_spec.replace(f_out=2) if self.use_logvar else data_spec,
        # )
            
        # self.model = SwinHPSobelTransformerSys_4(
        #     config.swin_hp_transformer_config,
        #     data_spec=data_spec.replace(f_out=2) if self.use_logvar else data_spec,
        # )
        
        self.model = SwinHPSobelTransformerSys_5(
            config.swin_hp_transformer_config,
            data_spec=data_spec.replace(f_out=2) if self.use_logvar else data_spec,
        )

        # print("EdgeAttention is in?:", 'edge_attention' in dict(self.model.named_children()))
        # for name, module in self.model.named_modules():
        #     print(f"module name: {name}")
        ###############################################################################################################################

        self.depth_uncertainty_loss = loss_depth_regression.mean_log_var_loss
        self.blur_alpha = config.common_depth_config.blur_alpha # 1.0
        self.loss = loss_depth_regression.get_depth_loss(config.common_depth_config)
        self.loss_type = config.common_depth_config.loss
        metric_dict = {
            "mse": custom_metrics.DepthMSE(),
        }

        if self.use_logvar:
            metric_dict.update(
                {
                    "mean_std": custom_metrics.MeanSTD(),
                    "median_std": custom_metrics.MeanSTDMedian(),
                }
            )

        metrics = MetricCollection(metric_dict)
        self.metric_dict = metrics
        self.train_metrics = metrics.clone(prefix="train_")  # train_mse
        self.val_metrics = metrics.clone(prefix="val_")  # val_mse

        self.learning_rate = config.optimizer_config.learning_rate

        self.convert_indices = torch.from_numpy(np.load(_runtime_assets_path("hp16384_1d2d_index.npy")))
        self.inverse_indices = torch.argsort(self.convert_indices)

    def forward(self, x):
        outputs = self.model(x.float())
        # print("outputs.shape", outputs.shape)
        # exit("-------------------debug-------------------")

        outputs[:, 0, ...] = depth_utils.unnormalize_and_retransform(
            data=outputs[:, 0, ...],
            normalization=self.normalize_data,
            data_stats=self.depth_data_statistics,
            data_transform=self.data_transform,
        )
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        imgs = batch["hp_imgs"]
        outputs = self(imgs)
        return outputs

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, self.train_metrics)
        self.log("train_loss", loss.item(), on_epoch=True)
        self.log("blur_alpha", self.blur_alpha, on_epoch=True)  # epoch blur_alpha
        return loss

    def training_epoch_end(self, outputs):
        train_metrics_values = self.train_metrics.compute()

        self.log_dict(train_metrics_values, on_epoch=True, on_step=False)  # epoch train_mse
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch, self.val_metrics)
        self.log(self.val_metrics_prefix + "val_loss", loss.item(), on_epoch=True, on_step=False) # epoch val_loss
        return preds

    def validation_epoch_end(self, outputs):
        val_metrics_values = self.val_metrics.compute()

        pref = self.val_metrics_prefix
        val_metric_values = {pref + key: value for key, value in val_metrics_values.items()}

        self.log_dict(val_metric_values, on_epoch=True, on_step=False)  # epoch val_mse
        self.val_metrics.reset()

        # self.log("depth_data_statistics_name", self.depth_data_statistics.name, on_epoch=True, on_step=False)
        self.log("depth_data_statistics_mean", self.depth_data_statistics.mean, on_epoch=True, on_step=False)
        self.log("depth_data_statistics_std", self.depth_data_statistics.std, on_epoch=True, on_step=False)

    def shared_step(self, batch, metrics):
        imgs, masks, sobels = batch  # imgs.shape = torch.Size([4, 3, 131072]), masks.shape = torch.Size([4, 131072])
        # imgs.shape torch.Size([4, 3, 131072])
        # sobels.shape torch.Size([4, 131072])
        # masks.shape torch.Size([4, 131072])

        sobels = sobels.unsqueeze(1)  # sobels.shape torch.Size([4, 1, 131072])
        input_data = torch.cat([imgs, sobels], dim=1)  # input_data.shape torch.Size([4, 4, 131072])
        outputs = self(input_data)

        outputs[:, 0, ...] = depth_utils.transform_and_normalize(
            data=outputs[:, 0, ...],
            normalization=self.normalize_data,
            data_stats=self.depth_data_statistics,
            data_transform=self.data_transform,
        )

        if "blur" in self.loss_type:
            loss, blur_loss, mse_loss = self.loss(outputs, masks, mask_background=self.mask_background)
            self.log("train_blur_loss", blur_loss.item(), on_epoch=True, on_step=False)  # epoch blur_loss
            self.log("train_mse_loss", mse_loss.item(), on_epoch=True, on_step=False)  # epoch mse_loss
        else:
            loss = self.loss(outputs, masks, mask_background=self.mask_background)

        ###############################################################################################################################

        outputs[:, 0, ...] = depth_utils.unnormalize_and_retransform(
            data=outputs[:, 0, ...],
            normalization=self.normalize_data,
            data_stats=self.depth_data_statistics,
            data_transform=self.data_transform,
        )
        masks = depth_utils.unnormalize_and_retransform(
            data=masks,
            normalization=self.normalize_data,
            data_stats=self.depth_data_statistics,
            data_transform=self.data_transform,
        )

        metrics.update(outputs, masks)
        return loss, outputs

    def configure_optimizers(self):
        # Define the optimizer, allowing frozen layers if applicable
        params = [p for p in self.parameters() if p.requires_grad]

        return get_lightning_optimizer_dict(params, self.config.optimizer_config)
