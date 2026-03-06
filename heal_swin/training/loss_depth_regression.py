from typing import Callable
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
from heal_swin.models_lightning.depth_estimation.depth_common_config import CommonDepthConfig
from pathlib import Path


HEAL_SWIN_ROOT = Path(__file__).resolve().parents[1]

def _runtime_assets_path(path: str) -> str:
    return str(HEAL_SWIN_ROOT / "runtime_assets" / path)

def mse(preds: torch.Tensor, target: torch.Tensor, mask_background: bool = False) -> torch.Tensor:
    '''
    [:, 0, ...], : (batch size ), 0 (channel ), ... ()
    '''
    means = preds[:, 0, ...]  # BCHW -> BHW
    means = means.unsqueeze(1)

    target = target.unsqueeze(1)  # Get target to same shape as preds: B1HW
    idxs = ~target.isinf().detach()

    sq_diff = torch.square(means[idxs] - target[idxs]) / 2

    loss = torch.mean(sq_diff)
    return loss


def mean_log_var_loss(
    preds: torch.Tensor, target: torch.Tensor, mask_background: bool = False
) -> torch.Tensor:

    means = preds[:, 0, ...]
    log_var = preds[:, 1, ...]

    # IMPORTANT: this is needed to decouple the indices from the torch autograd
    idxs = ~target.isinf().detach()

    std_weighted_sq_diff = 1 / 2 * log_var[idxs] + torch.square(means[idxs] - target[idxs]) * (
        0.5 * torch.exp(-log_var[idxs])
    )
    loss = torch.mean(std_weighted_sq_diff)

    return loss


def l1_loss(
    preds: torch.Tensor, target: torch.Tensor, mask_background: bool = False
) -> torch.Tensor:
    means = preds[:, 0, ...]
    means = means.unsqueeze(1)

    target = target.unsqueeze(1)  # Get target to same shape as preds: B1HW
    idxs = ~target.isinf().detach()

    l1_dist = torch.abs(means[idxs] - target[idxs])

    loss = torch.mean(l1_dist)
    return loss


def huber_loss(
    preds: torch.Tensor, target: torch.Tensor, mask_background: bool = False, delta=1
) -> torch.Tensor:
    means = preds[:, 0, ...]
    means = means.unsqueeze(1)

    target = target.unsqueeze(1)  # Get target to same shape as preds: B1HW
    idxs = ~target.isinf().detach()

    loss = torch.nn.SmoothL1Loss(reduction="mean", beta=delta)(preds[idxs], target[idxs])

    return loss

def berhu_loss(
        pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.2
        ) -> torch.Tensor:
    assert pred.dim() == target.dim(), "inconsistent dimensions"
    valid_mask = (target > 0).detach()

    diff = torch.abs(target - pred)
    diff = diff[valid_mask]
    delta = threshold * torch.max(diff).data.cpu().numpy()

    part1 = -F.threshold(-diff, -delta, 0.)
    part2 = F.threshold(diff ** 2 + delta ** 2, 2.0 * delta ** 2, 0.)
    part2 = part2 / (2. * delta)
    diff = part1 + part2
    loss = diff.mean()
    return loss


def calculate_gradient(image: torch.Tensor) -> torch.Tensor:
    """
    Calculate the gradient of the input image along x and y axes to measure edges.
    This simulates the blur effect related to object distance in compound eyes.

    Args:
        image: A tensor of shape (B, C, H, W), where B is the batch size, C is the channel number, 
               H is the height, and W is the width of the image.

    Returns:
        A tensor representing the gradient magnitude (edges) of the input image.
    """
    # Sobel operators for edge detection
    sobel_x = torch.tensor(
        [[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]], 
        device=image.device, dtype=torch.float32
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]], 
        device=image.device, dtype=torch.float32
    ).unsqueeze(0)

    padded_image = F.pad(image, (1, 1, 1, 1), mode='replicate')

    grad_x = F.conv2d(padded_image, sobel_x, padding=1)
    grad_y = F.conv2d(padded_image, sobel_y, padding=1)

    grad_x = grad_x[:, :, 1:-1, 1:-1].contiguous()
    grad_y = grad_y[:, :, 1:-1, 1:-1].contiguous()

    return torch.abs(grad_x), torch.abs(grad_y)

    # grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    # grad_magnitude = grad_magnitude[:, :, 1:-1, 1:-1]
    # grad_magnitude = grad_magnitude.contiguous()
    # return grad_magnitude


def calculate_gradient_scharr(image: torch.Tensor) -> torch.Tensor:
    """
    Calculate the gradient of the input image along x and y axes to measure edges.
    This simulates the blur effect related to object distance in compound eyes.

    Args:
        image: A tensor of shape (B, C, H, W), where B is the batch size, C is the channel number, 
               H is the height, and W is the width of the image.

    Returns:
        A tensor representing the gradient magnitude (edges) of the input image.
    """
    # Sobel operators for edge detection
    sobel_x = torch.tensor(
        [[[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]], 
        device=image.device, dtype=torch.float32
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]]], 
        device=image.device, dtype=torch.float32
    ).unsqueeze(0)

    padded_image = F.pad(image, (1, 1, 1, 1), mode='replicate')

    grad_x = F.conv2d(padded_image, sobel_x, padding=1)
    grad_y = F.conv2d(padded_image, sobel_y, padding=1)

    grad_x = grad_x[:, :, 1:-1, 1:-1].contiguous()
    grad_y = grad_y[:, :, 1:-1, 1:-1].contiguous()

    return torch.abs(grad_x), torch.abs(grad_y)


def calculate_gradient_laplacian(image: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Laplacian (second-order derivative) of the input image to detect edges.
    The Laplacian operator responds to regions of rapid intensity change.

    Args:
        image: A tensor of shape (B, C, H, W)

    Returns:
        A tensor representing the Laplacian response (edges) of the input image.
    """
    laplacian_kernel = torch.tensor(
        [[[0.,  1.,  0.],
          [1., -4.,  1.],
          [0.,  1.,  0.]]],
        device=image.device, dtype=torch.float32
    ).unsqueeze(0)  # shape: (1, 1, 3, 3)

    # laplacian_kernel = torch.tensor(
    #     [[[1.,  1.,  1.],
    #       [1., -8.,  1.],
    #       [1.,  1.,  1.]]],
    #     device=image.device, dtype=torch.float32
    # ).unsqueeze(0)

    padded_image = F.pad(image, (1, 1, 1, 1), mode='replicate')
    laplacian = F.conv2d(padded_image, laplacian_kernel, padding=1)

    laplacian = laplacian[:, :, 1:-1, 1:-1].contiguous()

    return laplacian.abs()


def calculate_gradient_LoG(image: torch.Tensor, kernel_size: int = 5, sigma: float = 0.5) -> torch.Tensor:
    """
    Correct LoG: Gaussian smoothing per channel + Laplacian.
    """
    B, C, H, W = image.shape
    device = image.device

    # Step 1: Gaussian Kernel (2D, channel-aware)
    coords = torch.arange(kernel_size, device=device) - kernel_size // 2
    x, y = torch.meshgrid(coords, coords)
    gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    gaussian = gaussian.expand(C, 1, kernel_size, kernel_size)  # channel-wise

    # Step 2: Gaussian blur (preserve per-channel)
    image_blur = F.conv2d(F.pad(image, (kernel_size//2,)*4, mode='replicate'),
                          gaussian, groups=C)

    # Step 3: Laplacian kernel (4-neighbor)
    laplacian_kernel = torch.tensor(
        [[[0., 1., 0.],
          [1., -4., 1.],
          [0., 1., 0.]]], device=device
    ).expand(C, 1, 3, 3)

    # Step 4: Laplacian filtering
    laplacian = F.conv2d(F.pad(image_blur, (1, 1, 1, 1), mode='replicate'),
                         laplacian_kernel, groups=C)

    return laplacian.abs()


def torch_1d_to_2d(tensor_1d, indices):
    """
    Convert a 1D tensor to a 2D tensor using precomputed indices.
    """
    if len(tensor_1d.shape) == 1:
        tensor_1d = tensor_1d.unsqueeze(0)
    bs = tensor_1d.shape[0]
    # print("tensor_1d.shape", tensor_1d.shape)
    return tensor_1d[:, indices].view(bs, 128, 128)


def torch_2d_to_1d(tensor_2d, indices):
    """
    Convert a 2D tensor back to a 1D tensor using precomputed indices.
    """
    # regenerate and consider the bs
    bs = tensor_2d.shape[0]
    inverse_indices = torch.argsort(indices)
    return tensor_2d.view(bs, -1)[:, inverse_indices]




def mse_blur(
    preds: torch.Tensor, target: torch.Tensor, convert_index, coords_heal, mask_background: bool = False, blur_alpha=0.1
) -> torch.Tensor:
    means = preds[:, 0, ...]
    means = means.unsqueeze(1)  # means.dtype torch.float32

    target = target.unsqueeze(1)
    idxs = ~target.isinf().detach()


    sq_diff = torch.square(means[idxs] - target[idxs]) / 2

    mse_loss = torch.mean(sq_diff)  # mse_loss.dtype torch.float64

    # mse_loss = berhu_loss(means[idxs], target[idxs])

    # Add blur loss here
    if len(means.shape) == 1:
        means = means.unsqueeze(0)
    if len(target.shape) == 1:
        target = target.unsqueeze(0)

    target = target.to(torch.float32)
    means_sobel_feature_x = torch.zeros_like(means)
    means_sobel_feature_y = torch.zeros_like(means)
    target_sobel_feature_x = torch.zeros_like(target)
    target_sobel_feature_y = torch.zeros_like(target)
    means = torch.split(means, 16384, dim=2)
    targets = torch.split(target, 16384, dim=2)

    for index, mean in enumerate(means):
        mean = torch.where(torch.isinf(mean), torch.full_like(mean, 0), mean)
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        array_2d = torch_1d_to_2d(mean.squeeze(), convert_index).unsqueeze(1)  # torch.Size([4, 1, 128, 128])  torch.float32
        sobel_feature_x, sobel_feature_y = calculate_gradient(array_2d)  # torch.Size([4, 1, 128, 128])
        reconstructed_soel_feature_x = torch_2d_to_1d(sobel_feature_x, convert_index).unsqueeze(1)  # torch.Size([4,1,16384])
        reconstructed_soel_feature_y = torch_2d_to_1d(sobel_feature_y, convert_index).unsqueeze(1)  # torch.Size([4,1,16384])
        reconstructed_soel_feature_x = torch.where(torch.isinf(reconstructed_soel_feature_x), torch.zeros_like(reconstructed_soel_feature_x), reconstructed_soel_feature_x)
        reconstructed_soel_feature_y = torch.where(torch.isinf(reconstructed_soel_feature_y), torch.zeros_like(reconstructed_soel_feature_y), reconstructed_soel_feature_y)

        means_sobel_feature_x[:, :, index*16384:(index+1)*16384] = reconstructed_soel_feature_x
        means_sobel_feature_y[:, :, index*16384:(index+1)*16384] = reconstructed_soel_feature_y
    means_sobel_feature_x[:,:,coords_heal[:, 2] < 0.01] = 0
    means_sobel_feature_y[:,:,coords_heal[:, 2] < 0.01] = 0

    for index, target in enumerate(targets):
        target = torch.where(torch.isinf(target), torch.full_like(target, 0), target)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)

        array_2d = torch_1d_to_2d(target.squeeze(), convert_index).unsqueeze(1) # torch.Size([4, 1, 128, 128])

        sobel_feature_x, sobel_feature_y = calculate_gradient(array_2d)  # torch.Size([4, 1, 128, 128])
        reconstructed_soel_feature_x = torch_2d_to_1d(sobel_feature_x, convert_index).unsqueeze(1)  # torch.Size([4,1,16384])
        reconstructed_soel_feature_y = torch_2d_to_1d(sobel_feature_y, convert_index).unsqueeze(1)  # torch.Size([4,1,16384])
        reconstructed_soel_feature_x = torch.where(torch.isinf(reconstructed_soel_feature_x), torch.zeros_like(reconstructed_soel_feature_x), reconstructed_soel_feature_x)
        reconstructed_soel_feature_y = torch.where(torch.isinf(reconstructed_soel_feature_y), torch.zeros_like(reconstructed_soel_feature_y), reconstructed_soel_feature_y)

        target_sobel_feature_x[:, :, index*16384:(index+1)*16384] = reconstructed_soel_feature_x
        target_sobel_feature_y[:, :, index*16384:(index+1)*16384] = reconstructed_soel_feature_y
    target_sobel_feature_x[:,:,coords_heal[:, 2] < 0.01] = 0
    target_sobel_feature_y[:,:,coords_heal[:, 2] < 0.01] = 0

    blur_loss = berhu_loss(means_sobel_feature_x[idxs], target_sobel_feature_x[idxs]) + berhu_loss(means_sobel_feature_y[idxs], target_sobel_feature_y[idxs])
    # Testing new blur loss
    loss = mse_loss + blur_alpha * blur_loss


    
    # blur_loss = torch.mean(torch.abs(means_sobel_feature[idxs] - target_sobel_feature[idxs]))
    
    ''' solution test: Regularization  117872 - f40947d8ad194a4ba5194470c51f2ee3  -644 nan
    blur_loss = torch.mean(torch.abs(means_sobel_feature[idxs] - target_sobel_feature[idxs]))
    regularization = 1e-6 * (torch.norm(means_sobel_feature[idxs], p=2) + torch.norm(target_sobel_feature[idxs], p=2))
    blur_loss = blur_loss + regularization
    '''
    

    # if blur_loss is nan
    if torch.isnan(blur_loss):
        print("detected nan in blur_loss, ", blur_loss.cpu().detach().numpy())
        blur_loss = torch.where(torch.isnan(blur_loss), torch.zeros_like(blur_loss), blur_loss)
    
    if torch.isnan(mse_loss):
        print("detected nan in mse_loss, ", mse_loss.cpu().detach().numpy())
        mse_loss = torch.where(torch.isnan(mse_loss), torch.zeros_like(mse_loss), mse_loss)
    
    # loss = mse_loss + blur_alpha * blur_loss    # solution test: log blur loss and mse loss 117900 - f44a521bacc34fa6bc04269264e1f37a  -269 nan
    # solution test: learnable blur alpha(init=0.1)  118136 - 7a31a4678eda4258a6d76ef791260479
    # solution test: learnable blur alpha(init=1.0)  118140 - 21135a94ece3456181f3c4fa4bbf4309

    '''# solution test: Loss Scaling  117917 - 0b08843060004e8f9de279c1c8b440a8  -699 nan
        # Loss Scaling + learnable blur alpha(init=0.1)  118146 - 55dc88b0ed084ab68c2b3a31b772fc10  - 255 nan
        # Loss Scaling + learnable blur alpha(init=1.0)  118147 - 60be4a3631a14ec08aa65584bd057de1

    mse_mean = mse_loss.detach().mean()
    blur_mean = blur_loss.detach().mean()
    if mse_mean == 0 or blur_mean == 0:
        print("mse_mean or blur_mean is 0, ", mse_mean, blur_mean)
        exit("--------------------------------debug--------------------------------")
    
    if mse_mean < blur_mean:
        scale_factor = mse_mean / blur_mean
        scaled_blur_loss = blur_loss * scale_factor
        loss = mse_loss + blur_alpha * scaled_blur_loss
    else:
        scale_factor = blur_mean / mse_mean
        scaled_mse_loss = mse_loss * scale_factor
        loss = scaled_mse_loss + blur_alpha * blur_loss
    '''

    if torch.isnan(loss):
        print("detected nan in loss, ", loss.cpu().detach().numpy())
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return loss, blur_loss, mse_loss


def mse_blur_planner(preds: torch.Tensor, target: torch.Tensor, mask_background: bool = False, blur_alpha=0.1) -> torch.Tensor:
    '''
     blur loss
    '''
    # print("===============Start calculating mse_blur_planner=============")
    means = preds[:, 0, ...]  # BCHW -> BHW
    means = means.unsqueeze(1)

    target = target.unsqueeze(1)  # Get target to same shape as preds: B1HW
    idxs = ~target.isinf().detach()

    sq_diff = torch.square(means[idxs] - target[idxs]) / 2

    mse_loss = torch.mean(sq_diff)
    # print("===============mse_loss donw=============")


    # print(means.shape, target.shape)

    if len(means.shape) == 2:
        means = means.unsqueeze(0)
    if len(target.shape) == 2:
        target = target.unsqueeze(0)
    # print(means.shape, target.shape)
    
    means_sobel_feature_x, means_sobel_feature_y = calculate_gradient(means.clone())
    target_sobel_feature_x, target_sobel_feature_y = calculate_gradient(target.clone())

    # print("===============sobel_feature done=============")

    means_sobel_feature_x = torch.where(torch.isinf(means_sobel_feature_x), torch.zeros_like(means_sobel_feature_x), means_sobel_feature_x)
    means_sobel_feature_y = torch.where(torch.isinf(means_sobel_feature_y), torch.zeros_like(means_sobel_feature_y), means_sobel_feature_y)
    target_sobel_feature_x = torch.where(torch.isinf(target_sobel_feature_x), torch.zeros_like(target_sobel_feature_x), target_sobel_feature_x)
    target_sobel_feature_y = torch.where(torch.isinf(target_sobel_feature_y), torch.zeros_like(target_sobel_feature_y), target_sobel_feature_y)

    # print("===============inf replace done=============")

    blur_loss = berhu_loss(means_sobel_feature_x[idxs], target_sobel_feature_x[idxs]) + berhu_loss(means_sobel_feature_y[idxs], target_sobel_feature_y[idxs])

    # print("===============blur_loss done=============")

    loss = mse_loss + blur_alpha * blur_loss

    return loss, blur_loss, mse_loss




def mse_blur_scharr(
    preds: torch.Tensor, target: torch.Tensor, convert_index, coords_heal, mask_background: bool = False, blur_alpha=0.1
) -> torch.Tensor:
    means = preds[:, 0, ...]
    means = means.unsqueeze(1)  # means.dtype torch.float32

    target = target.unsqueeze(1)
    idxs = ~target.isinf().detach()

    sq_diff = torch.square(means[idxs] - target[idxs]) / 2

    mse_loss = torch.mean(sq_diff)  # mse_loss.dtype torch.float64

    if len(means.shape) == 1:
        means = means.unsqueeze(0)
    if len(target.shape) == 1:
        target = target.unsqueeze(0)

    target = target.to(torch.float32)
    means_sobel_feature_x = torch.zeros_like(means)
    means_sobel_feature_y = torch.zeros_like(means)
    target_sobel_feature_x = torch.zeros_like(target)
    target_sobel_feature_y = torch.zeros_like(target)
    means = torch.split(means, 16384, dim=2)
    targets = torch.split(target, 16384, dim=2)

    for index, mean in enumerate(means):
        mean = torch.where(torch.isinf(mean), torch.full_like(mean, 0), mean)
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        array_2d = torch_1d_to_2d(mean.squeeze(), convert_index).unsqueeze(1)  # torch.Size([4, 1, 128, 128])  torch.float32
        sobel_feature_x, sobel_feature_y = calculate_gradient_scharr(array_2d)  # torch.Size([4, 1, 128, 128])
        reconstructed_soel_feature_x = torch_2d_to_1d(sobel_feature_x, convert_index).unsqueeze(1)  # torch.Size([4,1,16384])
        reconstructed_soel_feature_y = torch_2d_to_1d(sobel_feature_y, convert_index).unsqueeze(1)  # torch.Size([4,1,16384])
        reconstructed_soel_feature_x = torch.where(torch.isinf(reconstructed_soel_feature_x), torch.zeros_like(reconstructed_soel_feature_x), reconstructed_soel_feature_x)
        reconstructed_soel_feature_y = torch.where(torch.isinf(reconstructed_soel_feature_y), torch.zeros_like(reconstructed_soel_feature_y), reconstructed_soel_feature_y)

        means_sobel_feature_x[:, :, index*16384:(index+1)*16384] = reconstructed_soel_feature_x
        means_sobel_feature_y[:, :, index*16384:(index+1)*16384] = reconstructed_soel_feature_y
    means_sobel_feature_x[:,:,coords_heal[:, 2] < 0.01] = 0
    means_sobel_feature_y[:,:,coords_heal[:, 2] < 0.01] = 0

    for index, target in enumerate(targets):
        target = torch.where(torch.isinf(target), torch.full_like(target, 0), target)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)

        array_2d = torch_1d_to_2d(target.squeeze(), convert_index).unsqueeze(1) # torch.Size([4, 1, 128, 128])

        sobel_feature_x, sobel_feature_y = calculate_gradient_scharr(array_2d)  # torch.Size([4, 1, 128, 128])
        reconstructed_soel_feature_x = torch_2d_to_1d(sobel_feature_x, convert_index).unsqueeze(1)  # torch.Size([4,1,16384])
        reconstructed_soel_feature_y = torch_2d_to_1d(sobel_feature_y, convert_index).unsqueeze(1)  # torch.Size([4,1,16384])
        reconstructed_soel_feature_x = torch.where(torch.isinf(reconstructed_soel_feature_x), torch.zeros_like(reconstructed_soel_feature_x), reconstructed_soel_feature_x)
        reconstructed_soel_feature_y = torch.where(torch.isinf(reconstructed_soel_feature_y), torch.zeros_like(reconstructed_soel_feature_y), reconstructed_soel_feature_y)

        target_sobel_feature_x[:, :, index*16384:(index+1)*16384] = reconstructed_soel_feature_x
        target_sobel_feature_y[:, :, index*16384:(index+1)*16384] = reconstructed_soel_feature_y
    target_sobel_feature_x[:,:,coords_heal[:, 2] < 0.01] = 0
    target_sobel_feature_y[:,:,coords_heal[:, 2] < 0.01] = 0

    blur_loss = berhu_loss(means_sobel_feature_x[idxs], target_sobel_feature_x[idxs]) + berhu_loss(means_sobel_feature_y[idxs], target_sobel_feature_y[idxs])
    loss = mse_loss + blur_alpha * blur_loss

    if torch.isnan(blur_loss):
        print("detected nan in blur_loss, ", blur_loss.cpu().detach().numpy())
        blur_loss = torch.where(torch.isnan(blur_loss), torch.zeros_like(blur_loss), blur_loss)
    
    if torch.isnan(mse_loss):
        print("detected nan in mse_loss, ", mse_loss.cpu().detach().numpy())
        mse_loss = torch.where(torch.isnan(mse_loss), torch.zeros_like(mse_loss), mse_loss)
    
    if torch.isnan(loss):
        print("detected nan in loss, ", loss.cpu().detach().numpy())
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return loss, blur_loss, mse_loss


def mse_blur_laplacian(
    preds: torch.Tensor, target: torch.Tensor, convert_index, coords_heal, mask_background: bool = False, blur_alpha=0.1
) -> torch.Tensor:
    means = preds[:, 0, ...]
    means = means.unsqueeze(1)  # means.dtype torch.float32

    target = target.unsqueeze(1)
    idxs = ~target.isinf().detach()

    sq_diff = torch.square(means[idxs] - target[idxs]) / 2

    mse_loss = torch.mean(sq_diff)  # mse_loss.dtype torch.float64

    if len(means.shape) == 1:
        means = means.unsqueeze(0)
    if len(target.shape) == 1:
        target = target.unsqueeze(0)

    target = target.to(torch.float32)
    means_laplacian_feature = torch.zeros_like(means)
    target_laplacian_feature = torch.zeros_like(target)
    means = torch.split(means, 16384, dim=2)
    targets = torch.split(target, 16384, dim=2)

    for index, mean in enumerate(means):
        mean = torch.where(torch.isinf(mean), torch.full_like(mean, 0), mean)
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        array_2d = torch_1d_to_2d(mean.squeeze(), convert_index).unsqueeze(1)  # torch.Size([4, 1, 128, 128])  torch.float32
        laplacian_feature = calculate_gradient_laplacian(array_2d)  # torch.Size([4, 1, 128, 128])
        reconstructed_laplacian_feature = torch_2d_to_1d(laplacian_feature, convert_index).unsqueeze(1)  # torch.Size([4,1,16384])
        reconstructed_laplacian_feature = torch.where(torch.isinf(reconstructed_laplacian_feature), torch.zeros_like(reconstructed_laplacian_feature), reconstructed_laplacian_feature)
        means_laplacian_feature[:, :, index*16384:(index+1)*16384] = reconstructed_laplacian_feature
    means_laplacian_feature[:,:,coords_heal[:, 2] < 0.01] = 0

    for index, target in enumerate(targets):
        target = torch.where(torch.isinf(target), torch.full_like(target, 0), target)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)
        array_2d = torch_1d_to_2d(target.squeeze(), convert_index).unsqueeze(1) # torch.Size([4, 1, 128, 128])
        laplacian_feature = calculate_gradient_laplacian(array_2d)  # torch.Size([4, 1, 128, 128])
        reconstructed_laplacian_feature = torch_2d_to_1d(laplacian_feature, convert_index).unsqueeze(1)  # torch.Size([4,1,16384])
        reconstructed_laplacian_feature = torch.where(torch.isinf(reconstructed_laplacian_feature), torch.zeros_like(reconstructed_laplacian_feature), reconstructed_laplacian_feature)
        target_laplacian_feature[:, :, index*16384:(index+1)*16384] = reconstructed_laplacian_feature
    target_laplacian_feature[:,:,coords_heal[:, 2] < 0.01] = 0

    blur_loss = berhu_loss(means_laplacian_feature[idxs], target_laplacian_feature[idxs])
    loss = mse_loss + blur_alpha * blur_loss

    if torch.isnan(blur_loss):
        print("detected nan in blur_loss, ", blur_loss.cpu().detach().numpy())
        blur_loss = torch.where(torch.isnan(blur_loss), torch.zeros_like(blur_loss), blur_loss)
    
    if torch.isnan(mse_loss):
        print("detected nan in mse_loss, ", mse_loss.cpu().detach().numpy())
        mse_loss = torch.where(torch.isnan(mse_loss), torch.zeros_like(mse_loss), mse_loss)
    
    if torch.isnan(loss):
        print("detected nan in loss, ", loss.cpu().detach().numpy())
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return loss, blur_loss, mse_loss


def mse_blur_LoG(
    preds: torch.Tensor, target: torch.Tensor, convert_index, coords_heal, mask_background: bool = False, blur_alpha=0.1
) -> torch.Tensor:
    means = preds[:, 0, ...]
    means = means.unsqueeze(1)  # means.dtype torch.float32

    target = target.unsqueeze(1)
    idxs = ~target.isinf().detach()

    sq_diff = torch.square(means[idxs] - target[idxs]) / 2

    mse_loss = torch.mean(sq_diff)  # mse_loss.dtype torch.float64

    if len(means.shape) == 1:
        means = means.unsqueeze(0)
    if len(target.shape) == 1:
        target = target.unsqueeze(0)

    target = target.to(torch.float32)
    means_LoG_feature = torch.zeros_like(means)
    target_LoG_feature = torch.zeros_like(target)
    means = torch.split(means, 16384, dim=2)
    targets = torch.split(target, 16384, dim=2)

    for index, mean in enumerate(means):
        mean = torch.where(torch.isinf(mean), torch.full_like(mean, 0), mean)
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        array_2d = torch_1d_to_2d(mean.squeeze(), convert_index).unsqueeze(1)  # torch.Size([4, 1, 128, 128])  torch.float32
        LoG_feature = calculate_gradient_LoG(array_2d)  # torch.Size([4, 1, 128, 128])
        reconstructed_LoG_feature = torch_2d_to_1d(LoG_feature, convert_index).unsqueeze(1)  # torch.Size([4,1,16384])
        reconstructed_LoG_feature = torch.where(torch.isinf(reconstructed_LoG_feature), torch.zeros_like(reconstructed_LoG_feature), reconstructed_LoG_feature)
        means_LoG_feature[:, :, index*16384:(index+1)*16384] = reconstructed_LoG_feature
    means_LoG_feature[:,:,coords_heal[:, 2] < 0.01] = 0

    for index, target in enumerate(targets):
        target = torch.where(torch.isinf(target), torch.full_like(target, 0), target)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)
        array_2d = torch_1d_to_2d(target.squeeze(), convert_index).unsqueeze(1) # torch.Size([4, 1, 128, 128])
        LoG_feature = calculate_gradient_LoG(array_2d)  # torch.Size([4, 1, 128, 128])
        reconstructed_LoG_feature = torch_2d_to_1d(LoG_feature, convert_index).unsqueeze(1)  # torch.Size([4,1,16384])
        reconstructed_LoG_feature = torch.where(torch.isinf(reconstructed_LoG_feature), torch.zeros_like(reconstructed_LoG_feature), reconstructed_LoG_feature)
        target_LoG_feature[:, :, index*16384:(index+1)*16384] = reconstructed_LoG_feature
    target_LoG_feature[:,:,coords_heal[:, 2] < 0.01] = 0

    blur_loss = berhu_loss(means_LoG_feature[idxs], target_LoG_feature[idxs])
    loss = mse_loss + blur_alpha * blur_loss

    if torch.isnan(blur_loss):
        print("detected nan in blur_loss, ", blur_loss.cpu().detach().numpy())
        blur_loss = torch.where(torch.isnan(blur_loss), torch.zeros_like(blur_loss), blur_loss)
    
    if torch.isnan(mse_loss):
        print("detected nan in mse_loss, ", mse_loss.cpu().detach().numpy())
        mse_loss = torch.where(torch.isnan(mse_loss), torch.zeros_like(mse_loss), mse_loss)
    
    if torch.isnan(loss):
        print("detected nan in loss, ", loss.cpu().detach().numpy())
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return loss, blur_loss, mse_loss


def get_depth_loss(
    common_depth_config: CommonDepthConfig, blur_alpha=0.1
) -> Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor]:
    if common_depth_config.use_logvar:
        print("Only mse base loss available for uncertainty estimation")
        return mean_log_var_loss

    losses = {
        "l2": mse,
        "l1": l1_loss,
        "huber": partial(huber_loss, delta=common_depth_config.huber_delta),
        "l2+blur": partial(
            mse_blur, blur_alpha=blur_alpha,
            convert_index = torch.from_numpy(
                np.load(_runtime_assets_path("hp16384_1d2d_index.npy"))
                ),  # torch.Size([16384])
            coords_heal = torch.from_numpy(
                np.load(_runtime_assets_path("coords-128-131072.npy"))
                ),  # torch.Size([16384, 2])
        ),
        "l2+blur+planner": partial(mse_blur_planner, blur_alpha=blur_alpha),
        
        "l2+blur+scharr": partial(
            mse_blur_scharr, blur_alpha=blur_alpha,
            convert_index = torch.from_numpy(np.load(_runtime_assets_path("hp16384_1d2d_index.npy"))),  
            coords_heal = torch.from_numpy(np.load(_runtime_assets_path("coords-128-131072.npy"))), 
        ),

        "l2+blur+laplacian": partial(
            mse_blur_laplacian, blur_alpha=blur_alpha,
            convert_index = torch.from_numpy(np.load(_runtime_assets_path("hp16384_1d2d_index.npy"))),  
            coords_heal = torch.from_numpy(np.load(_runtime_assets_path("coords-128-131072.npy"))), 
        ),

        "l2+blur+LoG": partial(
            mse_blur_LoG, blur_alpha=blur_alpha,
            convert_index = torch.from_numpy(np.load(_runtime_assets_path("hp16384_1d2d_index.npy"))),  
            coords_heal = torch.from_numpy(np.load(_runtime_assets_path("coords-128-131072.npy"))), 
        ),
    }
    print("--"*100)
    print(f"using the loss function: {common_depth_config.loss}")
    print("--"*100)

    return losses[common_depth_config.loss]
