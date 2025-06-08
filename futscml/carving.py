import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def energy_map(image_chw):
    # compute gradient with convolution
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)[None, None]
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)[None, None]
    gradient_x = torch.abs(F.conv2d(image_chw[:, None], sobel_x, padding=1))[:, 0]
    gradient_y = torch.abs(F.conv2d(image_chw[:, None], sobel_y, padding=1))[:, 0]

    energy = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
    # save energy map for visualization
    # energy_map = energy.cpu().numpy()
    # energy_map = (energy_map - energy_map.min()) / (energy_map.max() - energy_map.min())
    # energy_map = (energy_map.mean(axis=0) * 255).astype(np.uint8)
    # cv2.imwrite('/home/spetlrad/datadata/ddpmst/experiments/STALP/cersei/9016251/energy_map.png', energy_map)
    return torch.mean(energy, dim=0)


def find_vertical_seam(energy, variant):
    h, w = energy.shape
    seam = torch.zeros(h, dtype=torch.long)
    cost = energy.clone()

    argmin_f = torch.argmin
    minimum_f = torch.minimum
    max_f = torch.max
    if variant == 'adding':
        argmin_f = torch.argmax
        minimum_f = torch.maximum
        max_f = torch.min

    cost = torch.cat([torch.ones(h, 1, dtype=cost.dtype) * max_f(energy), cost, torch.ones(h, 1, dtype=cost.dtype) * max_f(energy)], dim=1)

    for i in range(1, h):
        right = minimum_f(cost[i - 1, 1:-1], cost[i - 1, :-2])
        left = minimum_f(cost[i - 1, 1:-1], cost[i - 1, 2:])
        cost[i, 1:-1] += minimum_f(left, right)
        cost[i, 0] += cost[i - 1, 0] + max_f(energy)
        cost[i, -1] += cost[i - 1, -1] + max_f(energy)

    cost = cost[:, 1:-1]

    seam[-1] = argmin_f(cost[-1])
    for i in reversed(range(h - 1)):
        prev_x = seam[i + 1]
        if prev_x == 0:
            seam[i] = argmin_f(cost[i, :2])
        else:
            seam[i] = prev_x + argmin_f(cost[i, prev_x - 1:prev_x + 2]) - 1

    return seam


def remove_vertical_seam(image, seam, mask=None):
    c, h, w = image.shape
    if mask is None:
        mask = torch.ones(h, w, dtype=torch.bool)
        mask[torch.arange(h), seam] = False

    image_out = torch.zeros(c, h, w - 1, dtype=image.dtype)
    for i in range(c):
        image_out[i] = image[i][mask].view(h, w - 1)

    return image_out, mask


def vertical_seam_carving(energy, num_vertical_seams, image_chw, additional_tensors, variant):
    for _ in range(num_vertical_seams):
        seam = find_vertical_seam(energy, variant=variant)
        image_chw, mask = remove_vertical_seam(image_chw, seam)
        for tensor_idx, tensor in enumerate(additional_tensors):
            additional_tensors[tensor_idx], _ = remove_vertical_seam(tensor, seam, mask=mask)
        h, w = energy.shape
        energy = energy[mask].view(h, w - 1)

    return energy, image_chw, additional_tensors

def seam_carve(image_chw, additional_tensors, num_vertical_seams, num_horizontal_seams, variant='carving'):
    assert variant in ['carving', 'adding']
    energy = energy_map(image_chw)

    energy, image_chw, additional_tensors = (
        vertical_seam_carving(energy, num_vertical_seams, image_chw, additional_tensors, variant))

    energy = rearrange(energy, 'h w -> w h')
    image_chw = rearrange(image_chw, 'c h w -> c w h')
    for tensor_idx, tensor in enumerate(additional_tensors):
        additional_tensors[tensor_idx] = rearrange(tensor, 'c h w -> c w h')

    energy, image_chw, additional_tensors = (
        vertical_seam_carving(energy, num_horizontal_seams, image_chw, additional_tensors, variant))

    image_chw = rearrange(image_chw, 'c w h -> c h w')
    for tensor_idx, tensor in enumerate(additional_tensors):
        additional_tensors[tensor_idx] = rearrange(tensor, 'c w h -> c h w')

    return image_chw, *additional_tensors

