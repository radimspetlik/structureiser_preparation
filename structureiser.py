import random
import os
from contextlib import suppress as suppress
from argparse import ArgumentParser

import yaml
import torch
import torch as th
import torch.nn as nn
import torch.optim as opt
import torchvision.models as models
import torchvision.io
import numpy as np
import cv2
from torchvision.models import VGG19_Weights, VGG19_BN_Weights
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import functional as Tfunc
from tqdm import tqdm
from einops import repeat, rearrange

from vgg_adapter import ResidualConvLayer, Transformer2DWrapper
from futscml.sds import SDSControlNet


class ImageToImageGenerator_JohnsonFutschik(nn.Module):
    def __init__(self, norm_layer='batch_norm', use_bias=False, resnet_blocks=9, tanh=False,
                 filters=(64, 128, 128, 128, 128, 64), input_channels=3, output_channels=3,
                 append_blocks=None, blur_pool=False, conv_padding_mode='replicate',
                 config=None, **kwargs):
        super().__init__()
        assert norm_layer in [None, 'batch_norm', 'instance_norm']
        self.norm_layer = None
        if norm_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            self.norm_layer = nn.InstanceNorm2d
        self.use_bias = use_bias
        self.blur_pool = blur_pool
        self.conv_padding_mode = conv_padding_mode
        self.config = config
        self.use_attention = config['use_attention'] if 'use_attention' in config else False
        self.resnet_blocks = resnet_blocks
        self.append_blocks = append_blocks

        self.conv0 = self.relu_layer(in_filters=input_channels, out_filters=filters[0],
                                     size=7, stride=1, padding=3, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2),
                                     conv_padding_mode=self.conv_padding_mode)

        self.conv1 = self.relu_layer(in_filters=filters[0], out_filters=filters[1],
                                     size=3, stride=2, padding=1, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2),
                                     blur_pool=self.blur_pool,
                                     conv_padding_mode=self.conv_padding_mode)

        self.conv2 = self.relu_layer(in_filters=filters[1], out_filters=filters[2],
                                     size=3, stride=2, padding=1, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2),
                                     blur_pool=self.blur_pool,
                                     conv_padding_mode=self.conv_padding_mode)

        self.resnets = nn.ModuleList()
        for i in range(self.resnet_blocks):
            self.resnets.append(
                self.resnet_block(in_filters=filters[2], out_filters=filters[2],
                                  size=3, stride=1, padding=1, bias=self.use_bias,
                                  norm_layer=self.norm_layer, nonlinearity=nn.ReLU()))

        self.upconv2 = self.upconv_layer(in_filters=filters[3] + filters[2], out_filters=filters[4],
                                         norm_layer=self.norm_layer, nonlinearity=nn.ReLU())

        self.upconv1 = self.upconv_layer(in_filters=filters[4] + filters[1], out_filters=filters[4],
                                         norm_layer=self.norm_layer, nonlinearity=nn.ReLU())

        self.conv_11 = nn.Sequential(
            nn.Conv2d(filters[0] + filters[4] + input_channels, filters[5],
                      kernel_size=7, stride=1, padding=3, bias=self.use_bias, padding_mode=self.conv_padding_mode),
            nn.ReLU()
        )

        # initialize context to gaussian random noise
        self.context = torch.randn(1, 100,
                                   config['attention_context_dim'] if 'attention_context_dim' in config and config[
                                       'attention_context_dim'] is not None else 1)

        self.end_blocks = None
        if self.append_blocks is not None:
            self.end_blocks = nn.Sequential(
                nn.Conv2d(filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1,
                          padding_mode=self.conv_padding_mode),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=filters[5]),
                nn.Conv2d(filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1,
                          padding_mode=self.conv_padding_mode),
                nn.ReLU()
            )

        self.conv_12 = nn.Sequential(
            nn.Conv2d(filters[5], output_channels, kernel_size=1, stride=1, padding=0, bias=True))
        if tanh:
            self.conv_12.add_module('tanh', nn.Tanh())

    def forward(self, x):
        output_0 = self.conv0(x)
        output_1 = self.conv1(output_0)
        output = self.conv2(output_1)
        output_2 = self.conv2(output_1)
        for layer in self.resnets:
            output = layer(output) + output

        output = self.upconv2(torch.cat((output, output_2), dim=1))
        output = self.upconv1(torch.cat((output, output_1), dim=1))
        output = self.conv_11(torch.cat((output, output_0, x), dim=1))
        if self.end_blocks is not None:
            output = self.end_blocks(output)
        output = self.conv_12(output)
        return output

    def relu_layer(self, in_filters, out_filters, size, stride, padding, bias,
                   norm_layer, nonlinearity, blur_pool=False, conv_padding_mode='replicate'):
        out = []
        if blur_pool:
            assert size in [3, 5]
            assert stride > 1
            out.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                                 kernel_size=size, stride=1, padding=padding, bias=bias,
                                 padding_mode=conv_padding_mode))
            out.append(BlurPool2d(channels=out_filters, kernel_size=size, stride=stride))
        else:
            out.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                                 kernel_size=size, stride=stride, padding=padding, bias=bias,
                                 padding_mode=conv_padding_mode))
        if norm_layer:
            out.append(norm_layer(num_features=out_filters))
        if nonlinearity:
            out.append(nonlinearity)
        return nn.Sequential(*out)

    def resnet_block(self, in_filters, out_filters, size, stride, padding, bias,
                     norm_layer, nonlinearity):
        out = []
        if nonlinearity:
            out.append(nonlinearity)
        out.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                             kernel_size=size, stride=stride, padding=padding, bias=bias))
        if norm_layer:
            out.append(norm_layer(num_features=out_filters))
        if nonlinearity:
            out.append(nonlinearity)
        out.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                             kernel_size=size, stride=stride, padding=padding, bias=bias))
        return nn.Sequential(*out)

    def upconv_layer(self, in_filters, out_filters, norm_layer, nonlinearity):
        out = []
        out.append(SmoothUpsampleLayer(in_filters, out_filters))
        if norm_layer:
            out.append(norm_layer(num_features=out_filters))
        if nonlinearity:
            out.append(nonlinearity)
        return nn.Sequential(*out)


class ImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.objective = nn.MSELoss()

    def forward(self, x, y):
        return self.objective(x, y)


def rand_ind_2d(h, w, numind, device, unique=False):
    if not unique:
        hc = torch.randint(low=0, high=h, size=(numind,), device=device)
        wc = torch.randint(low=0, high=w, size=(numind,), device=device)
    else:
        hc = (torch.randperm(h * w, device=device) // w)[:numind]
        wc = (torch.randperm(h * w, device=device) // h)[:numind]
    return hc, wc


class Vgg19_Extractor(nn.Module):
    def __init__(self, capture_layers):
        super().__init__()
        self.vgg_layers = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # Load the old model if requested
        # self.vgg_layers.load_state_dict(torch.load('/home/futscdav/model_vault/old_vgg_converted_new_transform.pth'))
        self.vgg_layers = self.vgg_layers.features
        self.len_layers = 37  # len(self.vgg_layers)

        for param in self.parameters():
            param.requires_grad = False
        self.capture_layers = capture_layers

    def forward(self, x):
        feat = []
        if -1 in self.capture_layers:
            feat.append(x)
        i = 0
        for mod in self.vgg_layers:
            x = mod(x)
            i += 1
            if i in self.capture_layers:
                feat.append(x)
        return feat


class InnerProductLoss(nn.Module):
    def __init__(self, capture_layers, device):
        super().__init__()
        self.layers = capture_layers
        self.device = device
        self.vgg = Vgg19_Extractor(capture_layers).to(device)
        self.stored_mean = (torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1, -1, 1, 1))
        self.stored_std = (torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1, -1, 1, 1))
        self.gmm = GramMatrix()
        self.dist = nn.MSELoss()
        self.cache: Dict[float, List[torch.Tensor]] = {0.: [torch.empty((0))]}  # torch.Tensor
        self.attention_layers = []

    def extractor(self, x):
        # remap x to vgg range
        x = (x + 1.) / 2.
        x = x - self.stored_mean
        x = x / self.stored_std
        res = self.vgg(x)
        return res

    def run_scale(self, frame_y, pure_y, frame_x, pure_x, cache_y2: bool = True, scale: float = 1.):
        frame_y = F.interpolate(frame_y, scale_factor=float(scale), mode='bilinear', align_corners=False,
                                recompute_scale_factor=False)
        feat_frame_y = self.extractor(frame_y)
        if cache_y2:
            if scale not in self.cache:
                pure_y = F.interpolate(pure_y, scale_factor=scale, mode='bilinear', align_corners=False)
                self.cache[scale] = [self.gmm(l) for idx, l in enumerate(self.extractor(pure_y))]
            gmm_pure_y = self.cache[scale]
        else:
            pure_y = F.interpolate(pure_y, scale_factor=scale, mode='bilinear', align_corners=False)
            feat_pure_y = self.extractor(pure_y)
            gmm_pure_y = [self.gmm(l) for idx, l in enumerate(feat_pure_y)]

        # loss : List[torch.Tensor] = []
        loss = torch.empty((len(feat_frame_y),)).to(frame_y.device)
        for l in range(len(feat_frame_y)):
            gmm_frame_y = self.gmm(feat_frame_y[l])
            assert gmm_pure_y[l].shape[0] == gmm_frame_y.shape[0]
            assert not (gmm_pure_y[l].requires_grad)
            dist = self.dist(gmm_pure_y[l].detach(), gmm_frame_y)
            loss[l] = dist
        return torch.sum(loss)

    def forward(self, frame_y, pure_y, frame_x, pure_x, cache_y2: bool = True):
        scale_1_loss = self.run_scale(frame_y, pure_y, frame_x, pure_x, cache_y2, scale=1.0)
        # scale_2_loss = self.run_scale(y1, y2, cache_y2, scale=0.5)
        return scale_1_loss


class InferDataset(Dataset):
    def __init__(self, frames_dir, xform):
        self.root = os.path.join(frames_dir)
        self.frames = images_in_directory(self.root)
        self.tensors = []
        self.stems = []
        self.xform = xform
        for frame in self.frames:
            stem, _ = os.path.splitext(frame)
            x = pil_loader(os.path.join(self.root, frame))
            self.tensors.append(self.xform(x))
            self.stems.append(stem)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.stems[idx], self.tensors[idx]


class NullAugmentations:
    def __init__(self):
        pass

    def __call__(self, *items):
        return items


class ShapeAugmentations:
    def __init__(self):
        self.angle_min = -9
        self.angle_max = 9
        self.hflip_chance = .5

    def rng(self, min, max):
        return random.random() * (max - min) + min

    def __call__(self, *items):
        angle = self.rng(self.angle_min, self.angle_max)
        hflip = self.rng(0, 1) < self.hflip_chance

        def transform(x):
            if hflip:
                x = Tfunc.hflip(x)
            x = Tfunc.rotate(x, angle)
            return x

        augmented_items = []
        for item in items:
            augmented_items.append(transform(item))
        return augmented_items


class ColorAugmentations:
    def __init__(self):
        self.hflip_chance = 0.5
        self.adjust_contrast_min = 0.7
        self.adjust_contrast_max = 1.3
        self.adjust_hue_min = -0.1
        self.adjust_hue_max = 0.1
        self.adjust_saturation_min = 0.7
        self.adjust_saturation_max = 1.3

    def rng(self, min, max):
        return random.random() * (max - min) + min

    def __call__(self, *items):
        cnts = self.rng(self.adjust_contrast_min, self.adjust_contrast_max)
        hue = self.rng(self.adjust_hue_min, self.adjust_hue_max)
        sat = self.rng(self.adjust_saturation_min, self.adjust_saturation_max)

        def transform(x):
            x = Tfunc.adjust_contrast(x, cnts)
            x = Tfunc.adjust_hue(x, hue)
            x = Tfunc.adjust_saturation(x, sat)
            return x

        augmented_items = []
        for item in items:
            augmented_items.append(transform(item))
        return augmented_items


class TrainingDataset(Dataset):
    def __init__(self, frames_dir, keyframe_dir, xform, data_aux, disable_augment=False):
        self.frames_dir = frames_dir
        self.keyframe_dir = keyframe_dir
        self.xform = xform

        keys_in = [f for f in images_in_directory(self.keyframe_dir)]
        keys_out = [f for f in images_in_directory(self.keyframe_dir)]
        self.keypair_files = list(zip(keys_in, keys_out))

        print(f"Found {len(self.keypair_files)} keyframe pairs in {self.keyframe_dir}")

        self.aux_data = data_aux
        self.pairs = []
        self.stems = []
        for keyframe in self.keypair_files:
            key_in, key_out = keyframe
            stem, _ = os.path.splitext(key_in)
            keyframe_in = pil_loader(os.path.join(self.frames_dir, key_in))
            keyframe_out = pil_loader(os.path.join(self.keyframe_dir, key_out))
            self.pairs.append((keyframe_in, keyframe_out))
            self.stems.append(stem)
        self.shape_augment = ShapeAugmentations() if not disable_augment else NullAugmentations()
        self.color_augment = ColorAugmentations()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # choose a random sample from the dataset, different on each call, along with the original keyframe

        shaped0, shaped1 = self.shape_augment(self.pairs[idx][0], self.pairs[idx][1])
        colored0, = self.color_augment(shaped0)

        return self.stems[idx], self.xform(colored0), self.xform(shaped1), self.xform(self.pairs[idx][0]), self.xform(
            self.pairs[idx][1])
        # return self.pairs[idx][0], self.pairs[idx][1]


def log_verification_images(config, log, step, model, dataset, transform, additional_image=None):
    with torch.no_grad():
        model.eval()
        idx, example = enumerate(dataset).__next__()
        f = example[1].to(guess_model_device(model))
        if config['model_params']['input_channels'] == 4:
            ones = th.ones_like(f[:, :1], device=device)
            f = th.cat([ones, f], dim=1)
        pred = model(f)
        if additional_image is not None:
            log.log_image('Keyframe', pil_to_np(transform(additional_image[0].data.cpu())), step, format='HWC')
        log.log_image('First Unpaired Frame', pil_to_np(transform(pred[0].data.cpu())), step, format='HWC')
        log.flush()


def log_verification_video(config, log, step, label, model, dataset, transform, shape, max_frames=None, fps=1):
    with torch.no_grad():
        model.eval()
        vid_tensor = torch.empty(1, min(max_frames, len(dataset.dataset)) if max_frames is not None else len(
            dataset.dataset), shape[1], shape[2], shape[3], device=device)
        i = 0
        for _, batch in enumerate(dataset):
            _, b = batch
            if max_frames is not None and i >= max_frames: break
            if len(b.shape) == 3: b = b.unsqueeze(0)
            b = tensor_resample(b.to(device), [shape[2], shape[3]])
            if config['model_params']['input_channels'] == 4:
                ones = th.ones_like(b[:, :1], device=device)
                b = th.cat([ones, b], dim=1)
            frame = model(b)
            for j in range(frame.shape[0]):
                vid_tensor[:, i, :, :, :] = transform.denormalize_tensor(frame[j:j + 1])
            i = i + 1
        if torch.numel(vid_tensor) > 0:
            # log.log_video(label, vid_tensor.cpu(), step, fps=fps)
            torchvision.io.write_video(log.location() + f"/{step}.mp4",
                                       (vid_tensor[0].cpu().permute((0, 2, 3, 1)) * 255).to(torch.uint8), fps=8)


class ControlProcessor:
    def __init__(self, config, processor):
        self.config = config
        self.processor = processor
        self.control_type = self.config.get('control_type', 'direct')
        self.cache = {}

    def direct_control(self, stem, frame_x_0_255):
        control_image_0_255 = self.processor(rearrange(frame_x_0_255.cpu(), 'c h w -> h w c'), stem=stem,
                                             return_pil=False)
        return th.tensor(rearrange(control_image_0_255, 'h w c -> 1 c h w'), device=frame_x_0_255.device)

    def differentiable_control(self, stem, frame_x_0_255):
        control_image_0_255 = self.processor(frame_x_0_255, stem=None, return_pil=False)
        return control_image_0_255

    def warped_control(self, key_stem, flow_key, frame_x_0_255, keyframe_x_m1p1, keyframe_y_m1p1):
        frame_oflow_path = os.path.join(self.config['oflow_dir'], f'{flow_key}.flowouX16.pkl')

        oflow, occlusions, uncertainty = read_flowou_X16(frame_oflow_path)
        oflow_t = torch.tensor(oflow).to(keyframe_y_m1p1.device)
        occlusions_t = rearrange(torch.tensor(occlusions).to(keyframe_y_m1p1.device) * 255, 'c h w -> h w c')

        control_image_0_255 = self.direct_control(key_stem, keyframe_y_m1p1[0] * 127.5 + 127.5)
        warped_control_image_0_255, warped_keyframe_mask_t_0_255 = warp(oflow_t, occlusions_t,
                                                                        rearrange(control_image_0_255,
                                                                                  '1 c h w -> h w c'))

        warped_control_image_0_255 = (warped_keyframe_mask_t_0_255 / 255.0) * warped_control_image_0_255

        # cv2.imwrite(f'{flow_key}.png', warped_control_image_0_255.cpu().numpy().astype(np.uint8))

        return rearrange(warped_control_image_0_255, 'h w c -> 1 c h w').to(keyframe_y_m1p1.device)

    def __call__(self, key_stems, stems, frame_x_m1p1, keyframe_x_m1p1, keyframe_y_m1p1):
        frame_x_0_255 = frame_x_m1p1 * 127.5 + 127.5
        control_images_0_255 = []
        for frame_x_idx in range(frame_x_0_255.shape[0]):
            if self.control_type == 'direct':
                control_images_0_255.append(self.direct_control(stems[frame_x_idx], frame_x_0_255[frame_x_idx]))
            elif self.control_type == 'differentiable':
                control_images_0_255.append(
                    self.differentiable_control(stems[frame_x_idx], frame_x_0_255[frame_x_idx:frame_x_idx + 1]))
            elif self.control_type == 'warped':
                assert len(key_stems) == 1
                cache_key = f'{key_stems[0]}--{stems[frame_x_idx]}'
                if cache_key not in self.cache:
                    self.cache[cache_key] = self.warped_control(key_stems[0], cache_key, frame_x_0_255[frame_x_idx],
                                                                keyframe_x_m1p1, keyframe_y_m1p1)
                control_images_0_255.append(self.cache[cache_key])
            else:
                raise ValueError(f'Unknown control type: {self.control_type}')

        control_image_0_1 = torch.cat(control_images_0_255, dim=0) / 255.0

        return control_image_0_1


def cut_patches(images):
    patch_size = config.patch_size

    y = np.random.randint(0, patch_size)
    x = np.random.randint(0, patch_size)

    images = [image[..., y:-(patch_size - y), x:-(patch_size - x)] for image in images]

    images = [th.nn.functional.unfold(image, kernel_size=patch_size, stride=patch_size) for image in images]
    images = [rearrange(image, 'b (c k1 k2) n -> (b n) c k1 k2', k1=patch_size, k2=patch_size) for image in images]

    num_indices = config.num_patches
    image_random_indices = th.randint(0, images[0].shape[0], (num_indices,))
    images = [image[image_random_indices] for image in images]

    return images


import bisect
from typing import TypeVar, Optional, Dict

V = TypeVar("V")


def closest_value(d: Dict[int, V], x: int, width: int) -> Optional[V]:
    """
    Given a dict `d` with integer keys and any values,
    snap the input `x` to the nearest key’s value—unless `x`
    falls within `width` “no-man’s-land” around a midpoint
    between two adjacent keys, in which case return None.
    """
    if not d:
        return None

    # Sort the keys once
    keys = sorted(d)
    n = len(keys)

    # If x is before the first key or after the last, just clamp
    if x <= keys[0]:
        return keys[0]
    if x >= keys[-1]:
        return keys[-1]

    # If x exactly matches a key, return immediately
    if x in d:
        return x

    # Find the place where x would go in the sorted key list
    idx = bisect.bisect_left(keys, x)
    low = keys[idx - 1]
    high = keys[idx]

    # Compute the exact midpoint (float)
    mid = (low + high) / 2

    # Half-range of the “no-man’s-land” window
    half = width // 2

    # If x is too close to the midpoint, we’re in no-man’s-land
    if abs(x - mid) <= half:
        return None

    # Otherwise, snap to the closer of low/high
    target_key = low if x < mid else high
    return target_key


def train_with_similarity(config, model, iters, key_weight, style_weight, structure_weight, dataset_train, dataset_aux,
                          dataset_val, transform, device, log):
    model.to(device)
    reconstruction_model.to(device)

    if key_weight > 0.:
        image_loss = ImageLoss()

    params_to_optimize = list(model.parameters())
    params_to_optimize += list(reconstruction_model.parameters())

    optimizer = opt.AdamW(params_to_optimize, lr=3e-5)  # RMSprop works at 2e-4
    aux_sample = InfiniteDatasetSampler(dataset_aux)
    ebest = float('inf')

    log_image_update_every = 5000
    log_video_update_every = config['log_video_update_every'] if 'log_video_update_every' in config else 5000

    stopwatch = Stopwatch()
    snapshots = [
        (1 * 5 * 60, '05m'),
        (1 * 15 * 60, '15m'),
        (1 * 30 * 60, '30m'),
        (1 * 60 * 60, '01h'),
        (2 * 60 * 60, '02h'),
        (3 * 60 * 60, '03h'),
        (6 * 60 * 60, '06h'),
        (12 * 60 * 60, '12h')
    ]

    image_error_weight_annealing = ValueAnnealing(key_weight * 5, key_weight / 1024, 20_000)
    trange = tqdm(range(iters))

    control_processor = ControlProcessor(config, processor)

    for epoch in trange:
        # Reset to train mode & init random with new seed
        model.train()
        np.random.seed(epoch)

        with suppress():
            for batch_idx, batch in enumerate(dataset_train):
                error, style_loss, key_loss, structure_loss = 0, 0, 0, 0, 0, 0
                key_stems, *batch = batch
                batch = [thing.to(device) for thing in batch]
                keyframe_x, keyframe_y, pure_x, pure_y = batch


                _, aux_batch = aux_sample()
                stems, frame_x = aux_batch

                if config['lambda_sd'] > 0. or config.get('cdiff_weight', 0.) > 0.:
                    control_image_0_1 = control_processor(key_stems, stems, frame_x, keyframe_x, keyframe_y)
                    control_image_0_1 = control_image_0_1.to(device)

                frame_x = frame_x.to(device)

                optimizer.zero_grad()

                with suppress():
                    keyframe_x_ = keyframe_x.clone()
                    y = model(keyframe_x_)

                # L1 Loss Calculation
                key_loss += key_weight * image_loss(y, keyframe_y)

                with suppress():
                    frame_y = model(frame_x.clone())

                with suppress():
                    style_loss = style_weight * similarity_loss(frame_y, pure_y, frame_x, pure_x,
                                                                cache_y2=True)
                    structure_loss = config['lambda_sd'] * guidance_sd.train_step(frame_y / 2.0 + 0.5,
                                                                            control_image_0_1,
                                                                            epoch=epoch,
                                                                            inference_step=config['inference_step'])

                # Track values for logging
                error = style_loss + structure_loss + key_loss

                tracked_scalars = ['image_error', 'similarity_error', 'sds_loss', 'error']
                scalars = {name: value for name, value in locals().items() if name in tracked_scalars}
                log.log_multiple_scalars(scalars, epoch)

                error.backward()
                optimizer.step()

                trange.set_postfix({'err': f'{error:0.5f}',
                                    'key': f'{key_loss:0.5f}',
                                    'sty': f'{style_loss:0.5f}',
                                    'str': f'{structure_loss:0.5f}',
                                    })

            # Take snapshots
            for deadline, snap in snapshots:
                if stopwatch.just_passed(deadline):
                    log.log_checkpoint({'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()},
                                       f'{snap}_snapshot')
                    log.log_file(log._best_checkpoint_location(), output_name=f'{snap}_snapshot_best.pth')

            if config['max_time_minutes'] is not None and stopwatch.just_passed(config['max_time_minutes'] * 60):
                log_verification_video(config, log, epoch, 'Auxiliary Frames', model, dataset_aux, transform, y.shape,
                                       max_frames=None)
                log.log_checkpoint({'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, 'latest')
                print("Maximum time passed, exiting..")
                return

            if epoch % log_image_update_every == 0 and epoch != 0:
                log_verification_images(config, log, epoch, model, dataset_aux, transform, y)

                if error < ebest:
                    ebest = error
                    log.log_checkpoint_best({'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()})

            if epoch % log_video_update_every == 0 and epoch != 0:
                if dataset_val is not None:
                    log_verification_video(config, log, epoch, 'Validation Frames', model, dataset_val, transform,
                                           y.shape, max_frames=500, fps=25)
                log_verification_video(config, log, epoch, 'Auxiliary Frames', model, dataset_aux, transform, y.shape,
                                       max_frames=None)
                log.flush()
                log.log_checkpoint({'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, 'latest')
    log.log_checkpoint({'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, 'latest')


class CachedControlProcessor:
    def __init__(self):
        self.cache = {}

    def call(self, input_image, *_, **__):
        raise NotImplementedError

    def __call__(self, frame, stem=None, *args, **kwargs):
        if stem is not None and stem in self.cache:
            return self.cache[stem]

        if not isinstance(frame, np.ndarray):
            frame = np.array(frame, dtype=np.uint8)
        frame = HWC3(frame)

        control_image = self.call(frame, *args, **kwargs)

        self.cache[stem] = control_image
        return self.cache[stem]


class CannyProcessor(CachedControlProcessor):
    def __init__(self, low_threshold=100, high_threshold=200):
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def call(self, frame, *_, **__):
        canny = cv2.Canny(frame, self.low_threshold, self.high_threshold)
        return np.concatenate([canny[..., None], canny[..., None], canny[..., None]], axis=-1)


class DepthProcessor(CachedControlProcessor):
    def __init__(self):
        super().__init__()
        from transformers import pipeline
        self.depth_estimator = pipeline('depth-estimation')

    def call(self, input_image, *_, **__):
        image = self.depth_estimator(Image.fromarray(input_image))['depth']
        image = np.array(image)
        return np.concatenate([image[..., None], image[..., None], image[..., None]], axis=-1)


class CacheControlProcessor(CachedControlProcessor):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def __call__(self, frame, stem=None, *args, **kwargs):
        if stem is not None and stem in self.cache:
            return self.cache[stem]

        self.cache[stem] = self.processor(frame, *args, **kwargs)
        return self.cache[stem]


def prepare_cldm(config):
    if config['cldm_type'] == 'lineart':
        guidance_sd = SDSControlNet(device, fp16=False)
        processor = CacheControlProcessor(LineartDetector.from_pretrained("lllyasviel/Annotators"))
    elif config['cldm_type'] == 'difflineart':
        guidance_sd = SDSControlNet(device, fp16=False)
        processor = CacheControlProcessor(DifferentiableLineArt.from_pretrained("lllyasviel/Annotators"))
    elif config['cldm_type'] == 'canny':
        guidance_sd = SDSControlNet(device, fp16=False, checkpoint='lllyasviel/control_v11p_sd15_canny')
        processor = CannyProcessor()
    elif config['cldm_type'] == 'softedge':
        guidance_sd = SDSControlNet(device, fp16=False, checkpoint='lllyasviel/control_v11p_sd15_softedge')
        processor = CacheControlProcessor(PidiNetDetector.from_pretrained('lllyasviel/Annotators'))
    elif config['cldm_type'] == 'depth':
        guidance_sd = SDSControlNet(device, fp16=False, checkpoint='lllyasviel/control_v11f1p_sd15_depth')
        processor = DepthProcessor()
    else:
        raise ValueError(f"Unknown CLDM type {config['cldm_type']}")
    guidance_sd.get_text_embeds([config['prompt'] if config['prompt'] is not None else ""],
                                [config['negative_prompt'] if config['negative_prompt'] is not None else ""])

    return guidance_sd, processor


class ModelMock:
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def to(self, device):
        pass

    def eval(self):
        pass

    def train(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def parameters(self):
        return []

    def zero_grad(self):
        pass


import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------
# Patchify and Unpatchify
# --------------------
class Patchify(nn.Module):
    """
    Splits an image of shape (B, C, H, W) into a tensor of patches (B, N, C, p, p),
    where p=patch_size, N=(H/p)*(W/p).
    Uses nn.Unfold internally.
    """

    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: patches of shape (B, N, C, p, p)
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Extract patches using unfold:
        #   unfold => (B, C * p^2, N) where N = number of patches = (H/p)*(W/p)
        patches = F.unfold(x, kernel_size=p, stride=p)  # (B, C*p*p, N)

        # Rearrange to (B, N, C, p, p)
        patches = patches.transpose(1, 2)  # (B, N, C*p*p)
        patches = patches.view(B, -1, C, p, p)  # (B, N, C, p, p)

        return patches


class Unpatchify(nn.Module):
    """
    Reconstructs (folds) patches of shape (B, N, C, p, p) back into a full image (B, C, H, W).
    Uses nn.Fold internally. You need to specify the original image size or store it.
    """

    def __init__(self, patch_size=16, image_size=(128, 128)):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size  # (H, W)

    def forward(self, patches):
        """
        patches: (B, N, C, p, p)
        return: (B, C, H, W)
        """
        B, N, C, p, _ = patches.shape
        H, W = self.image_size

        # Invert the shape transformations used in patchify:
        #   currently (B, N, C, p, p) => reshape to (B, C*p*p, N)
        patches = patches.view(B, N, C * p * p).transpose(1, 2)  # (B, C*p*p, N)

        # Use fold to go back to (B, C, H, W)
        out = F.fold(patches, output_size=(H, W), kernel_size=p, stride=p)
        return out


# --------------------
# Local CNN for each patch
# --------------------
class LocalPatchCNN(nn.Module):
    """
    Example local CNN that processes each patch from (C, p, p) -> a hidden representation (64 channels).
    You can customize kernel sizes, #channels, etc.
    """

    def __init__(self, in_channels=3, hidden_dim=64, patch_size=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.patch_size = patch_size

    def forward(self, x):
        # x: (B, in_channels, p, p)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out  # shape: (B, hidden_dim, p, p)


# --------------------
# Patch Encoder
# --------------------
class PatchEncoder(nn.Module):
    """
    Encodes one patch into a latent vector.
    1) local CNN to get a feature map
    2) adaptive pooling -> single vector
    3) linear projection to latent_dim
    """

    def __init__(self, in_channels=3, hidden_dim=64, latent_dim=128, patch_size=16):
        super().__init__()
        self.local_cnn = LocalPatchCNN(in_channels, hidden_dim, patch_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x shape: (B, in_channels, p, p)
        feat_map = self.local_cnn(x)  # (B, hidden_dim, p, p)
        pooled = self.pool(feat_map)  # (B, hidden_dim, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, hidden_dim)
        latent = self.fc(pooled)  # (B, latent_dim)
        return latent


# --------------------
# Patchify Autoencoder
# --------------------
class PatchifyAutoencoder(nn.Module):
    """
    Full autoencoder that:
      1) Patchifies an image
      2) Encodes each patch into a latent vector
      3) Optionally applies a global aggregator
      4) Decodes each patch from the latent vector
      5) Unpatchifies to reconstruct the full image
    """

    def __init__(self,
                 patch_size=16,
                 image_size=(512, 512),
                 in_channels=3,
                 hidden_dim=64,
                 latent_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size

        # Modules for patchification
        self.patchify = Patchify(patch_size)
        self.unpatchify = Unpatchify(patch_size, image_size)

        # Patch-level encoder and decoder
        self.encoder = PatchEncoder(in_channels, hidden_dim, latent_dim, patch_size)

        # A simple aggregator: linear transformation of latent vectors
        # (could be a transformer, RNN, attention, etc. if you want to capture relations among patches)
        # self.aggregator = nn.Linear(latent_dim, latent_dim)
        self.aggregator = Transformer2DWrapper(in_channels=latent_dim, out_channels=latent_dim, patch_size=patch_size,
                                               norm_type='ada_norm_single',
                                               sample_size=(image_size[-1] // patch_size) ** 2)

        # Decoder head that turns a latent vector -> local patch representation
        # We'll decode to a hidden_dim feature map, then do a final conv to get 3 channels
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim * (patch_size // 1) * (patch_size // 1))
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, in_channels, kernel_size=3, padding=1),
        )

        self.final_projection = ResidualConvLayer(3, 3)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns: reconstruction of x with shape (B, 3, H, W)
        """
        B, C, H, W = x.shape

        # -- 1) Split the image into patches
        patches = self.patchify(x)  # shape: (B, N, C, p, p)
        B_, N, C_, p, p_ = patches.shape

        # Flatten patches into a single batch dimension for parallel encoding
        # patches = patches.view(B_ * N, C_, p, p_)
        patches = rearrange(patches, 'b n c p1 p2 -> (b n) c p1 p2')

        # -- 2) Encode each patch into a latent vector
        patch_latents = self.encoder(patches)  # (B*N, latent_dim)

        # -- 3) Aggregate or transform latents (optional)
        patch_latents = self.aggregator(patch_latents)  # (B*N, latent_dim)
        patch_latents = F.relu(patch_latents)

        # -- 4) Decode each patch from latent
        #     decode into a hidden_dim × p × p feature map
        dec_input = self.decoder_fc(patch_latents)  # shape: (B*N, hidden_dim * p * p)
        # dec_input = dec_input.view(B * N, -1, p, p)  # reshape to (B*N, hidden_dim, p, p)
        dec_input = rearrange(dec_input, 'BN (hidden p1 p2) -> BN hidden p1 p2', p1=p, p2=p)

        #     apply final conv transpose to get 3 channels
        reconstructed_patches = self.decoder_cnn(dec_input)  # (B*N, 3, p, p)

        # -- 5) Fold the reconstructed patches back into an image
        reconstructed_patches = reconstructed_patches.view(B_, N, C, p, p)
        x_hat = self.unpatchify(reconstructed_patches)  # (B, 3, H, W)

        x_hat = self.final_projection(x_hat)

        return x + x_hat


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('config_file')
    adrgs = parser.parse_args()

    with open(adrgs.config_file, 'r') as f:
        config = OmegaConf.load(f)

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    np.random.seed(0)

    key_frames_dir = config['key_frames_dir']
    frames_dir = config['frames_dir']

    # Additional validation data if valid exists
    data_root_valid = None
    if os.path.exists(os.path.join(config['key_frames_dir'], 'valid')):
        data_root_valid = os.path.join(config['key_frames_dir'], 'valid')

    # probe size
    data_aux_probe = InferDataset(frames_dir, lambda x: x)
    data_train_probe = TrainingDataset(frames_dir, key_frames_dir, lambda x: x, data_aux_probe,
                                       disable_augment=config['disable_augment'])

    size = None
    for pair in data_train_probe.pairs:
        x, y = pair
        if size is None: size = x.size
        if x.size != size:
            print("WARNING: One of the input images has different size.")
        if y.size != size:
            print("WARNING: One of the output images has different size")
    for im in data_aux_probe.tensors:
        if im.size != size:
            print("WARNING: One of the video frames has different size")

    del data_aux_probe
    del data_train_probe

    device = config['device']
    storage_to_cpu = False
    transform = ImageTensorConverter(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                                     resize=f'flex;8;max;{config["resize"]}' if config["resize"] is not None else f'flex;8',
                                     drop_alpha=True)
    model = ImageToImageGenerator_JohnsonFutschik(config=config, **config['model_params'])

    data_aux = InferDataset(frames_dir, transform)
    data_train = TrainingDataset(frames_dir, key_frames_dir, transform, data_aux,
                                 disable_augment=config['disable_augment'])
    data_validate = InferDataset(data_root_valid, transform) if data_root_valid is not None else None


    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    batch_size = config['batch_size']
    collate = lambda x: [torch.utils.data.dataloader.default_collate(x).to(device) for x in x]
    trainset = DataLoader(data_train, num_workers=0, worker_init_fn=worker_init_fn)
    auxset = DataLoader(data_aux, num_workers=0, worker_init_fn=worker_init_fn, batch_size=batch_size, drop_last=False)
    testset = DataLoader(data_validate, num_workers=0) if data_validate is not None else None

    key_weight = config['key_weight']
    style_weight = config['style_weight']
    structure_weight = config['structure_weight']

    log = TensorboardLogger(config['logdir'], checkpoint_fmt='checkpoint_%s.pth')
    with open(os.path.join(log.location(), log.experiment_name() + '.yml'), 'w') as f:
        OmegaConf.save(config, f)

    layers = config['vgg_layers']

    similarity_loss = InnerProductLoss(layers, device)

    guidance_sd, processor = prepare_cldm(config)

    train_with_similarity(config, model, config['iters'], key_weight, style_weight, structure_weight,
                          trainset, auxset, testset,
                          transform, device, log)

