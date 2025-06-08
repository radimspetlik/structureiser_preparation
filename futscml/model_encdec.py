import yaml
from torchvision.models import VGG19_Weights

from futscml import *
import random
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch.autograd.profiler as profiler
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.io
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import functional as Tfunc
from contextlib import suppress as suppress

from argparse import ArgumentParser
from tqdm import tqdm

from stalp import ViTFeatureExtractor_ViT_B_32, ViTFeatureExtractor_ViT_L_16
from einops import repeat


class ImageToImageGenerator_JohnsonFutschik(nn.Module):
    def __init__(self, norm_layer='batch_norm', use_bias=False, resnet_blocks=9, tanh=False,
                 filters=(64, 128, 128, 128, 128, 128, 128, 64), input_channels=3, output_channels=3,
                 append_blocks=None,
                 config=None):
        super().__init__()
        assert norm_layer in [None, 'batch_norm', 'instance_norm']
        self.norm_layer = None
        if norm_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            self.norm_layer = nn.InstanceNorm2d
        self.use_bias = use_bias
        self.config = config
        self.use_attention = config['use_attention']
        self.resnet_blocks = resnet_blocks
        self.append_blocks = append_blocks

        self.conv0 = self.relu_layer(in_filters=input_channels, out_filters=filters[0],
                                     size=7, stride=1, padding=3, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2))

        self.conv1 = self.relu_layer(in_filters=filters[0], out_filters=filters[1],
                                     size=3, stride=2, padding=1, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2))

        self.conv2 = self.relu_layer(in_filters=filters[1], out_filters=filters[2],
                                     size=3, stride=2, padding=1, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2))


        self.conv3 = self.relu_layer(in_filters=filters[2], out_filters=filters[3],
                                     size=3, stride=2, padding=1, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2))

        self.resnets = nn.ModuleList()
        for i in range(self.resnet_blocks):
            self.resnets.append(
                self.resnet_block(in_filters=filters[3], out_filters=filters[3],
                                  size=3, stride=1, padding=1, bias=self.use_bias,
                                  norm_layer=self.norm_layer, nonlinearity=nn.ReLU()))

            if self.use_attention:
                self.resnets.append(SpatialTransformer(filters[3], 4, filters[3] // 4,
                                               use_positional_encoding=config['use_positional_encoding'] if 'use_positional_encoding' in config else False,
                                               context_dim=config['attention_context_dim'] if 'attention_context_dim' in config else None))

        self.upconv3 = self.upconv_layer(in_filters=filters[len(filters) - 4] + filters[3], out_filters=filters[len(filters) - 2],
                                         size=4, stride=2, padding=1, bias=self.use_bias,
                                         norm_layer=self.norm_layer, nonlinearity=nn.ReLU())

        self.upconv2 = self.upconv_layer(in_filters=filters[len(filters) - 3] + filters[2], out_filters=filters[len(filters) - 2],
                                         size=4, stride=2, padding=1, bias=self.use_bias,
                                         norm_layer=self.norm_layer, nonlinearity=nn.ReLU())


        self.upconv1 = self.upconv_layer(in_filters=filters[len(filters) - 2] + filters[1], out_filters=filters[len(filters) - 2],
                                         size=4, stride=2, padding=1, bias=self.use_bias,
                                         norm_layer=self.norm_layer, nonlinearity=nn.ReLU())

        self.conv_11 = nn.Sequential(
            nn.Conv2d(filters[0] + filters[len(filters) - 2] + input_channels, filters[len(filters) - 1],
                      kernel_size=7, stride=1, padding=3, bias=self.use_bias),
            nn.ReLU()
        )

        # initialize context to gaussian random noise
        self.context = torch.randn(1, 100, config['attention_context_dim'] if 'attention_context_dim' in config else 1)

        self.end_blocks = None
        if self.append_blocks is not None:
            self.end_blocks = nn.Sequential(
                nn.Conv2d(filters[len(filters) - 1], filters[len(filters) - 1], kernel_size=3, bias=self.use_bias, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=filters[5]),
                nn.Conv2d(filters[len(filters) - 1], filters[len(filters) - 1], kernel_size=3, bias=self.use_bias, padding=1),
                nn.ReLU()
            )

        self.conv_12 = nn.Sequential(nn.Conv2d(filters[len(filters) - 1], output_channels, kernel_size=1, stride=1, padding=0, bias=True))
        if tanh:
            self.conv_12.add_module('tanh', nn.Tanh())

    def forward(self, x):
        output_0 = self.conv0(x)
        output_1 = self.conv1(output_0)
        output_2 = self.conv2(output_1)
        output = self.conv3(output_1)

        for layer in self.resnets:
            output = layer(output) + output

        output = self.upconv3(torch.cat((output, output_2), dim=1))
        output = self.upconv2(torch.cat((output, output_2), dim=1))
        output = self.upconv1(torch.cat((output, output_1), dim=1))
        output = self.conv_11(torch.cat((output, output_0, x), dim=1))
        if self.end_blocks is not None:
            output = self.end_blocks(output)
        output = self.conv_12(output)
        return output

    def relu_layer(self, in_filters, out_filters, size, stride, padding, bias,
                   norm_layer, nonlinearity):
        out = []
        out.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                             kernel_size=size, stride=stride, padding=padding, bias=bias))
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

    def upconv_layer(self, in_filters, out_filters, size, stride, padding, bias,
                     norm_layer, nonlinearity):
        out = []
        out.append(SmoothUpsampleLayer(in_filters, out_filters))
        if norm_layer:
            out.append(norm_layer(num_features=out_filters))
        if nonlinearity:
            out.append(nonlinearity)
        return nn.Sequential(*out)