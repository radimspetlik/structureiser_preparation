import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict

class HighresNet(nn.Module):
    def __init__(self, outputs=1000, bn=True):
        super().__init__()
        self.bn = bn
        self.highres_feat = nn.Sequential()
        self.highres_feat.add_module(f'block_{0:02d}', self.make_conv_block(f'conv_{0:02d}', 3, 64))
        for layer in range(1, 12):
            self.highres_feat.add_module(f'block_{layer:02d}', self.make_conv_block(f'conv_{layer:02d}', 64, 64))
        self.highres_feat.add_module(f'block_{12:02d}', self.make_conv_block(f'conv_{12:02d}', 64, 128))
        for layer in range(13, 20):
            self.highres_feat.add_module(f'block_{layer:02d}', self.make_conv_block(f'conv_{layer:02d}', 128, 128))
        self.highres_feat.add_module(f'block_{20:02d}', self.make_conv_block(f'conv_{20:02d}', 128, 192))
        for layer in range(21, 30):
            self.highres_feat.add_module(f'block_{layer:02d}', self.make_conv_block(f'conv_{layer:02d}', 192, 192))
        
        self.downpool0 = nn.AdaptiveMaxPool2d((64, 64))
        self.lowres_feat = nn.Sequential()
        self.lowres_feat.add_module(f'low_block_{0:02d}', self.make_conv_block(f'lowconv_{0:02d}', 192, 512))
        for layer in range(1, 5):
            self.lowres_feat.add_module(f'low_block_{layer:02d}', self.make_conv_block(f'lowconv_{layer:02d}', 512, 512))
        self.downpool1 = nn.AdaptiveMaxPool2d((4, 4))
        self.classifier = nn.Sequential()
        self.classifier.add_module(f'fc0', nn.Linear(in_features=4*4*512, out_features=4096))
        self.classifier.add_module(f'fc0_r', nn.ReLU(inplace=True))
        self.classifier.add_module(f'fc1', nn.Linear(in_features=4096, out_features=4096))
        self.classifier.add_module(f'fc1_r', nn.ReLU(inplace=True))
        self.classifier.add_module(f'fc2', nn.Linear(in_features=4096, out_features=outputs))
        
    def forward(self, x):
        highres = self.highres_feat(x)
        pooled0 = self.downpool0(highres)
        lowres = self.lowres_feat(pooled0)
        pooled1 = self.downpool1(lowres)
        classified = self.classifier(pooled1.view(-1, 4*4*512))
        return classified
    
    def change_num_outputs(self, outputs):
        children = list(self.classifier.children())[:-1]
        assert len(children) == 4
        new_class = nn.Sequential()
        new_class.add_module(f'fc0', children[0])
        new_class.add_module(f'fc0_r', children[1])
        new_class.add_module(f'fc1', children[2])
        new_class.add_module(f'fc1_r', children[3])
        new_class.add_module(f'fc2', nn.Linear(in_features=4096, out_features=outputs))
        self.classifier = new_class
        return self
    
    def make_conv_block(self, name, cin, cout):
        if self.bn:
            block = nn.Sequential(
                OrderedDict([
                    (name, nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=3, padding=1, padding_mode='replicate')),
                    (f'{name}_bn', nn.BatchNorm2d(cout)),
                    (f'{name}_r', nn.ReLU(inplace=True)),
                ])
            )
        else:
            block = nn.Sequential(
                OrderedDict([
                    (name, nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=3, padding=1, padding_mode='replicate')),
                    (f'{name}_r', nn.ReLU(inplace=True)),
                ])
            )
        return block

class FeatureExtract_HighresNet(nn.Module):
    def __init__(self, bn=True):
        super().__init__()
        self.module = HighresNet(1000, bn)
    
    def load_canonical_state_dict(self, state_dict):
        if 'model.state_dict' in state_dict:
            state_dict = state_dict['model.state_dict']
        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            print(error)
            self.load_state_dict(state_dict, strict=False)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        
    def forward(self, x, capture = ['block_00', 'block_01', 'block_02', 'block_03', 'block_04', 'block_06', 'block_10', 'block_15', 'block_20', 'block_29', 'low_block_00', 'low_block_03']):
        out = OrderedDict()
        for name, child in self.module.highres_feat.named_children():
            x = child(x)
            if name in capture:
                # print(name)
                out[name] = x
            if list(out.keys()) == capture:
                return out
        x = self.module.downpool0(x)
        for name, child in self.module.lowres_feat.named_children():
            x = child(x)
            if name in capture:
                # print(name)
                out[name] = x
            if list(out.keys()) == capture:
                return out
        return out
        

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        features = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features['layer_1'] = x
        x = self.layer2(x)
        features['layer_2'] = x
        x = self.layer3(x)
        features['layer_3'] = x
        x = self.layer4(x)
        features['layer_4'] = x

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return features

    def forward(self, x):
        return self._forward_impl(x)


class FeatureExtract_ResNet(nn.Module):
    def __init__(self, model='resnet_50'):
        super().__init__()
        if model != 'resnet_50': raise ValueError(f'NYI {model}')
        self.model = ResNet(Bottleneck, [3, 4, 6, 3])

    def load_canonical_state_dict(self, state_dict):
        if 'model' in state_dict:
            state_dict = state_dict['model']
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('module.model.'):
                w = state_dict.pop(key)
                state_dict[key[len('module.model.'):]] = w
            if key.startswith('module.attacker.'):
                state_dict.pop(key)

        if 'fc.weight' in state_dict: state_dict.pop('fc.weight')
        if 'fc.bias' in state_dict: state_dict.pop('fc.bias')
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            print(error)
            self.model.load_state_dict(state_dict, strict=False)
        for param in self.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, x):
        return self.model(x)

class FireModule(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class FeatureExtract_SqueezeNet(nn.Module):
    def __init__(self, version='1_0'):
        super().__init__()
        self.version = version
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),              # conv_1
                nn.ReLU(inplace=True),                                  # relu_1
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # pool_1
                FireModule(96, 16, 64, 64),                             # fire_1
                FireModule(128, 16, 64, 64),                            # fire_2
                FireModule(128, 32, 128, 128),                          # fire_3
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # pool_2
                FireModule(256, 32, 128, 128),                          # fire_4
                FireModule(256, 48, 192, 192),                          # fire_5
                FireModule(384, 48, 192, 192),                          # fire_6
                FireModule(384, 64, 256, 256),                          # fire_7
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # pool_3
                FireModule(512, 64, 256, 256),                          # fire_8
            )
            self.layer_names = [
                'conv_1',
                'relu_1',
                'pool_1',
                'fire_1',
                'fire_2',
                'fire_3',
                'pool_2',
                'fire_4',
                'fire_5',
                'fire_6',
                'fire_7',
                'pool_3',
                'fire_8'
            ]
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),              # conv_1
                nn.ReLU(inplace=True),                                  # relu_1
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # pool_1
                FireModule(64, 16, 64, 64),                             # fire_1
                FireModule(128, 16, 64, 64),                            # fire_2
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # pool_2
                FireModule(128, 32, 128, 128),                          # fire_3
                FireModule(256, 32, 128, 128),                          # fire_4
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # pool_3
                FireModule(256, 48, 192, 192),                          # fire_5
                FireModule(384, 48, 192, 192),                          # fire_6
                FireModule(384, 64, 256, 256),                          # fire_7
                FireModule(512, 64, 256, 256),                          # fire_8
            )
            self.layer_names = [
                'conv_1',
                'relu_1',
                'pool_1',
                'fire_1',
                'fire_2',
                'pool_2',
                'fire_3',
                'fire_4',
                'pool_3',
                'fire_5',
                'fire_6',
                'fire_7',
                'fire_8',
            ]
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def load_canonical_state_dict(self, state_dict):
        if 'model' in state_dict:
            state_dict = state_dict['model']
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('module.model.'):
                w = state_dict.pop(key)
                state_dict[key[len('module.model.'):]] = w
            if key.startswith('module.attacker.'):
                state_dict.pop(key)

        if 'classifier.1.weight' in state_dict:
            state_dict.pop('classifier.1.weight')
        if 'classifier.1.bias' in state_dict:
            state_dict.pop('classifier.1.bias')
        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            print(error)
            self.load_state_dict(state_dict, strict=False)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        out = OrderedDict()
        for i in range(len(self.features)):
            x = self.features[i](x)
            out[self.layer_names[i]] = x
        return out


class FeatureExtract_VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def reset_parameters(self):
        for child in self.children():
            if len(list(child.parameters())) > 0:
                child.reset_parameters()

    def load_canonical_state_dict(self, state_dict):
        md = state_dict
        if 'model' in md:
            md = md['model']
        if 'conv1_1.weight' not in state_dict:
            def reassign(a, b):
                md[a + '.weight'] = md.pop(b + '.weight')
                md[a + '.bias'] = md.pop(b + '.bias')
            reassign('conv1_1', 'features.0')
            reassign('conv1_2', 'features.2')
            reassign('conv2_1', 'features.5')
            reassign('conv2_2', 'features.7')
            reassign('conv3_1', 'features.10')
            reassign('conv3_2', 'features.12')
            reassign('conv3_3', 'features.14')
            reassign('conv3_4', 'features.16')
            reassign('conv4_1', 'features.19')
            reassign('conv4_2', 'features.21')
            reassign('conv4_3', 'features.23')
            reassign('conv4_4', 'features.25')
            reassign('conv5_1', 'features.28')
            reassign('conv5_2', 'features.30')
            reassign('conv5_3', 'features.32')
            reassign('conv5_4', 'features.34')
            [md.pop(a) for a in ['classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias',
                                'classifier.6.weight', 'classifier.6.bias', 'classifier.1.weight', 'classifier.1.bias', 
                                 'classifier.4.weight', 'classifier.4.bias'] if a in md]
        try:
            self.load_state_dict(md, strict=True)
        except RuntimeError as error:
            print(error)
            self.load_state_dict(md, strict=False)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
            
    def forward(self, x):
        out = OrderedDict()
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return out
        