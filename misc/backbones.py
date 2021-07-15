import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as models
from torchvision.models import densenet as densenet_models, resnet as resnet_models

from misc.wide_resnet_28_10 import WideResNet28x10


class Extractor(nn.Module):
    def __init__(self, base_net='resnet_50', pretrained=True):
        super(Extractor, self).__init__()
        if base_net == 'resnet_50':
            basenet = models.resnet50(pretrained=pretrained)
            self.extractor = nn.Sequential(*list(basenet.children())[:-1])
            self.my_fwd = self.forward_res50

        elif base_net == 'wide_resnet_50':
            basenet = models.wide_resnet50_2(pretrained=pretrained)
            self.extractor = nn.Sequential(*list(basenet.children())[:-1])
            self.my_fwd = self.forward_res50

        elif base_net == 'wide_resnet_28x10':
            self.extractor = WideResNet28x10([4, 4, 4])
            self.my_fwd = self.forward_res50

        elif base_net == 'densenet_201':
            basenet = models.densenet201(pretrained=pretrained)
            self.extractor = basenet.features
            self.my_fwd = self.forward_dense201

        elif 'densenet_201_' in base_net:
            self.extractor = build_backbone(base_net, idx_end=base_net.split('_')[2])
            self.my_fwd = self.base_fwd

        elif 'resnet_101_' in base_net:
            self.extractor = build_backbone(base_net, idx_end=base_net.split('_')[2])
            self.my_fwd = self.base_fwd
        else:
            raise ValueError('unsupported backbone')

    def forward_res50(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return x

    def forward_dense201(self, x):
        features = self.extractor(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def base_fwd(self, x):
        return self.extractor(x)

    def forward(self, x):
        return self.my_fwd(x)


def build_backbone(features_type, idx_start=0, idx_end=-1, pretrained=True):
    """

    :param features_type:
    :param idx_start:
    :param idx_end: (including)
    :param pretrained:
    :return:
    """
    if features_type.find('ResNet_101') >= 0:
        raise ValueError(f'{features_type} backbone is not supported yet in this generator')

    elif features_type.find('densenet_201') >= 0:
        model_ref = densenet_models.densenet201(pretrained=pretrained)
        idx_end = {'T1': -6, 'D2': -5, 'T2': -4, 'D3': -3, 'T3': -2, 'D4': -1, -1: -1, 0: 0}[idx_end]
        idx_start = {'T1': -6, 'D2': -5, 'T2': -4, 'D3': -3, 'T3': -2, 'D4': -1, -1: -1, 0: 0}[idx_start]
        backbone = nn.Sequential(*list(model_ref.children())[:-1][0][idx_start:idx_end])
        backbone.eval().float()
        return backbone

    elif features_type.find('resnet') >= 0:
        if features_type.find('resnet_101') >= 0:
            model_ref = resnet_models.resnet101(pretrained=pretrained)
        if features_type.find('resnet_50') >= 0:
            model_ref = resnet_models.resnet50(pretrained=pretrained)
        idx_end = {'L1': -5, 'L2': -4, 'L3': -3, 'L4': -2, -1: -1, 0: 0}[idx_end]
        idx_start = {'L1': -5, 'L2': -4, 'L3': -3, 'L4': -2, -1: -1, 0: 0}[idx_start]
        backbone = nn.Sequential(*list(model_ref.children())[idx_start:idx_end])
        backbone.eval().float()
        return backbone
