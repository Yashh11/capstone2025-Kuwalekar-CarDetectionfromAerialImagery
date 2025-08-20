import torch
import torch.nn as nn
from torchvision import models
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn','vgg19_bn', 'vgg19',]

#pretrained model urls
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

#localisation head
#custom model built on VGG backbone
class localizerVGG(nn.Module):
    def __init__(self, features, num_classes=1):
        super(localizerVGG, self).__init__()
        self.features = features
        self.bn2 = nn.BatchNorm2d(512)
        self.Alayer = self._make_Adapdation_layer(num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.bn2(x)
        x = self.Alayer(x)
        return x

    #small cnn head
    def _make_Adapdation_layer(self, num_class):
        layers = []
        layers.append(nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)))
        layers.append(nn.LeakyReLU(negative_slope=0.5, inplace=False))
        layers.append(nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1)))
        layers.append(nn.LeakyReLU(negative_slope=0.5, inplace=False))
        layers.append(nn.Conv2d(64, num_class, kernel_size=(1, 1), stride=(1, 1)))
        return nn.Sequential(*layers)

#replicates og VGG with classifier head for ImageNet
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#vgg architecture
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#generice vgg builder
def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

#vgg model variants
def vgg11(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)

def vgg11_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)

def vgg13(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)

def vgg13_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)

def vgg16(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)

#customised localiser vgg16
def localizervgg16(pretrained=False, progress=True, dsr=8, **kwargs):
    vgg = _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)

    #adjusting pooling layers for diff downsampling rates
    if dsr == 4:
        vgg.features[30] = nn.MaxPool2d(kernel_size=1, stride=1)
        vgg.features[23] = nn.MaxPool2d(kernel_size=1, stride=1)
        vgg.features[16] = nn.MaxPool2d(kernel_size=1, stride=1)
    elif dsr == 8:
        vgg.features[30] = nn.MaxPool2d(kernel_size=1, stride=1)
        vgg.features[23] = nn.MaxPool2d(kernel_size=1, stride=1)
        vgg.features[16] = nn.MaxPool2d(kernel_size=2, stride=2)
    else:
        vgg.features[30] = nn.MaxPool2d(kernel_size=1, stride=1)
        vgg.features[23] = nn.MaxPool2d(kernel_size=2, stride=2)
        vgg.features[16] = nn.MaxPool2d(kernel_size=2, stride=2)

    locVGG = localizerVGG(vgg.features, num_classes=1)
    return locVGG

def vgg16_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)

def vgg19(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)

def vgg19_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)